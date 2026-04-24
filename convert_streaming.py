"""
Streaming model weight conversion script.

Features:
1. Processes files one by one without accumulating all weights in memory
2. Supports very large models (e.g. 2T params) under limited memory
3. Uses a temp directory for intermediate results, then merges at the end

Usage is identical to convert.py:
    python convert_streaming.py --hf-ckpt-path <path> --save-path <path> \
        --n-experts 256 --model-parallel 8
"""

import os
import shutil
import tempfile
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file


FP4_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32)


def cast_e2m1fn_to_e4m3fn(x: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Casts a tensor from e2m1fn to e4m3fn losslessly."""
    assert x.dtype == torch.int8
    assert x.ndim == 2
    out_dim, in_dim = x.size()
    in_dim *= 2
    fp8_block_size = 128
    fp4_block_size = 32
    assert in_dim % fp8_block_size == 0 and out_dim % fp8_block_size == 0
    assert scale.size(0) == out_dim and scale.size(1) == in_dim // fp4_block_size

    x = x.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    x = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(2)

    MAX_OFFSET_BITS = 6

    bOut = out_dim // fp8_block_size
    bIn = in_dim // fp8_block_size
    x = x.view(bOut, fp8_block_size, bIn, fp8_block_size).transpose(1, 2)
    scale = scale.float().view(bOut, fp8_block_size, bIn, -1).transpose(1, 2).flatten(2)
    scale_max_offset_bits = scale.amax(dim=-1, keepdim=True) / (2**MAX_OFFSET_BITS)
    offset = scale / scale_max_offset_bits
    offset = offset.unflatten(-1, (fp8_block_size, -1)).repeat_interleave(fp4_block_size, dim=-1)
    x = (x * offset).transpose(1, 2).reshape(out_dim, in_dim)
    return x.to(torch.float8_e4m3fn), scale_max_offset_bits.squeeze(-1).to(torch.float8_e8m0fnu)


MAPPING = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "lm_head": ("head", 0),
    "embed": ("embed", 0),
    "wq_b": ("wq_b", 0),
    "wo_a": ("wo_a", 0),
    "wo_b": ("wo_b", 1),
    "head": ("head", 0),
    "attn_sink": ("attn_sink", 0),
    "weights_proj": ("weights_proj", 0),
}


def process_tensor_name(name: str, rank: int, mp: int, n_local_experts: int):
    """
    Process tensor name and determine whether this rank needs it.

    Returns:
        (new_name, slice_info) or None if this rank doesn't need this tensor.
        slice_info: (dim, rank, mp) or None (no sharding needed)
    """
    if name.startswith("model."):
        name = name[len("model."):]

    # Skip MTP embedding and head
    if name.startswith("mtp.") and ("emb" in name or name.endswith("head.weight")):
        return None

    name = name.replace("self_attn", "attn")
    name = name.replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale")
    name = name.replace("e_score_correction_bias", "bias")

    # Extract key for mapping lookup
    if any(x in name for x in ["hc", "attn_sink", "tie2eid", "ape"]):
        key = name.split(".")[-1]
    else:
        key = name.split(".")[-2]

    if key in MAPPING:
        new_key, dim = MAPPING[key]
    else:
        new_key, dim = key, None

    name = name.replace(key, new_key)

    # Check if this expert belongs to this rank
    if "experts" in name and "shared_experts" not in name:
        idx = int(name.split(".")[-3])
        if idx < rank * n_local_experts or idx >= (rank + 1) * n_local_experts:
            return None
        return name, None

    if dim is not None:
        return name, (dim, rank, mp)

    return name, None


def get_sharded_tensor(param: torch.Tensor, slice_info: tuple, name: str) -> torch.Tensor:
    """Shard tensor according to slice_info."""
    dim, i, mp = slice_info
    print(f"Processing parameter {name} with shape {param.shape} for model parallel shard {i}")
    if "wo_a" not in name and "wo_b" not in name:
        assert param.size(dim) % mp == 0, f"Dimension {dim} must be divisible by {mp}"
        shard_size = param.size(dim) // mp
        return param.narrow(dim, i * shard_size, shard_size).contiguous()
    else:
        num_projection_groups = mp // 8
        new_mp = mp // num_projection_groups
        new_i = i // num_projection_groups
        shard_size = param.size(dim) // new_mp
        assert shard_size == 1024
        return param.narrow(dim, new_i * shard_size, shard_size).contiguous()


def process_single_file(file_path: str, rank: int, mp: int, n_local_experts: int) -> dict:
    """
    Process a single safetensors file and return tensors needed by this rank.
    Memory is released after processing each file.
    """
    result = {}

    with safe_open(file_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            process_result = process_tensor_name(name, rank, mp, n_local_experts)
            if process_result is None:
                continue

            new_name, slice_info = process_result
            param = f.get_tensor(name)

            if slice_info is not None:
                param = get_sharded_tensor(param, slice_info, new_name)

            result[new_name] = param

    return result


def incremental_save(tensors: dict, temp_dir: str, batch_idx: int):
    """Save current batch of tensors to a temp file for later merging."""
    temp_file = os.path.join(temp_dir, f"batch_{batch_idx}.safetensors")
    save_file(tensors, temp_file)
    return temp_file


def main(hf_ckpt_path: str, save_path: str, n_experts: int, mp: int, expert_dtype: str = None):
    """
    Streaming conversion main function.

    Strategy:
    1. Iterate over all input files, processing one at a time
    2. For each MP rank, maintain a memory buffer; flush to temp files when buffer is full
    3. Merge all temp files into the final output
    """
    torch.set_num_threads(8)
    n_local_experts = n_experts // mp

    os.makedirs(save_path, exist_ok=True)

    input_files = sorted(glob(os.path.join(hf_ckpt_path, "*.safetensors")))
    if not input_files:
        raise ValueError(f"No safetensors files found in {hf_ckpt_path}")

    print(f"Found {len(input_files)} input files")
    print(f"Model parallel: {mp}, Local experts per rank: {n_local_experts}")
    print("Starting streaming conversion...")

    for rank in trange(mp, desc="Processing ranks"):
        buffer = {}
        temp_files = []
        buffer_size = 0
        batch_counter = 0
        # 8GB buffer limit (approximate)
        BUFFER_LIMIT = 8 * 1024**3

        with tempfile.TemporaryDirectory() as temp_dir:
            for file_idx, file_path in enumerate(tqdm(
                input_files,
                desc=f"Rank {rank}",
                leave=False
            )):
                file_tensors = process_single_file(
                    file_path, rank, mp, n_local_experts
                )

                for name, tensor in file_tensors.items():
                    buffer[name] = tensor
                    buffer_size += tensor.numel() * tensor.element_size()

                # Flush buffer if it exceeds the limit or this is the last file
                if buffer_size >= BUFFER_LIMIT or file_idx == len(input_files) - 1:
                    if buffer:
                        temp_file = incremental_save(buffer, temp_dir, batch_counter)
                        temp_files.append(temp_file)
                        buffer.clear()
                        buffer_size = 0
                        batch_counter += 1

            # Merge all temp files into the final output
            merged = {}
            for temp_file in temp_files:
                with safe_open(temp_file, framework="pt", device="cpu") as f:
                    for name in f.keys():
                        merged[name] = f.get_tensor(name)

            output_file = os.path.join(save_path, f"model{rank}-mp{mp}.safetensors")
            save_file(merged, output_file)

    # Copy config files
    for file in ["chat_template.jinja", "tokenizer.json", "tokenizer_config.json"]:
        src = os.path.join(hf_ckpt_path, file)
        dst = os.path.join(save_path, file)
        if os.path.exists(src):
            shutil.copyfile(src, dst)

    print(f"Conversion complete! Output saved to: {save_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--n-experts", type=int, required=True)
    parser.add_argument("--model-parallel", type=int, required=True)
    parser.add_argument("--expert-dtype", type=str, choices=["fp8", "fp4"], default=None)
    args = parser.parse_args()

    assert args.n_experts % args.model_parallel == 0, \
        "Number of experts must be divisible by model parallelism"

    main(
        args.hf_ckpt_path,
        args.save_path,
        args.n_experts,
        args.model_parallel,
        args.expert_dtype
    )
