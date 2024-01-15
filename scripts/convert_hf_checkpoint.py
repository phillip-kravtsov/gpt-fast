# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import re
import sys
from pathlib import Path
from typing import Optional

import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import ModelArgs


@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path = Path(
        "checkpoints/meta-Transformer/Transformer-2-7b-chat-hf"
    ),
    model_name: Optional[str] = None,
) -> None:
    if model_name is None:
        model_name = checkpoint_dir.name

    config = ModelArgs.from_name(model_name)
    print(f"Model config {config.__dict__}")

    # Load the json file containing weight mapping
    model_map_json = checkpoint_dir / "model.safetensors.index.json"

    if model_map_json.is_file():
        with open(model_map_json) as json_map:
            bin_index = json.load(json_map)
        bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}
    else:
        if (checkpoint_dir / "pytorch_model.bin").is_file():
            bin_files = {checkpoint_dir / "pytorch_model.bin"}
        else:
            bin_files = {checkpoint_dir / "model.safetensors"}

    weight_map = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
        "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
        "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
        "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
        "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
        "model.layers.{}.block_sparse_moe.experts.{}.w{}.weight": "layers.{}.block_sparse_moe.experts.{}.w{}.weight",
        "model.layers.{}.block_sparse_moe.gate.weight": "layers.{}.block_sparse_moe.gate.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }

    def permute(w, n_head):
        dim = config.dim
        return (
            w.view(n_head, 2, config.head_dim // 2, dim)
            .transpose(1, 2)
            .reshape(config.head_dim * n_head, dim)
        )

    merged_result = {}
    for file in sorted(bin_files):
        from safetensors.torch import load_file

        if str(file).endswith("safetensors"):
            state_dict = load_file(str(file), device="cpu")
        else:
            print(str(file))
            state_dict = torch.load(str(file), map_location="cpu", mmap=True)
        merged_result.update(state_dict)
    update_keys = {}
    for key, value in merged_result.items():
        if key.startswith('module'):
            update_keys[key] = key[7:]
    if len(update_keys):
        for key, new_key in update_keys.items():
            merged_result[new_key] = merged_result[key]
            del merged_result[key]

    final_result = {}
    for key, value in merged_result.items():
        if "layers" in key:
            abstract_key = re.sub(r"(\d+)", "{}", key)
            layer_num = re.search(r"\d+", key).group(0)
            if abstract_key in weight_map.values():
                continue
            new_key = weight_map[abstract_key]
            if new_key is None:
                continue
            if "experts" in new_key:
                layer_num, expert_num, linear_num = re.findall(r"\d+", key)
                new_key = new_key.format(layer_num, expert_num, linear_num)
            else:
                new_key = new_key.format(layer_num)
        else:
            if key in weight_map.values():
                continue
            new_key = weight_map[key]

        final_result[new_key] = value

    for key in tuple(final_result.keys()):
        if "wq" in key:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            v = final_result[key.replace("wq", "wv")]
            q = permute(q, config.n_head)
            k = permute(k, config.n_local_heads)
            final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v])
            del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]
    print(f"Saving checkpoint to {checkpoint_dir / 'model.pth'}")
    torch.save(final_result, checkpoint_dir / "model.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert HuggingFace checkpoint.")
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("checkpoints/meta-llama/llama-2-7b-chat-hf"),
    )
    parser.add_argument("--model_name", type=str, default=None)

    args = parser.parse_args()
    convert_hf_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
    )
