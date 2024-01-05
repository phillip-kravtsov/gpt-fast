# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import torch.distributed as dist
import torch._dynamo
from torch.distributed import _functional_collectives as funcol


import numpy as np


def promote_scalar(x: torch.Tensor) -> torch.Tensor:
    return x.view(1) if len(x.size()) == 0 else x


try:
    import megablocks.ops as ops
except ImportError as e:
    print(e)
    print(
        "MegaBlocks not found, please see "
        "https://github.com/stanford-futuredata/megablocks/. "
        "Note that MegaBlocks depends on mosaicml-turbo, which only "
        "supports python 3.10."
    )
try:
    import stk
except ImportError:
    print("STK not found: please see https://github.com/stanford-futuredata/stk")


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    num_local_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [
            config
            for config in transformer_configs
            if config in str(name).upper() or config in str(name)
        ]
        assert len(config) == 1, name
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(
        block_size=16384, vocab_size=32000, n_layer=32, dim=4096, rope_base=1000000
    ),
    "TinyLlama-1.1B-Chat-v1.0": dict(
        block_size=2048,
        vocab_size=32000,
        dim=2048,
        n_layer=22,
        n_head=32,
        n_local_heads=4,
        intermediate_size=5632,
    ),
    "TinyLlama-1.1B-intermediate-step-480k-1T": dict(
        block_size=2048,
        vocab_size=32000,
        dim=2048,
        n_layer=22,
        n_head=32,
        n_local_heads=4,
        intermediate_size=5632,
    ),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(
        n_layer=48,
        n_head=64,
        dim=8192,
        vocab_size=32000,
        n_local_heads=8,
        intermediate_size=22016,
        rope_base=1000000,
    ),  # CodeLlama-34B-Python-hf
    "70B": dict(
        n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672
    ),
    "Mixtral-8x7B-Instruct-v0.1": dict(
        block_size=4096,
        n_layer=32,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        num_local_experts=8,
        num_experts_per_tok=2,
        rope_base=1000000,
    ),
}


class KVCache(nn.Module):
    def __init__(
        self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val, batch_index=None):
        # Either keep the current design and pass in how to slice
        # across different batches
        # or--pass in the KV cache and add functionality to get
        # and combine
        # KVCaches = model.setup_caches()
        # model.forward(input_pos, attention_mask, kv_caches=KVCaches)
        # slice = KVCaches.get(i)
        # Or keep as is and update input pos
        assert input_pos.shape[1] == k_val.shape[2], f"{input_pos.shape}, {k_val.shape}"

        k_out = self.k_cache
        v_out = self.v_cache

        if batch_index is not None:
            k_out[batch_index, :, input_pos.squeeze(0)] = k_val
            v_out[batch_index, :, input_pos.squeeze(0)] = v_val
            return k_out[batch_index : batch_index + 1], v_out[
                batch_index : batch_index + 1
            ]
        else:
            # input_pos is of shape (B, T)
            # kv_val is (B, NH, T, H)
            # kv_out is (B, NH, S, H)
            B = input_pos.size(0)
            # TODO: this should be correctly torchified
            for i in range(B):
                k_out[i, :, input_pos[i]] = k_val[i]
                v_out[i, :, input_pos[i]] = v_val[i]
            return k_out, v_out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        Block = (
            MixtralBlock if config.num_local_experts is not None else TransformerBlock
        )
        self.layers = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length):
        if (
            self.max_seq_length >= max_seq_length
            and self.max_batch_size >= max_batch_size
        ):
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.config.n_local_heads, head_dim
            )

        self.freqs_cis = precompute_freqs_cis(
            self.config.block_size,
            self.config.dim // self.config.n_head,
            self.config.rope_base,
        )
        self.causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        )

    def forward(
        self, idx: Tensor, input_pos: Optional[Tensor] = None, batch_index=None
    ) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[input_pos]
        mask = mask[:, None, :, :]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask, batch_index)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        x: Tensor,
        input_pos: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        batch_index,
    ) -> Tensor:
        h = x + self.attention(
            self.attention_norm(x), freqs_cis, mask, input_pos, batch_index
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class MixtralBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        # self.block_sparse_moe = BlockSparseMoE(config)
        self.block_sparse_moe = MixtralSparseMoeBlock(config)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor
    ) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.block_sparse_moe(self.ffn_norm(h))
        return out


class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hidden_dim = config.dim
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        assert self.num_experts is not None and self.top_k is not None
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        # self.register_buffer("eye", torch.eye(self.num_experts))

        self.experts = nn.ModuleList(
            [FeedForward(config) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        eye = torch.eye(self.num_experts).to(selected_experts.device)
        expert_mask = eye[selected_experts].permute(2, 1, 0)

        # expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer(current_state)
                * routing_weights[top_x_list, idx_list, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states


class MixtralAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        kv_seq_len = k.size(1)
        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
        )

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)
        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
        batch_index=None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v, batch_index)

        # print(f"After cache, k shape: {k.shape}, v shape: {v.shape}")
        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        # print(f"Pre-SDPA, k shape: {k.shape}, v shape: {v.shape} mask shape {mask.shape}")
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y


class BlockSparseMoE(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hidden_dim = config.dim
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        assert self.num_experts is not None
        self.top_k = config.num_experts_per_tok

        # gating
        self.ffn_dim_per_partition = self.ffn_dim
        self.gate = nn.Linear(
            self.hidden_dim, self.num_experts, bias=False, device="meta"
        )

        # merged expert weights, all of size  (ffn_dim * n_experts, model_dim)
        self.w1 = nn.Parameter(
            torch.empty(
                self.ffn_dim_per_partition * self.num_experts,
                self.hidden_dim,
                device="meta",
            )
        )
        self.w2 = nn.Parameter(
            torch.empty(
                self.ffn_dim_per_partition * self.num_experts,
                self.hidden_dim,
                device="meta",
            )
        )
        self.w3 = nn.Parameter(
            torch.empty(
                self.ffn_dim_per_partition * self.num_experts,
                self.hidden_dim,
                device="meta",
            )
        )

        # Calculate the number of bits needed to represent the expert indices
        # so that we can pass it to radix sort.
        self.sort_end_bit = max(int(np.ceil(np.log2(self.num_experts))), 1)
        self.blocking = 128
        self.quantize_scatter_num_bits = -1

        # Calculate the number of bits needed to represent the column indices
        # in the intermediate sparse matrix.
        max_column_index = (self.ffn_dim * self.num_experts) // self.blocking
        self.transpose_sort_end_bit = max(int(np.ceil(np.log2(max_column_index))), 1)

    def sparse_transpose(
        self, size: int, row_indices, column_indices
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_columns = size[1] // self.blocking

        # Sort row indices by column indices to get the transposed matrix's
        # column indices.
        #
        # NOTE: Our sort operation uses the same width indices as the input
        # values. To avoid overflow when we have large activation matrices
        # we cast to 32-bit before sorting.
        _, gather_indices = ops.sort(column_indices.int(), self.transpose_sort_end_bit)

        # There are a constant number of blocks in every row of the sparse
        # matrix. A blocks offset is:
        #
        # row_index * blocks_per_row + column_index % blocks_per_row
        #
        # Once we have the block offsets ordered for transposition we can
        # divide by blocks_per_row to get the transposed column indices.
        column_indices_t = row_indices.gather(0, gather_indices.long())
        block_offsets_t = gather_indices.int()

        zero = torch.zeros((1,), dtype=torch.int32, device=row_indices.device)
        nnz_per_column = ops.histogram(column_indices, block_columns)
        nnz_per_column = ops.inclusive_cumsum(nnz_per_column, 0)
        offsets_t = torch.cat([zero, nnz_per_column])
        return column_indices_t, offsets_t, block_offsets_t

    def topology(self, x: torch.Tensor, padded_bins: torch.Tensor):  # -> stk.Matrix:
        padded_tokens, _ = x.size()
        assert padded_tokens % self.blocking == 0
        assert self.ffn_dim_per_partition % self.blocking == 0

        # Offsets for the sparse matrix. All rows have the
        # same number of nonzero blocks dictated by the
        # dimensionality of a single expert.
        block_rows = padded_tokens // self.blocking
        blocks_per_row = self.ffn_dim_per_partition // self.blocking
        offsets = torch.arange(
            0,
            block_rows * blocks_per_row + 1,
            blocks_per_row,
            dtype=torch.int32,
            device=x.device,
        )

        # Indices for the sparse matrix. The indices for
        # the intermediate matrix are dynamic depending
        # on the mapping of tokens to experts.
        column_indices = ops.topology(
            padded_bins, self.blocking, block_rows, blocks_per_row
        )

        # TODO(tgale): This is unused. Remove the need for this in stk.
        # For now, use meta init to save the device memory.
        data = torch.empty(
            column_indices.numel(),
            self.blocking,
            self.blocking,
            dtype=x.dtype,
            device=torch.cuda.current_device(),
        )
        shape = (padded_tokens, self.ffn_dim_per_partition * self.num_experts)
        row_indices = stk.ops.row_indices(shape, data, offsets, column_indices)
        column_indices_t, offsets_t, block_offsets_t = self.sparse_transpose(
            shape, row_indices, column_indices
        )
        return stk.Matrix(
            shape,
            data,
            row_indices,
            column_indices,
            offsets,
            column_indices_t,
            offsets_t,
            block_offsets_t,
        )

    def indices_and_padded_bins(
        self, selected_experts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        selected_experts = selected_experts.int()
        bin_ids, indices = ops.sort(selected_experts, self.sort_end_bit)

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        tokens_per_expert = ops.histogram(selected_experts, self.num_experts)

        # Round the token counts up to the block size used in
        # the matrix muliplications. Caculate the starting
        # position of each bin.
        padded_tokens_per_expert = ops.round_up(tokens_per_expert, self.blocking)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
        padded_bins = promote_scalar(padded_bins)

        # Calculate the bin bounds for the sorted tokens.
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = promote_scalar(bins)
        return indices, bin_ids, bins, padded_bins, tokens_per_expert

    @torch._dynamo.disable()
    def suffix(self, x, weights, selected_experts):
        (indices, bin_ids, bins, padded_bins, _) = self.indices_and_padded_bins(
            selected_experts
        )

        # Permute tokens and pad to prepare expert computation
        # (top_k * sequence_length + padding, model_dim)
        x = ops.padded_gather(x, indices, bin_ids, bins, padded_bins, self.top_k)

        # Create the sparse matrix topology
        topo = self.topology(x, padded_bins)

        # Perform the expert computation
        # First Dense x Dense -> Sparse for w1 and w3,
        # (top_k * sequence_length + padding, ffn_dim * n_experts)
        x = stk.Matrix(
            topo.size(),
            F.silu(stk.ops.sdd(x, self.w1.t(), topo).data)
            * stk.ops.sdd(x, self.w3.t(), topo).data,
            topo.row_indices,
            topo.column_indices,
            topo.offsets,
            topo.column_indices_t,
            topo.offsets_t,
            topo.block_offsets_t,
        )

        # Then Sparse x Dense -> Dense for w2
        # (top_k * sequence_length + padding, model_dim)
        x = stk.ops.dsd(x, self.w2)
        dist.all_reduce(x)
        # x = funcol.all_reduce(x, reduceOp='sum', group=list(range(2)))

        # Permute back and remove padding
        # (top_k * sequence_length, model_dim)
        x = ops.padded_scatter(
            x,
            indices,
            bin_ids,
            weights,
            bins,
            padded_bins,
            self.top_k,
            self.quantize_scatter_num_bits,
        )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (sequence_length, model_dim)
        gate_logits: (sequence_length, n_experts)
        """
        # optional reshape
        input_shape = x.shape
        x = x.view(-1, input_shape[-1]).clone()

        # gate_logits: (sequence_length, n_experts)
        gate_logits = self.gate(x)
        # all_probs: (sequence_length, n_experts) and upcast for softmax
        all_probs = F.softmax(gate_logits, dim=1, dtype=torch.float)
        # weights, selected_experts: (sequence_length, top-k)
        weights, selected_experts = torch.topk(all_probs, self.top_k, dim=-1)
        weights /= weights.sum(dim=-1, keepdim=True)
        weights = weights.flatten()
        # weights = weights.to(x.dtype)
        selected_experts = selected_experts.flatten()
        x = self.suffix(x, weights, selected_experts)
        return x.view(*input_shape)


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(xshaped.size(0), xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
