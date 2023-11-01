# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from loguru import logger
import torch
import torch.distributed
import intel_extension_for_pytorch as ipex
import re
from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from typing import Optional, List, Tuple

# Flash attention imports
if torch.cuda.is_available():
    import dropout_layer_norm

from text_generation_server.utils import paged_attention, flash_attn
from text_generation_server.utils.flash_attn import attention, ref_reshape_and_cache, ref_single_query_cached_kv_attention
from text_generation_server.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    PositionRotaryEmbedding,
    TensorParallelHead,
    get_linear,
)


class LlamaConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_scaling=None,
        rope_theta=10000.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class LlamaRMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual=None):
        if residual is not None:
            hidden_states += residual
        residual = hidden_states
        return torch.ops.torch_ipex.rmsnorm(hidden_states, self.weight.to(hidden_states.dtype), self.variance_epsilon), residual

        # if hidden_states.shape[-1] > 8192 or not torch.cuda.is_available():
        #     if residual is not None:
        #         hidden_states += residual
        #     residual = hidden_states

        #     hidden_states = hidden_states.to(torch.float32)
        #     variance = hidden_states.pow(2).mean(-1, keepdim=True)
        #     hidden_states = hidden_states * torch.rsqrt(
        #         variance + self.variance_epsilon
        #     )

        #     # convert into half-precision if necessary
        #     if self.weight.dtype in [torch.float16, torch.bfloat16]:
        #         hidden_states = hidden_states.to(self.weight.dtype)

        #     return self.weight * hidden_states, residual
        # else:
        #     # faster post attention rms norm
        #     normed_hidden_states, res, *rest = dropout_layer_norm.dropout_add_ln_fwd(
        #         hidden_states,
        #         residual,
        #         self.weight,
        #         None,
        #         None,
        #         None,
        #         None,
        #         None,
        #         0.0,
        #         self.variance_epsilon,
        #         1.0,
        #         0,
        #         None,
        #         False,
        #         True,  # Activate RMSNorm
        #     )
        #     if res is None:
        #         res = hidden_states

        #     return normed_hidden_states, res


def load_attention(config, prefix, weights):
    if config.num_attention_heads != config.num_key_value_heads:
        return _load_gqa(config, prefix, weights)
    else:
        if config.model_type == "baichuan":
            return TensorParallelColumnLinear.load_qkv(
                config,
                prefix=f"{prefix}.W_pack",
                weights=weights,
                bias=False,
            )
        else:
            return TensorParallelColumnLinear.load_multi(
                config,
                prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
                dim=0,
                weights=weights,
                bias=False,
            )


def _load_gqa(config, prefix: str, weights):
    assert config.hidden_size % config.num_attention_heads == 0
    assert config.num_attention_heads % weights.process_group.size() == 0

    weight = weights.get_multi_weights_col(
        prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
        quantize=config.quantize,
        dim=0,
    )

    if config.quantize not in ["gptq", "awq"]:
        weight = weight.to(dtype=weights.dtype).to(device=weights.device)

        head_size = config.hidden_size // config.num_attention_heads
        num_heads = config.num_attention_heads // weights.process_group.size()
        num_key_value_heads = config.num_key_value_heads // weights.process_group.size()
        assert list(weight.shape) == [
            (num_heads + 2 * num_key_value_heads) * head_size,
            config.hidden_size,
        ], f"{list(weight.shape)} != {[(num_heads + 2 * config.num_key_value_heads) * head_size, config.hidden_size]}"

    return TensorParallelColumnLinear(
        get_linear(weight, bias=None, quantize=config.quantize)
    )

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, max_position_embeddings, dim, backbone, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached,
            dtype=self.inv_freq.dtype,
        )
        self.model_backbone = str(backbone)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        if re.search("falcon", str(backbone), re.IGNORECASE) or re.search(
            "rw", str(backbone), re.IGNORECASE
        ):
            self.sin_cos = torch.cat(
                (freqs.sin().repeat(1, 2), freqs.cos().repeat(1, 2)), dim=-1
            )
            self.emb = torch.cat((freqs, freqs), dim=-1).float()
            self.cos_cached = self.emb.cos()[None, :, :]
            self.sin_cached = self.emb.sin()[None, :, :]
        else:
            self.sin_cos = torch.cat((torch.sin(freqs), torch.cos(freqs)), dim=1)
            self.emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer(
                "cos_cached", self.emb.cos()[None, None, :, :], persistent=False
            )
            self.register_buffer(
                "sin_cached", self.emb.sin()[None, None, :, :], persistent=False
            )

    def forward(self, seq_len=None):
        if seq_len is not None and seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            if re.search("falcon", self.model_backbone, re.IGNORECASE) or re.search(
                "rw", self.model_backbone, re.IGNORECASE
            ):
                self.sin_cos = torch.cat(
                    (freqs.sin().repeat(1, 2), freqs.cos().repeat(1, 2)), dim=-1
                )
                self.emb = torch.cat((freqs, freqs), dim=-1).float()
                self.cos_cached = self.emb.cos()[None, :, :]
                self.sin_cached = self.emb.sin()[None, :, :]
            else:
                self.sin_cos = torch.cat((torch.sin(freqs), torch.cos(freqs)), dim=1)
                self.emb = torch.cat((freqs, freqs), dim=-1)
                self.cos_cached = self.emb.cos()[None, None, :, :]
                self.sin_cached = self.emb.sin()[None, None, :, :]
                self.cos_cached[:, :, :seq_len, ...]
                self.sin_cached[:, :, :seq_len, ...]
        return self.sin_cos, self.sin_cached, self.cos_cached

class _IPEXRopeCPU(nn.Module):
    def __init__(
        self,
        max_position_embeddings,
        pos_embd_dim,
        base=10000,
        backbone=None,
    ):
        super().__init__()
        self.embed_positions = RotaryEmbedding(
            max_position_embeddings, pos_embd_dim, backbone, base
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        num_head: int,
        head_size: int,
        offset: int,
        rotary_ndims: int,
        seq_len: Optional[int] = None,
    ):
        position_ids = position_ids.contiguous().to(torch.long)
        sin_cos, _, _ = self.embed_positions(seq_len)
        x = x.contiguous()
        torch.ops.torch_ipex.rotary_position_embedding(
            x,
            sin_cos,
            position_ids,
            num_head,
            head_size,
            offset,
            rotary_ndims,
        )

        return x

class FlashLlamaAttention(torch.nn.Module):
    def __init__(
        self,
        prefix: str,
        config,
        weights,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads
        # self.rotary_emb = PositionRotaryEmbedding.load(
        #     config=config, prefix=f"{prefix}.rotary_emb", weights=weights
        # )
        # self.rotary_emb = PositionRotaryEmbedding.static(
        #     config=config,
        #     dim=self.head_size,
        #     base=config.rope_theta,
        #     device=weights.device,
        # )
        self.model_backbone = config.architectures[0]

        self.max_position_embeddings = (
            config.max_position_embeddings
            if hasattr(config, "max_position_embeddings")
            else 2048
        )

        self.pos_embd_dim = self.head_size
        self.rope_base = (
            config.rotary_emb_base if hasattr(config, "rotary_emb_base") else 10000
        )

        self._IPEXROPE = _IPEXRopeCPU(
            self.max_position_embeddings,
            self.pos_embd_dim,
            self.rope_base,
            self.model_backbone,
        )

        self.softmax_scale = self.head_size**-0.5

        if self.num_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_heads = self.num_heads // weights.process_group.size()
        self.num_key_value_heads = (
            config.num_key_value_heads // weights.process_group.size()
        )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_size, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_size, self.hidden_size, bias=False)

        # self.query_key_value = load_attention(config, prefix, weights)

        # self.o_proj = TensorParallelRowLinear.load(
        #     config,
        #     prefix=f"{prefix}.o_proj",
        #     weights=weights,
        #     bias=False,
        # )
        self.num_groups = self.num_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=weights.device
        ).repeat_interleave(self.num_groups)

    def forward(
        self,
        hidden_states,
        cos,
        sin,
        cu_seqlen_prefill,
        kv_cache,
        block_tables,
        slots,
        input_lengths,
        max_s,
        position_ids
    ):
        
        # qkv = self.query_key_value(hidden_states)
        # query, kv = qkv.split(
        #     [
        #         self.head_size * self.num_heads,
        #         2 * self.head_size * self.num_key_value_heads,
        #     ],
        #     dim=1,
        # )
        # query = query.view(-1, self.num_heads, self.head_size)
        # kv = kv.view(-1, 2, self.num_key_value_heads, self.head_size)
        logger.info(hidden_states.size())
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_size)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_size)
        value_states = value_states.view(-1, self.num_key_value_heads, self.head_size)

        # self.rotary_emb(query_states, cos, sin)
        # self.rotary_emb(key_states, cos, sin)

        key_states = self._IPEXROPE(
            key_states,
            position_ids,
            self.num_key_value_heads,
            self.head_size,
            self.head_size // 2,
            self.head_size,
        ).view(-1, self.num_key_value_heads, self.head_size)

        query_states = self._IPEXROPE(
            query_states,
            position_ids,
            self.num_heads,
            self.head_size,
            self.head_size // 2,
            self.head_size,
        ).view(-1, self.num_heads, self.head_size)
        
        if torch.cuda.is_available():
            paged_attention.reshape_and_cache(
                kv[:, 0], kv[:, 1], kv_cache[0], kv_cache[1], slots
            )
        else:
            ref_reshape_and_cache(key_states, value_states, kv_cache[0], kv_cache[1], slots)
        key_states = key_states.to(query_states.dtype)
        value_states = value_states.to(query_states.dtype)
        # output tensor
        attn_output = torch.empty_like(query_states)

        # Prefill
        if cu_seqlen_prefill is not None:
            # flash attention
            flash_attn.attention(
                query_states,
                key_states,
                value_states,
                attn_output,
                cu_seqlen_prefill,
                max_s,
                self.softmax_scale,
            )
        # Decode
        else:
            if torch.cuda.is_available():
                paged_attention.attention(
                    attn_output,
                    query_states,
                    kv_cache[0],
                    kv_cache[1],
                    self.kv_head_mapping,
                    self.softmax_scale,
                    block_tables,
                    input_lengths,
                    max_s,
                )
            else:
                ref_single_query_cached_kv_attention(
                    attn_output,
                    query_states.to(torch.bfloat16),
                    kv_cache[0],
                    kv_cache[1],
                    block_tables,
                    input_lengths,
                )

        # return self.o_proj(attn_output.view(-1, self.num_heads * self.head_size))
        return self.o_proj(attn_output.to(self.o_proj.weight.dtype).view(bsz, q_len, self.num_heads * self.head_size))



class LlamaMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        act = config.hidden_act
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate="tanh"
                if act in ["gelu_fast", "gelu_pytorch_tanh"]
                else "none",
            )
        )
        # Fuse gate and up proj
        # self.gate_up_proj = TensorParallelColumnLinear.load_multi(
        #     config,
        #     prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
        #     weights=weights,
        #     dim=0,
        #     bias=False,
        # )
        # self.down_proj = TensorParallelRowLinear.load(
        #     config,
        #     prefix=f"{prefix}.down_proj",
        #     weights=weights,
        #     bias=False,
        # )
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # self.intermediate_size = (
        #     config.intermediate_size // weights.process_group.size()
        # )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    def forward(self, hidden_states):
        # gate_up_states = self.gate_up_proj(hidden_states)
        # gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
        # return self.down_proj(self.act(gate_up_states[:, 0]) * gate_up_states[:, 1])
        return self.down_proj(self.act(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class FlashLlamaLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"model.layers.{layer_id}"
        self.self_attn = FlashLlamaAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        self.mlp = LlamaMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)

        self.input_layernorm = LlamaRMSNorm(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states,
        residual,
        cos,
        sin,
        cu_seqlen_prefill,
        kv_cache,
        block_tables,
        slots,
        input_lengths,
        max_s,
        position_ids
    ):
        normed_hidden_states, res = self.input_layernorm(hidden_states, residual)

        # Self Attention
        attn_output = self.self_attn(
            normed_hidden_states,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
            position_ids
        )

        # faster post attention rms norm
        normed_attn_res_output, attn_res = self.post_attention_layernorm(
            attn_output, res
        )

        mlp_output = self.mlp(normed_attn_res_output)

        return mlp_output, attn_res


class FlashLlamaModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        # self.embed_tokens = TensorParallelEmbedding(
        #     prefix="model.embed_tokens", weights=weights
        # )
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                FlashLlamaLayer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(
            prefix="model.norm", weights=weights, eps=config.rms_norm_eps
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads
        self.num_key_value_heads = self.layers[0].self_attn.num_key_value_heads

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_s: int,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states.unsqueeze(0)
        # Get rotary cos and sin for this forward
        # Avoid to index in each layer
        # cos, sin = self.layers[0].self_attn.rotary_emb.get_cos_sin(
        #     position_ids, max_s, hidden_states.dtype
        # )
        cos = None
        sin = None

        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                cos,
                sin,
                cu_seqlen_prefill,
                kv_cache[i],
                block_tables,
                slots,
                input_lengths,
                max_s,
                position_ids
                
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class FlashLlamaForCausalLM(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        self.model = FlashLlamaModel(config, weights)
        # self.lm_head = TensorParallelHead.load(
        #     config,
        #     prefix="lm_head",
        #     weights=weights,
        # )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_s: int,
        lm_head_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        hidden_states = self.model(
            input_ids,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
        )
        hidden_states=hidden_states.squeeze(0)
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits = self.lm_head(hidden_states.unsqueeze(0))
        logits = logits.squeeze(0)
        return logits
