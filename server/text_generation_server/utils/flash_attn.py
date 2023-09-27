import os
import torch

from loguru import logger

if os.getenv("USE_FLASH_ATTENTION", "").lower() == "false":
    raise ImportError("`USE_FLASH_ATTENTION` is false.")

#if not torch.cuda.is_available():
 #   raise ImportError("CUDA is not available")

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    is_sm75 = major == 7 and minor == 5
    is_sm8x = major == 8 and minor >= 0
    is_sm90 = major == 9 and minor == 0

HAS_FLASH_ATTN = False
HAS_FLASH_ATTN_V2 = False
if torch.cuda.is_available():
    try:
        try:
            import flash_attn_2_cuda
        except ImportError:
            raise ImportError(
                "Flash Attention V2 is not installed.\n"
                "Use the official Docker image (ghcr.io/huggingface/text-generation-inference:latest) "
                "or install flash attention v2 with `cd server && make install install-flash-attention-v2`"
            )
        if not (is_sm8x or is_sm90):
            raise ImportError(
                f"GPU with CUDA capability {major} {minor} is not supported for "
                "Flash Attention V2"
            )
        HAS_FLASH_ATTN_V2 = True
    except ImportError as e:
        try:
            import flash_attn_cuda
        except ImportError:
            raise ImportError(
                "Flash Attention is not installed.\n"
                "Use the official Docker image (ghcr.io/huggingface/text-generation-inference:latest) "
                "or install flash attention with `cd server && make install install-flash-attention`"
            ) from e

        if not (is_sm75 or is_sm8x or is_sm90):
            raise ImportError(
                f"GPU with CUDA capability {major} {minor} is not supported"
            ) from e
        logger.warning(f"Unable to use Flash Attention V2: {e}")
        HAS_FLASH_ATTN = True


def attention(
    q,
    k,
    v,
    out,
    cu_seqlens,
    max_s,
    softmax_scale,
    window_size_left=-1,
):
    if HAS_FLASH_ATTN_V2:
        return flash_attn_2_cuda.varlen_fwd(
            q,
            k,
            v,
            out,
            cu_seqlens,
            cu_seqlens,
            max_s,
            max_s,
            0.0,
            softmax_scale,
            False,
            True,
            window_size_left,
            0,
            False,
            None,
        )

    if HAS_FLASH_ATTN:
        if window_size_left != -1:
            raise NotImplementedError(
                "window_size_left is only available with flash attn v2"
            )

        # Flash attention v1 requires q, k and v to have the same number of heads
        if k.shape[1] != q.shape[1]:
            # MQA expand
            if k.shape[1] == 1:
                k = k.expand(-1, q.shape[1], -1)
            # Grouped attention reshape
            else:
                original_shape = k.shape
                k = (
                    k.unsqueeze(2)
                    .expand(-1, -1, q.shape[1] // k.shape[1], -1)
                    .reshape(original_shape[0], -1, original_shape[2])
                )
        if v.shape[1] != q.shape[1]:
            # MQA expand
            if v.shape[1] == 1:
                v = v.expand(-1, q.shape[1], -1)
            # Grouped attention reshape
            else:
                original_shape = v.shape
                v = (
                    v.unsqueeze(2)
                    .expand(-1, -1, q.shape[1] // v.shape[1], -1)
                    .reshape(original_shape[0], -1, original_shape[2])
                )

        return flash_attn_cuda.fwd(
            q,
            k,
            v,
            out,
            cu_seqlens,
            cu_seqlens,
            max_s,
            max_s,
            0.0,
            softmax_scale,
            False,
            True,
            False,
            0,
            None,
        )

    q = q.view(-1,max_s,q.shape[1],q.shape[2]).transpose(1, 2)
    k = k.view(-1,max_s,k.shape[1],k.shape[2]).transpose(1, 2)
    v = v.view(-1,max_s,v.shape[1],v.shape[2]).transpose(1, 2)
    a = torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=None, dropout_p=0.0, is_causal=True)
    out.copy_(a.transpose(1, 2).reshape(-1, out.shape[-2],out.shape[-1]))
    return out


def ref_reshape_and_cache(key, value, key_cache, value_cache, slots):
    x = 16 // torch.tensor([], dtype=key.dtype).element_size()
    num_key_value_heads = key.shape[-2]
    head_size = key.shape[-1]
    reshaped_key = key.reshape(-1, num_key_value_heads, head_size // x, x)
    num_tokens = value.shape[0]
    block_size = value_cache.shape[3]
    for i in range(num_tokens):
        block_idx = torch.div(slots[i], block_size, rounding_mode='floor')
        block_offset = slots[i] % block_size
        key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
        value_cache[block_idx, :, :, block_offset] = value[i]

def ref_single_query_cached_kv_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
) -> None:
    num_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]

    num_input_tokens = query.shape[0]
    for i in range(num_input_tokens):
        q = query[i].unsqueeze(0)
        block_table = block_tables[i]
        context_len = int(context_lens[i])
        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_heads, head_size)
            keys.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)

        scale = 1.0 / (head_size ** 0.5)
        out = torch.nn.functional.scaled_dot_product_attention(q.view(1,-1,num_heads,head_size).transpose(1,2),
                keys.view(1,-1,num_heads,head_size).transpose(1,2),
                values.view(1,-1,num_heads,head_size).transpose(1,2),
                attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.view(num_heads, head_size)
        output[i].copy_(out, non_blocking=True)

