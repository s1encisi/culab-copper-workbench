"""Exact query-axis chunking for the pinned TabPFN CPU attention fallback."""


def enable_chunked_cpu_attention(query_chunk=64):
    import torch
    from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention

    if hasattr(MultiHeadAttention, "_copper_original_attention"):
        return MultiHeadAttention._copper_original_attention
    original = MultiHeadAttention.compute_attention_heads

    def compute(q, k, v, kv, qkv, dropout_p=None, softmax_scale=None):
        original_inputs = (q, k, v, kv, qkv)
        if qkv is not None:
            q, k, v = qkv.unbind(dim=-3)
        elif kv is not None:
            k, v = kv.unbind(dim=-3)
        if q.device.type != "cpu" or q.shape[1] <= query_chunk or dropout_p not in (None, 0, 0.0):
            return original(*original_inputs, dropout_p=dropout_p, softmax_scale=softmax_scale)
        outputs = []
        for start in range(0, q.shape[1], query_chunk):
            outputs.append(
                original(
                    q[:, start : start + query_chunk],
                    k,
                    v,
                    None,
                    None,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                )
            )
        return torch.cat(outputs, dim=1)

    MultiHeadAttention._copper_original_attention = original
    MultiHeadAttention.compute_attention_heads = staticmethod(compute)
    return original
