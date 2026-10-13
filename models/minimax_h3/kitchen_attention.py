"""Kitchen HIP INT8 attention for ordinary H3 denoiser blocks only."""

import torch
from mmgp import offload


_announced = False


def callback(is_denoiser):
    return attention if is_denoiser and offload.shared_state.get("_attention") == "kitchen_int8" else None


@torch.compiler.disable()
def attention(qkv_list, group=None):
    import comfy_kitchen as ck

    if not torch.version.hip or not ck.int8_attention_is_available():
        raise RuntimeError("Kitchen INT8 H3 attention requires a supported Comfy Kitchen HIP backend")
    global _announced
    if not _announced:
        print("[MiniMax H3] Kitchen HIP INT8 denoiser attention enabled; encoders and VAEs use SDPA.")
        _announced = True
    query, key, value = qkv_list
    qkv_list.clear()
    return ck.int8_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)).transpose(1, 2)
