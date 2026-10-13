"""Temporary, H3-denoiser-only comparison of SDPA and Kitchen HIP INT8."""

import collections

import torch

from shared.attention import pay_attention


_backend = None
_events = []
_shapes = collections.Counter()


def configure(backend):
    global _backend
    if backend not in (None, "sdpa", "kitchen_int8"):
        raise ValueError(f"Unknown experimental backend: {backend}")
    if backend == "kitchen_int8":
        import comfy_kitchen as ck
        if not torch.version.hip or not ck.int8_attention_is_available():
            raise RuntimeError("The experiment requires Kitchen HIP INT8 attention")
    _backend = backend
    _events.clear()
    _shapes.clear()


def callback(is_denoiser):
    return attention if is_denoiser and _backend is not None else None


def attention(qkv_list, group=None):
    shape = tuple(qkv_list[0].shape)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    if _backend == "sdpa":
        output = pay_attention(qkv_list, force_attention="sdpa", recycle_q=True)
    elif _backend == "kitchen_int8":
        import comfy_kitchen as ck
        query, key, value = qkv_list
        qkv_list.clear()
        output = ck.int8_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)).transpose(1, 2)
    else:
        raise RuntimeError("Experimental attention called without a backend")
    end.record()
    _events.append((start, end))
    _shapes[shape] += 1
    return output


def metrics():
    torch.cuda.synchronize()
    times = [start.elapsed_time(end) for start, end in _events]
    return {
        "backend": _backend,
        "calls": len(times),
        "attention_gpu_s": sum(times) / 1000,
        "shapes": [{"shape": list(shape), "calls": calls} for shape, calls in _shapes.items()],
    }
