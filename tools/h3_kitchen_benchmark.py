"""Run an isolated, warm H3 attention A/B comparison through the Python API."""

import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/experiment")


def write_json(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2) + "\n")


def main():
    from models.minimax_h3 import attention_experiment as experiment

    torch.manual_seed(123)
    tensors = [torch.randn(1, 257, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    cpu = [tensor.float().cpu().transpose(1, 2) for tensor in tensors]
    expected = F.scaled_dot_product_attention(*cpu).transpose(1, 2)
    checks = []
    for backend in ("sdpa", "kitchen_int8"):
        experiment.configure(backend)
        actual = experiment.attention(list(tensors)).float().cpu()
        assert actual.shape == expected.shape and torch.isfinite(actual).all()
        rmse = (actual - expected).square().mean().sqrt() / expected.square().mean().sqrt()
        cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
        assert rmse < 0.1 and cosine > 0.99, (backend, rmse.item(), cosine.item())
        checks.append({"backend": backend, "relative_rmse": rmse.item(), "cosine": cosine.item(), **experiment.metrics()})
    write_json("kernel-checks.json", checks)
    print("KERNEL_CHECKS", json.dumps(checks), flush=True)
    del tensors, cpu, expected, actual
    torch.cuda.empty_cache()

    from shared.api import init

    session = init(root=Path("/opt/WanGP"), config_path=ROOT / "config/wgp_config.json",
                   output_dir=ROOT / "renders",
                   cli_args=["--attention", "sdpa", "--profile", "4", "--settings", str(ROOT / "settings-cache"), "--loras", "/loras"],
                   console_isatty=False)
    settings = json.loads((ROOT / "settings.json").read_text())
    runs = []
    sequence = [("warmup", "sdpa"), ("sdpa-1", "sdpa"), ("kitchen-1", "kitchen_int8"),
                ("sdpa-2", "sdpa"), ("kitchen-2", "kitchen_int8")]
    for label, backend in sequence:
        task = copy.deepcopy(settings)
        if label == "warmup":
            task.update(video_length=22, num_inference_steps=2)
        task["output_filename"] = label
        experiment.configure(backend)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        print("RUN_START", label, json.dumps(task), flush=True)
        result = session.run_task(task)
        wall = time.monotonic() - started
        row = {"label": label, "wall_s": wall, "success": result.success,
               "generated_files": result.generated_files,
               "errors": [error.message for error in result.errors],
               "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
               "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
               **experiment.metrics()}
        runs.append(row)
        write_json("results.json", runs)
        print("RUN_RESULT", json.dumps(row), flush=True)
        if not result.success:
            raise RuntimeError(f"{label} failed: {row['errors']}")
        if not row["calls"]:
            raise RuntimeError("H3 denoiser experiment did not execute")
    experiment.configure(None)


if __name__ == "__main__":
    main()
