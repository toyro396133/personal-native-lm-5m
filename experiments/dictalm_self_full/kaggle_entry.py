from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path("/kaggle/working")
SRC = Path(__file__).resolve().parent


def run(cmd, env=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, env=env)


def install():
    run([
        sys.executable, "-m", "pip", "install", "-q",
        "transformers>=4.51,<5",
        "accelerate>=1.2,<2",
        "datasets>=3,<5",
        "sentencepiece",
        "safetensors",
    ])


def make_accelerate_config(gpus: int):
    if gpus < 2:
        return None

    cfg = ROOT / "accelerate-fsdp.yaml"
    cfg.write_text(
        f"""compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: {gpus}
rdzv_backend: static
same_network: true
use_cpu: false
fsdp_config:
  fsdp_activation_checkpointing: false
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: false
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: Qwen3DecoderLayer
  fsdp_use_orig_params: true
""",
        encoding="utf-8",
    )
    return cfg


def main():
    install()

    import torch

    cfg = json.loads((SRC / "run_config.json").read_text(encoding="utf-8"))
    model = cfg.get("model", "dicta-il/DictaLM-3.0-1.7B-Base")
    target = int(cfg.get("target_tokens", 2_000_000))
    val_tokens = int(cfg.get("val_tokens", 200_000))
    seq_len = int(cfg.get("seq_len", 512))

    print(
        json.dumps(
            {
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count(),
                "gpus": [
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                ],
                "target_tokens_per_arm": target,
                "model": model,
            },
            indent=2,
        ),
        flush=True,
    )
    if not torch.cuda.is_available():
        raise SystemExit("Kaggle did not allocate a GPU; aborting before training.")

    data_dir = ROOT / "data"
    run([
        sys.executable,
        SRC / "prepare_data.py",
        "--model", model,
        "--target-train-tokens", str(target),
        "--target-val-tokens", str(val_tokens),
        "--out-dir", data_dir,
    ])

    gpus = torch.cuda.device_count()
    acc_cfg = make_accelerate_config(gpus)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)

    for arm in ("baseline", "self"):
        out = results / f"{arm}.json"

        if acc_cfg is not None:
            launcher = [
                "accelerate", "launch",
                "--config_file", acc_cfg,
            ]
        else:
            launcher = [
                "accelerate", "launch",
                "--num_processes", "1",
                "--mixed_precision", "fp16",
            ]

        run([
            *launcher,
            SRC / "train_full.py",
            "--arm", arm,
            "--model", model,
            "--train", data_dir / "train.i32",
            "--val", data_dir / "val.i32",
            "--target-tokens", str(target),
            "--seq-len", str(seq_len),
            "--micro-batch", "1",
            "--grad-accum", "8",
            "--lr", "1e-5",
            "--seed", "71",
            "--out", out,
        ])

    baseline = json.loads((results / "baseline.json").read_text(encoding="utf-8"))
    self_result = json.loads((results / "self.json").read_text(encoding="utf-8"))

    b = baseline["final_validation"]["nats_per_token"]
    s = self_result["final_validation"]["nats_per_token"]
    off = self_result["self_adapter_off_validation"]["nats_per_token"]

    comparison = {
        "experiment": "DictaLM-3.0-1.7B continued FULL training SELF A/B canary",
        "model": model,
        "target_tokens_per_arm": target,
        "baseline": baseline,
        "self": self_result,
        "self_minus_baseline_nats_per_token": s - b,
        "self_relative_change_percent": 100.0 * (s - b) / b,
        "self_off_minus_self_normal_nats_per_token": off - s,
        "interpretation_keys": {
            "self_minus_baseline_negative": "SELF arm has lower validation loss than matched continued-training baseline.",
            "self_off_minus_self_normal_positive": "Disabling SELF after training hurts, indicating learned runtime dependence on SELF.",
        },
    }
    (ROOT / "dictalm-self-comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
