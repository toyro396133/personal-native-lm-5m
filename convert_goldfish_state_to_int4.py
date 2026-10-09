from __future__ import annotations

import argparse
import torch

from goldfish_checkpoint_codec import save_int4_model_checkpoint


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    x = torch.load(args.state, map_location="cpu", weights_only=False)
    meta = {
        k: x.get(k)
        for k in (
            "schema_version",
            "experiment",
            "variant_code",
            "variant_name",
            "stage",
            "model_id",
            "model_revision",
            "parameter_count",
            "rank",
            "projection_rank",
            "nominal_total_tokens",
            "cumulative_actual_tokens",
            "seq_len",
        )
    }
    save_int4_model_checkpoint(args.out, x["model_state"], meta, block_size=256)
    print(
        "converted",
        x.get("variant_code"),
        "stage",
        x.get("stage"),
        "to",
        args.out,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
