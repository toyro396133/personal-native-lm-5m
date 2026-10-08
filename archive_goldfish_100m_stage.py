from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--variant-code", required=True)
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--exact-state", required=True)
    ap.add_argument("--int4", required=True)
    ap.add_argument("--bf16")
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--token", required=True)
    args = ap.parse_args()

    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo,
        repo_type="model",
        private=True,
        exist_ok=True,
    )

    label = f"{args.stage:03d}m"
    uploads = [
        (args.exact_state, f"resume/{args.variant_code}/latest.pt"),
        (args.int4, f"checkpoints/{args.variant_code}/{label}/model-int4.pt"),
        (args.metrics, f"metrics/{args.variant_code}/{label}.json"),
    ]
    if args.bf16:
        uploads.append(
            (args.bf16, f"checkpoints/{args.variant_code}/{label}/model-bf16.pt")
        )

    for local, remote in uploads:
        p = Path(local)
        if not p.exists():
            raise FileNotFoundError(local)
        print(f"upload {p} -> {args.repo}/{remote}", flush=True)
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=remote,
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"{args.variant_code} stage {args.stage} {p.name}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
