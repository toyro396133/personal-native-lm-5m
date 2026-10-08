from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--used-hashes", required=True)
    ap.add_argument("--data-report", required=True)
    ap.add_argument("--comparison")
    ap.add_argument("--token", required=True)
    args = ap.parse_args()

    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo,
        repo_type="model",
        private=True,
        exist_ok=True,
    )

    progress = {
        "schema_version": 1,
        "experiment": "goldfish-124m-self-100m-100-stage",
        "completed_stage": args.stage,
        "nominal_total_tokens": args.stage * 1_000_000,
        "next_stage": None if args.stage >= 100 else args.stage + 1,
    }
    progress_path = Path("progress.json")
    progress_path.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    uploads = [
        (args.used_hashes, "progress/used_doc_hashes.txt"),
        (args.data_report, f"data/stage-{args.stage:03d}/data_report.json"),
        (str(progress_path), "progress/current.json"),
    ]
    if args.comparison:
        uploads.append((args.comparison, f"structural/stage-{args.stage:03d}/comparison.json"))

    for local, remote in uploads:
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"Finalize Goldfish stage {args.stage}",
        )

    print("Squashing repository history to retain files without old resume-state blobs...", flush=True)
    try:
        api.super_squash_history(
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"Squash through stage {args.stage}",
        )
    except TypeError:
        api.super_squash_history(repo_id=args.repo, repo_type="model")

    print(json.dumps(progress, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
