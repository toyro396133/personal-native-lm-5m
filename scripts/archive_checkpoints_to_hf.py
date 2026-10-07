#!/usr/bin/env python3
"""Archive code-named model checkpoints from GitHub Actions to Hugging Face Hub.

Reads models/code-variants/V*/manifest.json, downloads each immutable Actions
artifact by artifact_id, verifies/extracts the checkpoint, computes SHA-256, and
uploads it to a private (by default) Hugging Face model repository.

The script is resumable: archive-index.json on Hugging Face records completed
artifact IDs, so reruns skip already archived checkpoints unless --force is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests
from huggingface_hub import (
    CommitOperationAdd,
    HfApi,
    hf_hub_download,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = ROOT / "models" / "code-variants"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_local_checkpoints() -> list[dict]:
    rows: list[dict] = []
    for manifest_path in sorted(MANIFEST_ROOT.glob("V*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        variant = manifest["codename"]
        for cp in manifest["checkpoints"]:
            rows.append(
                {
                    "variant": variant,
                    "tokens_m": int(cp["tokens_m"]),
                    "source_run": int(cp["source_run"]),
                    "artifact_id": int(cp["artifact_id"]),
                    "lineage": cp.get("lineage"),
                }
            )
    rows.sort(key=lambda x: (x["variant"], x["tokens_m"]))
    return rows


def github_json(repo: str, artifact_id: int, token: str) -> dict:
    url = f"https://api.github.com/repos/{repo}/actions/artifacts/{artifact_id}"
    r = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def download_artifact(repo: str, artifact_id: int, token: str, out_zip: Path) -> None:
    url = f"https://api.github.com/repos/{repo}/actions/artifacts/{artifact_id}/zip"
    with requests.get(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=120,
        stream=True,
        allow_redirects=True,
    ) as r:
        r.raise_for_status()
        with out_zip.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)


def extract_checkpoint(zip_path: Path, out_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    pts = sorted(p for p in out_dir.rglob("*.pt") if p.is_file())
    if len(pts) != 1:
        raise RuntimeError(
            f"Expected exactly one .pt checkpoint in {zip_path.name}, found {len(pts)}: "
            + ", ".join(str(p.relative_to(out_dir)) for p in pts)
        )
    return pts[0]


def load_remote_index(repo_id: str, token: str) -> dict:
    try:
        p = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename="archive-index.json",
            token=token,
        )
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return {
            "schema_version": 1,
            "repo_id": repo_id,
            "checkpoints": {},
        }


def model_card(repo_id: str, private: bool, count: int) -> str:
    visibility = "private" if private else "public"
    return f"""---
tags:
- research
- checkpoints
- pytorch
---

# Personal Native LM — code-named checkpoint archive

Permanent checkpoint archive synchronized from GitHub Actions.

- Hugging Face repository: `{repo_id}`
- Intended visibility: **{visibility}**
- Variants are intentionally stored under stable code names `V1`…`V7`.
- Source of truth for checkpoint discovery:
  `models/code-variants/V*/manifest.json` in
  `toyro396133/personal-native-lm-5m`.
- Archived checkpoints tracked by this sync: **{count}**.

## Layout

```
checkpoints/
  V1/
    025M/
      checkpoint.pt
      metadata.json
    ...
  V7/
    140M/
      checkpoint.pt
      metadata.json
archive-index.json
```

Every metadata file records the source GitHub run, immutable artifact ID,
original checkpoint filename, SHA-256 and byte size.

This repository is an archival research store, not a claim that every
checkpoint is an inference-ready Transformers model.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-name", default="personal-native-lm-checkpoints")
    ap.add_argument("--namespace", default="")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Archive at most N checkpoints (0=all)")
    args = ap.parse_args()

    gh_token = os.environ.get("GITHUB_TOKEN", "")
    hf_token = os.environ.get("HF_TOKEN", "")
    gh_repo = os.environ.get("GITHUB_REPOSITORY", "toyro396133/personal-native-lm-5m")
    if not gh_token:
        raise SystemExit("GITHUB_TOKEN is required")
    if not hf_token:
        raise SystemExit("HF_TOKEN is required (GitHub Actions secret named HF_TOKEN)")

    api = HfApi(token=hf_token)
    who = api.whoami()
    namespace = args.namespace.strip() or who["name"]
    repo_id = f"{namespace}/{args.repo_name}"
    private = not args.public

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )

    checkpoints = load_local_checkpoints()
    if args.limit > 0:
        checkpoints = checkpoints[: args.limit]

    index = load_remote_index(repo_id, hf_token)
    index.setdefault("checkpoints", {})
    index["repo_id"] = repo_id
    index["source_github_repo"] = gh_repo

    archived = 0
    skipped = 0

    for i, row in enumerate(checkpoints, start=1):
        key = f'{row["variant"]}@{row["tokens_m"]}M'
        old = index["checkpoints"].get(key)
        if (
            old
            and not args.force
            and int(old.get("artifact_id", -1)) == row["artifact_id"]
            and int(old.get("source_run", -1)) == row["source_run"]
        ):
            print(f"[{i}/{len(checkpoints)}] SKIP {key} (already archived)")
            skipped += 1
            continue

        info = github_json(gh_repo, row["artifact_id"], gh_token)
        if info.get("expired"):
            raise RuntimeError(
                f"GitHub artifact {row['artifact_id']} for {key} has expired before archival"
            )

        print(
            f"[{i}/{len(checkpoints)}] ARCHIVE {key} "
            f"artifact={row['artifact_id']} name={info.get('name')}"
        )

        with tempfile.TemporaryDirectory(prefix="hf-archive-") as td:
            td = Path(td)
            zip_path = td / "artifact.zip"
            extract_dir = td / "extracted"
            extract_dir.mkdir()

            download_artifact(gh_repo, row["artifact_id"], gh_token, zip_path)
            checkpoint = extract_checkpoint(zip_path, extract_dir)

            digest = sha256_file(checkpoint)
            size = checkpoint.stat().st_size
            prefix = f'checkpoints/{row["variant"]}/{row["tokens_m"]:03d}M'
            model_path = f"{prefix}/checkpoint.pt"
            metadata_path = f"{prefix}/metadata.json"

            metadata = {
                "schema_version": 1,
                "variant": row["variant"],
                "tokens_m": row["tokens_m"],
                "lineage": row.get("lineage"),
                "source_github_repo": gh_repo,
                "source_run": row["source_run"],
                "artifact_id": row["artifact_id"],
                "artifact_name": info.get("name"),
                "artifact_created_at": info.get("created_at"),
                "original_checkpoint_filename": checkpoint.name,
                "sha256": digest,
                "size_bytes": size,
                "archived_at": datetime.now(timezone.utc).isoformat(),
                "hf_path": model_path,
            }
            meta_local = td / "metadata.json"
            meta_local.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

            api.create_commit(
                repo_id=repo_id,
                repo_type="model",
                operations=[
                    CommitOperationAdd(path_in_repo=model_path, path_or_fileobj=str(checkpoint)),
                    CommitOperationAdd(path_in_repo=metadata_path, path_or_fileobj=str(meta_local)),
                ],
                commit_message=f"Archive {key} from GitHub artifact {row['artifact_id']}",
            )

            index["checkpoints"][key] = metadata
            archived += 1

        # Persist progress after every checkpoint so interrupted runs are resumable.
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
            json.dump(index, tf, indent=2)
            tf.write("\n")
            index_tmp = tf.name
        try:
            api.upload_file(
                repo_id=repo_id,
                repo_type="model",
                path_or_fileobj=index_tmp,
                path_in_repo="archive-index.json",
                commit_message=f"Update archive index after {key}",
            )
        finally:
            Path(index_tmp).unlink(missing_ok=True)

    # Refresh model card after successful sync.
    index["updated_at"] = datetime.now(timezone.utc).isoformat()
    index["checkpoint_count"] = len(index["checkpoints"])
    with tempfile.TemporaryDirectory(prefix="hf-index-") as td:
        td = Path(td)
        idx = td / "archive-index.json"
        readme = td / "README.md"
        idx.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        readme.write_text(model_card(repo_id, private, len(index["checkpoints"])), encoding="utf-8")
        api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=[
                CommitOperationAdd(path_in_repo="archive-index.json", path_or_fileobj=str(idx)),
                CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(readme)),
            ],
            commit_message="Refresh checkpoint archive index",
        )

    print(
        json.dumps(
            {
                "repo_id": repo_id,
                "private": private,
                "archived_this_run": archived,
                "skipped_existing": skipped,
                "total_indexed": len(index["checkpoints"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
