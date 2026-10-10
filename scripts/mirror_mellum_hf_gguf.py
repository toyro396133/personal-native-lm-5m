#!/usr/bin/env python3
"""Mirror JetBrains Mellum GGUF to separately downloadable GitHub Release parts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

from huggingface_hub import HfApi, hf_hub_download

REPO = "JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF"
QUANT = os.environ.get("QUANT", "Q4_K_M")
if QUANT not in {"Q4_K_M", "MXFP4_MOE"}:
    raise SystemExit(f"Unsupported quantization: {QUANT}")
FILE = f"Mellum2.1-12B-A2.5B-Thinking-{QUANT}.gguf"
TAG = f"mellum2.1-12b-a2.5b-thinking-{QUANT.lower()}"
CHUNK_SIZE = 1_600_000_000  # Below GitHub's 2 GiB per-release-asset limit.


def gh(*args):
    subprocess.run(["gh", *args], check=True)


def main():
    api = HfApi()
    info = api.model_info(repo_id=REPO, files_metadata=True)
    revision = info.sha
    sibling = next((s for s in info.siblings if s.rfilename == FILE), None)
    if sibling is None:
        raise RuntimeError(f"Model file missing at upstream revision: {FILE}")

    previous = subprocess.run(
        ["gh", "release", "view", TAG, "--json", "isDraft,body,assets"],
        capture_output=True, text=True,
    )
    if previous.returncode == 0:
        current = json.loads(previous.stdout)
        if not current["isDraft"]:
            raise RuntimeError(f"Release {TAG} is already published: refusing overwrite")
        if revision not in current.get("body", ""):
            raise RuntimeError("Existing draft belongs to a different upstream revision")
        existing = {a["name"]: a["size"] for a in current["assets"]}
        print(f"Resuming matching release draft: {TAG}", flush=True)
    else:
        existing = {}

    print(f"Downloading {REPO}@{revision}: {FILE}", flush=True)
    model = Path(hf_hub_download(repo_id=REPO, filename=FILE, revision=revision))
    actual_size = model.stat().st_size
    declared_size = getattr(sibling, "size", None)
    if declared_size is not None and actual_size != declared_size:
        raise RuntimeError(f"Upstream size mismatch: {actual_size} vs {declared_size}")
    whole_hash = hashlib.sha256()
    with model.open("rb") as src:
        for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
            whole_hash.update(block)
    sha256 = whole_hash.hexdigest()
    lfs = getattr(sibling, "lfs", None) or {}
    expected_sha256 = lfs.get("sha256") if isinstance(lfs, dict) else None
    if expected_sha256 and sha256 != expected_sha256:
        raise RuntimeError(f"Upstream SHA256 mismatch: {sha256} != {expected_sha256}")
    print(f"Source verified: {actual_size} bytes; SHA256 {sha256}", flush=True)

    if previous.returncode != 0:
        notes = (
            f"Source: https://huggingface.co/{REPO}/blob/{revision}/{FILE}\n"
            f"Pinned revision: {revision}\nOriginal GGUF SHA256: {sha256}\n"
            "License: Apache-2.0. These are numbered raw parts, NOT ZIP files. "
            "Download all .partXX assets, concatenate in order and verify with the manifest. "
            "The model belongs to its original authors."
        )
        gh("release", "create", TAG, "--draft", "--target", "main",
           "--title", f"JetBrains Mellum 2.1 12B-A2.5B Thinking {QUANT} (GGUF)",
           "--notes", notes)

    parts = []
    with model.open("rb") as source:
        i = 0
        while source.tell() < actual_size:
            i += 1
            name = f"{FILE}.part{i:02d}"
            chunk = Path(name)
            left = min(CHUNK_SIZE, actual_size - source.tell())
            chunk_hash = hashlib.sha256()
            with chunk.open("wb") as target:
                while left > 0:
                    data = source.read(min(left, 8 * 1024 * 1024))
                    if not data:
                        raise RuntimeError("Unexpected end of original model")
                    target.write(data)
                    chunk_hash.update(data)
                    left -= len(data)
            size = chunk.stat().st_size
            parts.append({"name": name, "bytes": size, "sha256": chunk_hash.hexdigest()})
            if name in existing:
                if existing[name] != size:
                    raise RuntimeError(f"Existing part is wrong size: {name}")
                print(f"Existing part confirmed by size: {name}", flush=True)
            else:
                print(f"Uploading {i}: {name} ({size} bytes)", flush=True)
                gh("release", "upload", TAG, name)
            chunk.unlink()

    manifest = {
        "source_repo": REPO, "source_revision": revision,
        "file": FILE, "quantization": QUANT, "license": "Apache-2.0",
        "size_bytes": actual_size, "sha256": sha256, "parts": parts,
    }
    manifest_file = f"{FILE}.manifest.json"
    Path(manifest_file).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if manifest_file not in existing:
        gh("release", "upload", TAG, manifest_file)
    else:
        print("Existing manifest retained", flush=True)
    gh("release", "edit", TAG, "--draft=false")
    print(f"Published: https://github.com/{os.environ['GH_REPO']}/releases/tag/{TAG}", flush=True)


if __name__ == "__main__":
    main()
