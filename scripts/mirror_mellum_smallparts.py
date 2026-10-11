#!/usr/bin/env python3
"""Publish fixed-size Mellum Q4_K_M GGUF parts, verified against pinned upstream."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from huggingface_hub import hf_hub_download

REPO = "JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF"
REVISION = "20b439426f1e997ba575eaccaf4856ac0f76f5d5"
NAME = "Mellum2.1-12B-A2.5B-Thinking-Q4_K_M.gguf"
SIZE = 8_071_295_264
SHA256 = "ecc4d5b8107fc219e23c494b2d107552137637b6c6002ee713e0ba8af7fa2755"
TAG = "mellum2.1-12b-a2.5b-thinking-q4_k_m-smallparts"
CHUNK_SIZE = 200_000_000
ORIGINAL_PART_SIZE = 1_600_000_000
MANIFEST_NAME = NAME + ".smallparts.manifest.json"


def gh(*args, capture=False):
    return subprocess.run(["gh", *args], check=not capture, text=True,
                          capture_output=capture)


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(buf)
    return digest.hexdigest()


def upload(tag, path):
    for attempt in range(1, 5):
        ret = gh("release", "upload", tag, str(path), capture=True)
        if ret.returncode == 0:
            return
        print(f"Upload {path.name} attempt {attempt} failed: {ret.stderr[-400:]}", flush=True)
        if attempt < 4:
            time.sleep(10 * attempt)
    raise RuntimeError(f"Failed to upload asset {path.name}")


def main():
    model = Path(hf_hub_download(repo_id=REPO, filename=NAME, revision=REVISION))
    if model.stat().st_size != SIZE or checksum(model) != SHA256:
        raise RuntimeError("Source file failed original SHA256 or size validation.")
    print("Verified exact upstream GGUF", flush=True)

    existing = gh("release", "view", TAG, "--json", "isDraft,assets,body", capture=True)
    if existing.returncode == 0:
        data = json.loads(existing.stdout)
        if not data["isDraft"] or SHA256 not in data["body"]:
            raise RuntimeError("Existing release is published or has unexpected provenance.")
        uploaded = {a["name"]: a["size"] for a in data["assets"]}
        print("Resuming matching draft", flush=True)
    else:
        notes = (
            f"Original source: https://huggingface.co/{REPO}/blob/{REVISION}/{NAME}\\n"
            f"SHA256 original GGUF: {SHA256}\\n"
            "Apache-2.0. These are standalone ordered 200 MB binary chunks (not ZIP). "
            "Combine in order and verify with the manifest. "
            "This alternate download is intended for interrupted or filtered connections."
        )
        gh("release", "create", TAG, "--draft", "--target", "main",
           "--title", "Mellum 2.1 Thinking Q4_K_M — resilient 200 MB downloads",
           "--notes", notes)
        uploaded = {}

    items = []
    with model.open("rb") as src:
        offset = 0
        index = 0
        while offset < SIZE:
            index += 1
            size = min(CHUNK_SIZE, SIZE - offset)
            partname = f"{NAME}.smallpart{index:02d}"
            part = Path(partname)
            digest = hashlib.sha256()
            remaining = size
            with part.open("wb") as out:
                while remaining:
                    buf = src.read(min(8 * 1024 * 1024, remaining))
                    if not buf:
                        raise RuntimeError("Unexpected source EOF")
                    digest.update(buf)
                    out.write(buf)
                    remaining -= len(buf)
            items.append({"name": partname, "offset": offset, "bytes": size,
                          "sha256": digest.hexdigest()})
            if partname in uploaded:
                if uploaded[partname] != size:
                    raise RuntimeError(f"Existing release asset incorrect size: {partname}")
                print(f"Already uploaded {index}", flush=True)
            else:
                upload(TAG, part)
                print(f"Uploaded {index}/{(SIZE + CHUNK_SIZE - 1)//CHUNK_SIZE}: {partname}", flush=True)
            part.unlink()
            offset += size

    manifest = {
        "source_repo": REPO, "source_revision": REVISION, "file": NAME,
        "size_bytes": SIZE, "sha256": SHA256, "license": "Apache-2.0",
        "chunk_size_bytes": CHUNK_SIZE,
        "original_part_size_bytes": ORIGINAL_PART_SIZE,
        "parts": items,
    }
    path = Path(MANIFEST_NAME)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if path.name not in uploaded:
        upload(TAG, path)
    gh("release", "edit", TAG, "--draft=false")
    print(f"Published: https://github.com/{os.environ['GH_REPO']}/releases/tag/{TAG}", flush=True)


if __name__ == "__main__":
    main()
