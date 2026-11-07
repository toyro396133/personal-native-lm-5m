#!/usr/bin/env python3
"""Mirror the selected ACE-Step 1.5 core checkpoints as individual GitHub Release assets.

The two large model weights are split into 200 MB parts so every Release asset
stays comfortably below GitHub's per-asset size limit. The source revision and
the published SHA-256 values are pinned. No model conversion is performed.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_download, hf_hub_url

REPO = "ACE-Step/Ace-Step1.5"
REVISION = "19671f406d603126926c1b7e2adc169acbcade22"
TAG = "ace-step-1.5-core-200mb"
MANIFEST_NAME = "ace-step-1.5-core.manifest.json"
INSTALLER_NAME = "install_ace_step_1_5_core.ps1"
PART_SIZE = 200_000_000
GH_REPO = os.environ.get("GH_REPO", "toyro396133/personal-native-lm-5m")

WORK = Path("ace_step_work")
DIST = WORK / "dist"
DIST.mkdir(parents=True, exist_ok=True)

# Independently published SHA-256 and exact byte sizes from Hugging Face Xet/LFS pointers.
BIG_FILES = [
    {
        "path": "acestep-v15-turbo/model.safetensors",
        "asset_prefix": "ace-step-turbo-model.safetensors",
        "size_bytes": 4_787_825_604,
        "sha256": "3f6e0797fad420a39bd33979eb6e840e30989e34a3794e843d23b60ec6e422d7",
    },
    {
        "path": "Qwen3-Embedding-0.6B/model.safetensors",
        "asset_prefix": "ace-step-qwen3-model.safetensors",
        "size_bytes": 1_191_586_416,
        "sha256": "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
    },
]

SINGLE_FILES = [
    ("config.json", "ace-step-root-config.json", None, None),
    ("acestep-v15-turbo/config.json", "ace-step-turbo-config.json", None, None),
    ("acestep-v15-turbo/configuration_acestep_v15.py", "ace-step-turbo-configuration_acestep_v15.py", None, None),
    ("acestep-v15-turbo/modeling_acestep_v15_turbo.py", "ace-step-turbo-modeling_acestep_v15_turbo.py", None, None),
    ("acestep-v15-turbo/silence_latent.pt", "ace-step-turbo-silence_latent.pt", None, None),
    ("vae/config.json", "ace-step-vae-config.json", None, None),
    (
        "vae/diffusion_pytorch_model.safetensors",
        "ace-step-vae-diffusion_pytorch_model.safetensors",
        337_431_388,
        "da17edb604c40deaf09e9b24974e590d1ca83a374070e5d0884cfa4bed9a99b0",
    ),
    ("Qwen3-Embedding-0.6B/tokenizer.json", "ace-step-qwen3-tokenizer.json", None, None),
    ("Qwen3-Embedding-0.6B/vocab.json", "ace-step-qwen3-vocab.json", None, None),
    ("Qwen3-Embedding-0.6B/merges.txt", "ace-step-qwen3-merges.txt", None, None),
    ("Qwen3-Embedding-0.6B/tokenizer_config.json", "ace-step-qwen3-tokenizer_config.json", None, None),
    ("Qwen3-Embedding-0.6B/special_tokens_map.json", "ace-step-qwen3-special_tokens_map.json", None, None),
    ("Qwen3-Embedding-0.6B/added_tokens.json", "ace-step-qwen3-added_tokens.json", None, None),
    ("Qwen3-Embedding-0.6B/chat_template.jinja", "ace-step-qwen3-chat_template.jinja", None, None),
    ("Qwen3-Embedding-0.6B/config.json", "ace-step-qwen3-config.json", None, None),
]

LICENSE_TEXT = """MIT License

Copyright (c) 2026 ACEStep

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

def gh(*args, check=True):
    result = subprocess.run(["gh", *args], capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(
            f"gh {' '.join(args[:3])} failed: {result.stderr[-1200:]}"
        )
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_release():
    result = gh(
        "release", "view", TAG, "--repo", GH_REPO,
        "--json", "isDraft,assets,body", check=False
    )
    if result.returncode:
        return None
    return json.loads(result.stdout)


def asset_by_name(release, name):
    return next((asset for asset in release["assets"] if asset["name"] == name), None)


def upload_asset(path, expected_sha):
    path = Path(path)
    release = get_release()
    existing = asset_by_name(release, path.name) if release else None
    if existing:
        remote_digest = (existing.get("digest") or "").lower()
        if (
            int(existing["size"]) == path.stat().st_size
            and remote_digest == "sha256:" + expected_sha
        ):
            print(f"ALREADY VERIFIED: {path.name}", flush=True)
            return

    args = ["release", "upload", TAG, str(path), "--repo", GH_REPO]
    if existing:
        args.append("--clobber")
    gh(*args)

    release = get_release()
    uploaded = asset_by_name(release, path.name) if release else None
    if not uploaded:
        raise RuntimeError(f"GitHub Release asset missing after upload: {path.name}")
    if int(uploaded["size"]) != path.stat().st_size:
        raise RuntimeError(f"GitHub Release size mismatch: {path.name}")
    remote_digest = (uploaded.get("digest") or "").lower()
    if remote_digest != "sha256:" + expected_sha:
        raise RuntimeError(
            f"GitHub Release SHA-256 mismatch for {path.name}: "
            f"{remote_digest or 'missing digest'}"
        )
    print(f"PUBLISHED AND VERIFIED: {path.name}", flush=True)


def source_file(filename, label, expected_size=None):
    cache_dir = WORK / f"hf-cache-{label}"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    url = hf_hub_url(repo_id=REPO, filename=filename, revision=REVISION)
    metadata = get_hf_file_metadata(url)
    if metadata.size is None:
        raise RuntimeError(f"Hugging Face did not provide a file size: {filename}")
    if expected_size is not None and int(metadata.size) != int(expected_size):
        raise RuntimeError(
            f"Upstream metadata size mismatch for {filename}: "
            f"{metadata.size} != {expected_size}"
        )

    downloaded = Path(
        hf_hub_download(
            repo_id=REPO,
            filename=filename,
            revision=REVISION,
            cache_dir=str(cache_dir),
            force_download=True,
        )
    ).resolve(strict=True)

    if downloaded.stat().st_size != int(metadata.size):
        raise RuntimeError(f"Downloaded size mismatch: {filename}")
    return downloaded, cache_dir, int(metadata.size)


def ensure_draft_release():
    provenance = {
        "source_repo": REPO,
        "source_revision": REVISION,
        "chunk_size_bytes": PART_SIZE,
        "included_components": [
            "acestep-v15-turbo",
            "vae",
            "Qwen3-Embedding-0.6B",
        ],
        "excluded_components": ["acestep-5Hz-lm-1.7B"],
        "license": "MIT",
        "selected_source_files": sorted(
            [item["path"] for item in BIG_FILES]
            + [item[0] for item in SINGLE_FILES]
        ),
    }
    fingerprint = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    release = get_release()
    if release:
        if not release["isDraft"] or fingerprint not in (release.get("body") or ""):
            raise RuntimeError(
                "Existing release is published or tied to different source provenance."
            )
        print("RESUME: matching draft release found.", flush=True)
        return

    notes = (
        "Selected ACE-Step 1.5 core checkpoints mirrored from the official "
        "Hugging Face repository at a pinned revision.\n\n"
        "Included: Turbo DiT, VAE, Qwen3-Embedding-0.6B, root config and required "
        "model/tokenizer code files.\n"
        "Excluded: acestep-5Hz-lm-1.7B (not needed for the requested configuration).\n\n"
        "Large weights are published as individually downloadable 200 MB parts. "
        "The Windows installer reassembles them, verifies every part and the full "
        "SHA-256, and places the original filenames in the exact checkpoints tree. "
        "No ZIP download is required on the target PC.\n\n"
        "Upstream: https://huggingface.co/" + REPO + "/tree/" + REVISION + "\n"
        "Model license: MIT. Official license is included as an asset.\n\n"
        "PROVENANCE_JSON=" + fingerprint
    )
    gh(
        "release", "create", TAG, "--repo", GH_REPO, "--draft",
        "--target", "main",
        "--title", "ACE-Step 1.5 — Core checkpoints (200 MB parts)",
        "--notes", notes,
    )
    print("Created draft Release.", flush=True)


def upload_single(filename, asset_name, expected_size=None, expected_sha=None):
    downloaded, cache_dir, metadata_size = source_file(
        filename, "single-" + str(len(list(WORK.glob("hf-cache-*")))), expected_size
    )
    try:
        actual_sha = sha256_file(downloaded)
        if expected_sha and actual_sha != expected_sha:
            raise RuntimeError(
                f"Official SHA-256 mismatch for {filename}: {actual_sha} != {expected_sha}"
            )
        local_asset = DIST / asset_name
        shutil.copyfile(downloaded, local_asset)
        asset_sha = sha256_file(local_asset)
        upload_asset(local_asset, asset_sha)
        record = {
            "path": filename,
            "asset_name": asset_name,
            "size_bytes": metadata_size,
            "sha256": actual_sha,
        }
        return record
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
        (DIST / asset_name).unlink(missing_ok=True)


def upload_chunked(spec, index):
    filename = spec["path"]
    downloaded, cache_dir, metadata_size = source_file(
        filename, "chunked-" + str(index), spec["size_bytes"]
    )
    try:
        if metadata_size != spec["size_bytes"]:
            raise RuntimeError(f"Unexpected upstream size for {filename}")
        full_sha = sha256_file(downloaded)
        if full_sha != spec["sha256"]:
            raise RuntimeError(
                f"Official SHA-256 mismatch for {filename}: {full_sha} != {spec['sha256']}"
            )

        parts = []
        offset = 0
        part_index = 0
        with downloaded.open("rb") as source:
            while True:
                block = source.read(PART_SIZE)
                if not block:
                    break
                part_index += 1
                asset_name = f"{spec['asset_prefix']}.part{part_index:03d}"
                local_asset = DIST / asset_name
                with local_asset.open("wb") as output:
                    output.write(block)
                part_sha = sha256_file(local_asset)
                upload_asset(local_asset, part_sha)
                parts.append({
                    "name": asset_name,
                    "offset": offset,
                    "size_bytes": len(block),
                    "sha256": part_sha,
                })
                offset += len(block)
                local_asset.unlink()
                print(
                    f"CHUNKED {filename}: {part_index} "
                    f"({offset:,}/{metadata_size:,} bytes)",
                    flush=True,
                )

        if offset != spec["size_bytes"]:
            raise RuntimeError(f"Chunk total does not match source size: {filename}")
        return {
            "path": filename,
            "size_bytes": spec["size_bytes"],
            "sha256": spec["sha256"],
            "parts": parts,
        }
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def main():
    if not GH_REPO:
        raise RuntimeError("GH_REPO is not set.")
    WORK.mkdir(parents=True, exist_ok=True)

    info = HfApi().model_info(REPO, revision=REVISION)
    if info.sha != REVISION:
        raise RuntimeError(f"Unexpected Hugging Face revision: {info.sha}")

    ensure_draft_release()
    file_records = []

    for index, spec in enumerate(BIG_FILES, start=1):
        file_records.append(upload_chunked(spec, index))

    for filename, asset_name, expected_size, expected_sha in SINGLE_FILES:
        file_records.append(
            upload_single(filename, asset_name, expected_size, expected_sha)
        )

    license_path = DIST / "ACE-STEP-MIT-LICENSE.txt"
    license_path.write_text(LICENSE_TEXT, encoding="utf-8", newline="\n")
    license_sha = sha256_file(license_path)
    upload_asset(license_path, license_sha)
    file_records.append({
        "path": "ACE-STEP-MIT-LICENSE.txt",
        "asset_name": "ACE-STEP-MIT-LICENSE.txt",
        "size_bytes": license_path.stat().st_size,
        "sha256": license_sha,
        "source_url": "https://github.com/ace-step/ACE-Step-1.5/blob/main/LICENSE",
    })
    license_path.unlink()

    manifest = {
        "format": "ace-step-1.5-core-files-v1",
        "source_repo": REPO,
        "source_revision": REVISION,
        "license": "MIT",
        "license_source": "https://github.com/ace-step/ACE-Step-1.5/blob/main/LICENSE",
        "chunk_size_bytes": PART_SIZE,
        "excluded_components": ["acestep-5Hz-lm-1.7B"],
        "files": file_records,
    }
    manifest_path = WORK / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    upload_asset(manifest_path, sha256_file(manifest_path))

    installer = Path("scripts") / INSTALLER_NAME
    if not installer.is_file():
        raise RuntimeError(f"Installer script not found in checkout: {installer}")
    installer_asset = DIST / INSTALLER_NAME
    shutil.copyfile(installer, installer_asset)
    upload_asset(installer_asset, sha256_file(installer_asset))
    installer_asset.unlink()

    gh("release", "edit", TAG, "--repo", GH_REPO, "--draft=false")
    total_bytes = sum(int(item["size_bytes"]) for item in file_records)
    part_count = sum(len(item.get("parts", [])) for item in file_records)
    print(
        f"COMPLETE: https://github.com/{GH_REPO}/releases/tag/{TAG}\n"
        f"Files in local checkpoints tree: {len(file_records)}\n"
        f"Chunked weight assets: {part_count}\n"
        f"Total source bytes: {total_bytes:,}\n"
        f"Excluded LM: acestep-5Hz-lm-1.7B",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
