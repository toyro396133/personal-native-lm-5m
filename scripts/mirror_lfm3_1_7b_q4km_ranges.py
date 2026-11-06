#!/usr/bin/env python3
"""Mirror the official LiquidAI LFM3-1.7B Q4_K_M GGUF to 200 MB GitHub assets.

The upstream file is pinned by immutable revision, byte size and SHA-256.
Each HTTP range is validated before upload; the concatenated source ranges
must reproduce the published full-file SHA-256. No quantization or conversion
is performed here.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import requests
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_download, hf_hub_url

REPO = "LiquidAI/LFM3-1.7B-GGUF"
REVISION = "b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf"
NAME = "LFM3-1.7B-Q4_K_M.gguf"
SIZE = 1_099_627_808
SHA256 = "3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4"
TAG = "lfm3-1.7b-q4-k-m-200mb"
PART_SIZE = 200_000_000
LICENSE_ASSET = "LFM-OPEN-LICENSE.txt"
MANIFEST_NAME = "lfm3-1.7b-q4km.manifest.json"
RETRIES = 5
OUTDIR = Path("transfer_tmp")
OUTDIR.mkdir(exist_ok=True)


def gh(*args, check=True):
    result = subprocess.run(["gh", *args], capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(f"gh {' '.join(args[:3])} failed: {result.stderr[-1000:]}")
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_release():
    result = gh("release", "view", TAG, "--json", "isDraft,assets,body", check=False)
    if result.returncode:
        return None
    return json.loads(result.stdout)


def asset_by_name(release, name):
    return next((asset for asset in release["assets"] if asset["name"] == name), None)


def upload_asset(path, expected_sha):
    release = get_release()
    existing = asset_by_name(release, path.name) if release else None
    if existing:
        if int(existing["size"]) != path.stat().st_size:
            raise RuntimeError(f"Existing release asset has wrong size: {path.name}")
        remote_digest = (existing.get("digest") or "").lower()
        if remote_digest and remote_digest != "sha256:" + expected_sha:
            raise RuntimeError(f"Existing release asset has wrong SHA-256: {path.name}")
        print(f"Already uploaded and size-checked: {path.name}", flush=True)
        return
    for attempt in range(1, RETRIES + 1):
        result = gh("release", "upload", TAG, str(path), check=False)
        if result.returncode == 0:
            break
        print(f"Upload {path.name} attempt {attempt}/{RETRIES}: {result.stderr[-350:]}", flush=True)
        if attempt == RETRIES:
            raise RuntimeError(f"GitHub upload failed: {path.name}")
        time.sleep(min(30, 4 * attempt))
    release = get_release()
    uploaded = asset_by_name(release, path.name) if release else None
    if not uploaded or int(uploaded["size"]) != path.stat().st_size:
        raise RuntimeError(f"GitHub asset verification failed: {path.name}")
    remote_digest = (uploaded.get("digest") or "").lower()
    if remote_digest and remote_digest != "sha256:" + expected_sha:
        raise RuntimeError(f"GitHub asset SHA-256 mismatch: {path.name}")


def source_metadata():
    api = HfApi()
    info = api.model_info(REPO, revision=REVISION)
    if info.sha != REVISION:
        raise RuntimeError(f"Unexpected upstream revision: {info.sha}")
    url = hf_hub_url(repo_id=REPO, filename=NAME, revision=REVISION)
    metadata = get_hf_file_metadata(url)
    if int(metadata.size) != SIZE:
        raise RuntimeError(f"Unexpected upstream size: {metadata.size} != {SIZE}")
    print(f"PINNED SOURCE: {REPO}@{REVISION}/{NAME}", flush=True)
    print(f"EXPECTED SIZE: {SIZE} bytes; SHA256: {SHA256}", flush=True)
    return url


def download_range(url, start, end, target):
    expected_length = end - start + 1
    for attempt in range(1, RETRIES + 1):
        response = None
        try:
            response = requests.get(
                url,
                headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                stream=True,
                timeout=(40, 120),
                allow_redirects=True,
            )
            if response.status_code != 206:
                raise RuntimeError(
                    f"HTTP {response.status_code} ignored requested Range; refusing a full-file download"
                )
            wanted = f"bytes {start}-{end}/{SIZE}"
            actual = response.headers.get("Content-Range", "").strip()
            if actual != wanted:
                raise RuntimeError(f"Invalid Content-Range: {actual!r}, expected {wanted!r}")
            length_header = response.headers.get("Content-Length")
            if length_header and int(length_header) != expected_length:
                raise RuntimeError(f"Incorrect Content-Length: {length_header}")
            digest = hashlib.sha256()
            count = 0
            with target.open("wb") as output:
                for block in response.iter_content(4 * 1024 * 1024):
                    if block:
                        count += len(block)
                        if count > expected_length:
                            raise RuntimeError("Server sent more bytes than requested")
                        digest.update(block)
                        output.write(block)
            if count != expected_length:
                raise RuntimeError(f"Truncated range: {count}/{expected_length}")
            return digest.hexdigest()
        except Exception as error:
            target.unlink(missing_ok=True)
            print(f"Range {start}-{end} attempt {attempt}/{RETRIES}: {error}", flush=True)
            if attempt == RETRIES:
                raise
            time.sleep(min(30, 4 * attempt))
        finally:
            if response is not None:
                response.close()


def main():
    url = source_metadata()
    provenance = {
        "source_repo": REPO,
        "source_revision": REVISION,
        "file": NAME,
        "size_bytes": SIZE,
        "sha256": SHA256,
        "license": "LFM Open License v1.0",
    }
    fingerprint = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    release = get_release()
    if release:
        if not release["isDraft"] or fingerprint not in (release.get("body") or ""):
            raise RuntimeError("Existing release is published or tied to different source provenance.")
        print("RESUME: matching draft release", flush=True)
    else:
        notes = (
            "Official LiquidAI LFM3-1.7B Q4_K_M GGUF, mirrored unchanged as ordered 200 MB chunks. "
            "Not a ZIP; combine parts in order and verify the manifest.\\n\\n"
            "Upstream: https://huggingface.co/" + REPO + "/blob/" + REVISION + "/" + NAME +
            "\\nLicense: LFM Open License v1.0 (license attached as an asset).\\n\\n"
            "PROVENANCE_JSON=" + fingerprint
        )
        gh(
            "release", "create", TAG, "--draft", "--target", "main",
            "--title", "LiquidAI LFM3-1.7B Q4_K_M — resumable 200 MB parts",
            "--notes", notes,
        )
        print("Created draft release", flush=True)

    rows = []
    full_hash = hashlib.sha256()
    count = (SIZE + PART_SIZE - 1) // PART_SIZE
    for index in range(count):
        start = index * PART_SIZE
        end = min(start + PART_SIZE, SIZE) - 1
        part_name = f"{NAME}.part{index + 1:03d}"
        local = OUTDIR / part_name
        part_sha = download_range(url, start, end, local)
        with local.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                full_hash.update(block)
        upload_asset(local, part_sha)
        rows.append({
            "name": part_name,
            "offset": start,
            "bytes": end - start + 1,
            "sha256": part_sha,
        })
        local.unlink()
        print(f"VERIFIED AND PUBLISHED {index + 1}/{count}: {part_name}", flush=True)

    actual_full_hash = full_hash.hexdigest()
    if actual_full_hash != SHA256:
        raise RuntimeError(f"FULL FILE SHA256 MISMATCH: {actual_full_hash} != {SHA256}")
    print(f"FULL FILE SHA256 PASS: {actual_full_hash}", flush=True)

    manifest = {
        "format": "lfm3-200mb-parts-v1",
        **provenance,
        "chunk_size_bytes": PART_SIZE,
        "parts": rows,
        "license_asset": LICENSE_ASSET,
    }
    manifest_path = OUTDIR / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    upload_asset(manifest_path, sha256_file(manifest_path))

    license_source = Path(hf_hub_download(repo_id=REPO, filename="LICENSE", revision=REVISION))
    license_copy = OUTDIR / LICENSE_ASSET
    shutil.copyfile(license_source, license_copy)
    upload_asset(license_copy, sha256_file(license_copy))

    gh("release", "edit", TAG, "--draft=false")
    print(
        f"COMPLETE: https://github.com/{os.environ['GH_REPO']}/releases/tag/{TAG}",
        flush=True,
    )


if __name__ == "__main__":
    main()
