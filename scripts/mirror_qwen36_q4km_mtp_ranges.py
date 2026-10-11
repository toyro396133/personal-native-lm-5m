#!/usr/bin/env python3
"""Stream Qwen3.6 35B Q4_K_M and matching MTP Q4_K_M as 200 MB GitHub release assets.
Do not stage the 22 GB original file on a GitHub runner. Validate each HTTP Range
and the full original SHA256 (as reported by Hugging Face's immutable revision).
"""
import hashlib
import json
import os
import pathlib
import re
import subprocess
import time

import requests
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

TAG = "qwen3.6-35b-a3b-q4km-mtp-200mb"
PART_SIZE = 200_000_000
SOURCES = [
    ("model", "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF", "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"),
    ("mtp", "tomngdev/MTP-Qwen3.6-35B-A3B-GGUF", "mtp-Q4_K_M.gguf"),
]
RETRIES = 5
OUTDIR = pathlib.Path("transfer_tmp")
OUTDIR.mkdir(exist_ok=True)


def gh(*args, check=True):
    r = subprocess.run(["gh", *args], capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"gh {' '.join(args[:3])} failed: {r.stderr[-1000:]}")
    return r


def get_release():
    res = gh("release", "view", TAG, "--json", "isDraft,assets,body", check=False)
    if res.returncode:
        return None
    return json.loads(res.stdout)


def source_metadata():
    api = HfApi()
    records = []
    for kind, repo, name in SOURCES:
        revision = api.model_info(repo).sha
        url = hf_hub_url(repo_id=repo, filename=name, revision=revision)
        meta = get_hf_file_metadata(url)
        size = int(meta.size)
        digest = str(meta.etag).lower().strip('"')
        if size < 1_000_000 or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(f"Missing reliable Xet/LFS metadata: {repo}/{name}, {size}, {digest}")
        records.append(dict(kind=kind, repo=repo, name=name, revision=revision,
                            bytes=size, sha256=digest, url=url))
        print(f"UPSTREAM {kind}: {repo}@{revision} {name}, {size} B sha256={digest}", flush=True)
    return records


def download_range(record, start, end, target):
    url = record["url"]
    expected_len = end - start + 1
    for attempt in range(1, RETRIES + 1):
        response = None
        try:
            response = requests.get(
                url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                stream=True, timeout=(40, 120), allow_redirects=True)
            if response.status_code != 206:
                raise RuntimeError(f"HTTP {response.status_code} ignored requested Range; "
                                   "refusing to download the complete model into runner storage")
            content_range = response.headers.get("Content-Range", "").strip()
            wanted = f"bytes {start}-{end}/{record['bytes']}"
            if content_range != wanted:
                raise RuntimeError(f"Invalid Content-Range: {content_range!r}, expected {wanted!r}")
            length_header = response.headers.get("Content-Length")
            if length_header and int(length_header) != expected_len:
                raise RuntimeError(f"Incorrect HTTP Content-Length: {length_header}")
            digest = hashlib.sha256()
            count = 0
            with target.open("wb") as writer:
                for chunk in response.iter_content(4 * 1024 * 1024):
                    if chunk:
                        count += len(chunk)
                        if count > expected_len:
                            raise RuntimeError("Server sent more data than requested")
                        digest.update(chunk)
                        writer.write(chunk)
            if count != expected_len:
                raise RuntimeError(f"Truncated range: {count}/{expected_len}")
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


def upload_part(local_path, expected_sha):
    name = local_path.name
    for attempt in range(1, RETRIES + 1):
        r = gh("release", "upload", TAG, str(local_path), check=False)
        if r.returncode == 0:
            break
        print(f"Upload {name} retry {attempt}: {r.stderr[-350:]}", flush=True)
        if attempt == RETRIES:
            raise RuntimeError(f"GitHub upload failed: {name}")
        time.sleep(min(30, 4 * attempt))
    uploaded = next((a for a in get_release()["assets"] if a["name"] == name), None)
    if not uploaded or uploaded["size"] != local_path.stat().st_size:
        raise RuntimeError(f"GitHub asset verification failed after upload: {name}")
    remote_sha = (uploaded.get("digest") or "").lower()
    if remote_sha and remote_sha != "sha256:" + expected_sha:
        raise RuntimeError(f"GitHub digest mismatch: {name} {remote_sha}")


def main():
    records = source_metadata()
    provenance = {"files": [{k: r[k] for k in ("kind","repo","name","revision","bytes","sha256")}
                            for r in records]}
    fingerprint = json.dumps(provenance, sort_keys=True)
    rel = get_release()
    if rel:
        if not rel["isDraft"] or fingerprint not in rel["body"]:
            raise RuntimeError("Existing tag published or tied to different source files")
        print("RESUME: matching draft release", flush=True)
    else:
        notes = (
            "Two matching files for Qwen3.6-35B-A3B: main Q4_K_M GGUF and MTP Q4_K_M GGUF. "
            "These are raw numbered chunks (NOT zip) intended for Windows downloads. "
            "Copyright remains with the original authors. License Apache-2.0.\\n\\n"
            "PROVENANCE_JSON=" + fingerprint
        )
        gh("release", "create", TAG, "--draft", "--target", "main",
           "--title", "Qwen3.6 35B-A3B Q4_K_M + MTP Q4_K_M (200MB parts)",
           "--notes", notes)

    manifests = []
    for record in records:
        large_hash = hashlib.sha256()
        rows = []
        count = (record["bytes"] + PART_SIZE - 1) // PART_SIZE
        for idx in range(count):
            start = idx * PART_SIZE
            end = min(start + PART_SIZE, record["bytes"]) - 1
            name = f"{record['kind']}-{record['name']}.part{idx+1:03d}"
            target = OUTDIR / name
            sha = download_range(record, start, end, target)
            with target.open("rb") as f:
                for buf in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    large_hash.update(buf)
            asset = next((a for a in get_release()["assets"] if a["name"] == name), None)
            if asset:
                if int(asset["size"]) != target.stat().st_size:
                    raise RuntimeError(f"Existing asset size mismatch: {name}")
                remote_sha = (asset.get("digest") or "").lower()
                if remote_sha and remote_sha != "sha256:" + sha:
                    raise RuntimeError(f"Existing asset SHA256 mismatch: {name}")
                print(f"Validated pre-existing {record['kind']} part {idx+1}/{count}", flush=True)
            else:
                upload_part(target, sha)
                print(f"UPLOADED {record['kind']} part {idx+1}/{count}", flush=True)
            rows.append({"name": name, "offset": start, "bytes": end-start+1, "sha256": sha})
            target.unlink()

        actual = large_hash.hexdigest()
        if actual != record["sha256"]:
            raise RuntimeError(f"FULL FILE HASH MISMATCH for {record['kind']}: {actual} vs {record['sha256']}")
        manifests.append({k: record[k] for k in ("kind","repo","name","revision","bytes","sha256")} |
                         {"parts": rows})
        print(f"FULL SHA256 PASS for {record['kind']}: {actual}", flush=True)

    manifest = {"format": "qwen36-200mb-parts-v1", "part_size": PART_SIZE, "files": manifests,
                "license": "Apache-2.0"}
    path = OUTDIR / "qwen3.6-35b-a3b-q4km-mtp.manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    previous = next((a for a in get_release()["assets"] if a["name"] == path.name), None)
    if previous:
        if previous["size"] != path.stat().st_size:
            raise RuntimeError("Existing manifest differs; manual inspection required")
    else:
        upload_part(path, hashlib.sha256(path.read_bytes()).hexdigest())

    gh("release", "edit", TAG, "--draft=false")
    print(f"COMPLETE: https://github.com/{os.environ['GH_REPO']}/releases/tag/{TAG}", flush=True)


if __name__ == "__main__":
    main()
