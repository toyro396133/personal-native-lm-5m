#!/usr/bin/env python3
"""Mirror the official DictaLM 3.0 1.7B Thinking Q4_K_M GGUF to GitHub Releases."""
import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
from huggingface_hub import hf_hub_download, hf_hub_url, get_hf_file_metadata

REPO = "dicta-il/DictaLM-3.0-1.7B-Thinking-GGUF"
FILE = "DictaLM-3.0-1.7B-Thinking-Q4_K_M.gguf"
EXPECTED_SHA256 = "a92a6b58ea48a3ea1b6099bda070bb0dab440b17e2afc527874827be96d28c94"
TAG = "dictalm-3.0-1.7b-thinking-q4km-200mb"
MANIFEST = "dictalm-3.0-1.7b-thinking.manifest.json"
INSTALLER = "install_dictalm_3_0_1_7b_thinking.ps1"
PART_SIZE = 200_000_000
GH_REPO = os.environ.get("GH_REPO", "toyro396133/personal-native-lm-5m")
WORK = Path("dictalm_work")
DIST = WORK / "dist"

def run(*args):
    p = subprocess.run(["gh", *args], text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError(f"gh {' '.join(args[:3])}: {p.stderr[-1600:]}")
    return p.stdout

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def release():
    p = subprocess.run(["gh", "release", "view", TAG, "-R", GH_REPO,
                        "--json", "isDraft,assets,body"], text=True, capture_output=True)
    return json.loads(p.stdout) if p.returncode == 0 else None

def upload(path):
    path = Path(path)
    current = release()
    existing = next((a for a in current["assets"] if a["name"] == path.name), None) if current else None
    digest = sha(path)
    if existing and int(existing["size"]) == path.stat().st_size and (existing.get("digest") or "").lower() == "sha256:" + digest:
        print("ALREADY VERIFIED:", path.name, flush=True)
        return
    args = ["release", "upload", TAG, str(path), "-R", GH_REPO]
    if existing:
        args.append("--clobber")
    run(*args)
    current = release()
    asset = next((a for a in current["assets"] if a["name"] == path.name), None)
    if not asset or int(asset["size"]) != path.stat().st_size or (asset.get("digest") or "").lower() != "sha256:" + digest:
        raise RuntimeError("GitHub Release asset failed remote verification: " + path.name)
    print("UPLOADED AND VERIFIED:", path.name, flush=True)

def main():
    if not GH_REPO:
        raise RuntimeError("GH_REPO is missing")
    WORK.mkdir(exist_ok=True)
    DIST.mkdir(exist_ok=True)
    url = hf_hub_url(REPO, FILE)
    meta = get_hf_file_metadata(url)
    if not meta.size or int(meta.size) < 1_000_000_000:
        raise RuntimeError(f"Unexpected upstream file size: {meta.size}")
    source = Path(hf_hub_download(REPO, FILE, force_download=True, cache_dir=str(WORK / "hf-cache")))
    if source.stat().st_size != int(meta.size):
        raise RuntimeError("Downloaded file size does not match Hugging Face metadata")
    actual = sha(source)
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"Official SHA-256 mismatch: {actual}")
    provenance = {"source_repo": REPO, "source_file": FILE, "source_sha256": EXPECTED_SHA256,
                  "size_bytes": int(meta.size), "chunk_size_bytes": PART_SIZE, "license": "Apache-2.0"}
    fingerprint = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    current = release()
    if current:
        if not current["isDraft"] or fingerprint not in (current.get("body") or ""):
            raise RuntimeError("Existing release is published or has different provenance; refusing to overwrite")
    else:
        notes = ("Official DictaLM 3.0 1.7B Thinking Q4_K_M GGUF, mirrored without conversion. "
                 "The model is split into individually downloadable 200 MB assets. "
                 "The Windows installer verifies every part and the reconstructed file.\n\n"
                 f"Upstream: https://huggingface.co/{REPO}/blob/main/{FILE}\n"
                 "License: Apache-2.0.\n\nPROVENANCE_JSON=" + fingerprint)
        run("release", "create", TAG, "-R", GH_REPO, "--draft", "--target", "main",
            "--title", "DictaLM 3.0 1.7B Thinking Q4_K_M (200 MB parts)", "--notes", notes)
    parts, offset, index = [], 0, 0
    with source.open("rb") as f:
        while True:
            block = f.read(PART_SIZE)
            if not block:
                break
            index += 1
            name = f"dictalm-3.0-1.7b-thinking-q4km.part{index:03d}"
            local = DIST / name
            local.write_bytes(block)
            digest = sha(local)
            upload(local)
            parts.append({"name": name, "offset": offset, "size_bytes": len(block), "sha256": digest})
            offset += len(block)
            local.unlink()
    if offset != int(meta.size):
        raise RuntimeError("Chunk lengths do not equal the upstream file size")
    manifest = {"format": "dictalm-3.0-1.7b-thinking-gguf-v1", **provenance,
                "files": [{"path": FILE, "size_bytes": int(meta.size),
                           "sha256": EXPECTED_SHA256, "parts": parts}]}
    mp = WORK / MANIFEST
    mp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    upload(mp)
    installer = Path("scripts") / INSTALLER
    if not installer.is_file():
        raise RuntimeError("Installer source file is missing: " + str(installer))
    staged = DIST / INSTALLER
    shutil.copyfile(installer, staged)
    upload(staged)
    staged.unlink()
    run("release", "edit", TAG, "-R", GH_REPO, "--draft=false")
    print(f"COMPLETE: https://github.com/{GH_REPO}/releases/tag/{TAG}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr, flush=True)
        raise
