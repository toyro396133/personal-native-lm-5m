#!/usr/bin/env python3
"""UNIFOLD v0: deterministic symbolic rule-learning episodes; standard library only.

TRAIN: unseen inputs for held-in L1 semantic functions.
L1 heldout: heldout functions of exactly two primitive steps.
L2: three-step functions proven distinct from every <=2-step function on the
ENTIRE finite input domain. L2 does not appear in training.

This tests structural extrapolation, not yet improvement in learning-to-learn.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
from typing import Sequence

SCHEMA = "unifold-taskgen-v0.1"
WIDTH = 4
ALPHABET = 4
DOMAIN = tuple(itertools.product(range(ALPHABET), repeat=WIDTH))
OPS = (
    "reverse", "rotate_left", "rotate_right", "swap_0_1", "swap_1_2",
    "swap_2_3", "increment_all", "decrement_all", "increment_0",
    "increment_2", "xor_0_1", "add_last_to_first", "copy_first_to_last",
    "add_first_to_last",
)
SPLIT_COUNTS_10000 = {
    "train_l1": 6000, "val_l1": 1000, "test_l1": 1000,
    "val_l2": 1000, "test_l2": 1000,
}


def op(name: str, data: tuple[int, ...]) -> tuple[int, ...]:
    a = list(data)
    if name == "reverse":
        return tuple(reversed(a))
    if name == "rotate_left":
        return tuple(a[1:] + a[:1])
    if name == "rotate_right":
        return tuple(a[-1:] + a[:-1])
    if name.startswith("swap_"):
        _, i, j = name.split("_")
        a[int(i)], a[int(j)] = a[int(j)], a[int(i)]
    elif name == "increment_all":
        a = [(v + 1) % ALPHABET for v in a]
    elif name == "decrement_all":
        a = [(v - 1) % ALPHABET for v in a]
    elif name == "increment_0":
        a[0] = (a[0] + 1) % ALPHABET
    elif name == "increment_2":
        a[2] = (a[2] + 1) % ALPHABET
    elif name == "xor_0_1":
        a[0] ^= a[1]
    elif name == "add_last_to_first":
        a[0] = (a[0] + a[-1]) % ALPHABET
    elif name == "copy_first_to_last":
        a[-1] = a[0]
    elif name == "add_first_to_last":
        a[-1] = (a[-1] + a[0]) % ALPHABET
    else:
        raise ValueError(f"Unknown primitive: {name}")
    return tuple(a)


def apply_program(program: Sequence[str], data: tuple[int, ...]) -> tuple[int, ...]:
    for name in program:
        data = op(name, data)
    return data


def encode(data: Sequence[int]) -> int:
    out = 0
    for value in data:
        out = out * ALPHABET + value
    return out


def fingerprint(signature: bytes) -> str:
    return hashlib.sha256(signature).hexdigest()


def hash_split(prefix: str, signature: bytes, seed: int, train=False) -> str:
    h = hashlib.sha256(f"{SCHEMA}/{seed}/{prefix}/".encode() + signature).digest()
    bucket = int.from_bytes(h[:8], "big") % 100
    if prefix == "l1":
        return "train" if bucket < 65 else ("val" if bucket < 82 else "test")
    return "val" if bucket < 50 else "test"


def make_pool(seed: int) -> dict[str, list[tuple[tuple[str, ...], bytes]]]:
    """Exhaustively deduplicate behavioral functions on all 4^4 inputs."""
    transitions = {name: bytes(encode(op(name, value)) for value in DOMAIN) for name in OPS}
    elementary = set(transitions.values())
    seen_l1: set[bytes] = set()
    pool = {"train_l1": [], "val_l1": [], "test_l1": [], "val_l2": [], "test_l2": []}
    pairs: list[tuple[tuple[str, str], bytes]] = []
    for left, right in itertools.product(OPS, repeat=2):
        sig = bytes(transitions[right][v] for v in transitions[left])
        pairs.append(((left, right), sig))
        if sig in seen_l1 or sig in elementary:
            continue
        seen_l1.add(sig)
        split = hash_split("l1", sig, seed)
        pool[f"{split}_l1"].append(((left, right), sig))
    seen_l2: set[bytes] = set()
    for (left, middle), pair_sig in pairs:
        for right in OPS:
            sig = bytes(transitions[right][v] for v in pair_sig)
            # No new level if behavior is equivalent to a shallower rule.
            if sig in elementary or sig in seen_l1 or sig in seen_l2:
                continue
            seen_l2.add(sig)
            split = hash_split("l2", sig, seed)
            pool[f"{split}_l2"].append(((left, middle, right), sig))
    if any(not p for p in pool.values()):
        raise RuntimeError(f"Missing split or structural level: {[k for k,v in pool.items() if not v]}")
    return pool


def counts_for(total: int) -> dict[str, int]:
    if total < 5:
        raise ValueError("At least five episodes are needed for five test buckets")
    base = {key: total * count // 10000 for key, count in SPLIT_COUNTS_10000.items()}
    for key in SPLIT_COUNTS_10000:
        if base[key] == 0:
            base[key] = 1
    extra = total - sum(base.values())
    if extra < 0:
        # Can occur only for very small totals. Start one per bucket instead.
        base = dict.fromkeys(SPLIT_COUNTS_10000, 1)
        extra = total - 5
    for key in ("train_l1", "val_l1", "test_l1", "val_l2", "test_l2"):
        if extra <= 0:
            break
        base[key] += extra
        extra -= 1
    return base


def serialize(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def generate(out: Path, count: int = 10000, seed: int = 42, support_size: int = 12,
             query_size: int = 16) -> dict:
    if support_size < 1 or query_size < 1 or support_size + query_size > len(DOMAIN):
        raise ValueError("Invalid support/query sizes")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    pool = make_pool(seed)
    candidate_functions = {level: [sig for key, rows in pool.items() if key.endswith(level) for _, sig in rows] for level in ("l1", "l2")}
    buckets = [key for key, n in counts_for(count).items() for _ in range(n)]
    rng.shuffle(buckets)
    tasks_path, answer_path = out / "tasks.jsonl", out / "answers.PRIVATE.jsonl"
    with tasks_path.open("w", encoding="utf-8", newline="\n") as tasks, answer_path.open("w", encoding="utf-8", newline="\n") as answers:
        for i, bucket in enumerate(buckets):
            program, sig = rng.choice(pool[bucket])
            candidates = candidate_functions[bucket.split("_")[1]]
            # A task is accepted only if its support uniquely identifies the
            # target among ALL distinct functions at that same level.
            for _ in range(100):
                points = rng.sample(range(len(DOMAIN)), support_size + query_size)
                supp, queries = points[:support_size], points[support_size:]
                if sum(1 for other in candidates if all(other[j] == sig[j] for j in supp)) == 1:
                    break
            else:
                raise RuntimeError(f"Insufficient support to distinguish rule in {bucket}; increase --support")
            uid = f"uf-{i:06d}"
            public = {
                "id": uid, "schema": SCHEMA, "split": bucket.split("_")[0],
                "level": bucket.split("_")[1].upper(),
                "support": [{"input": list(DOMAIN[j]), "output": list(DOMAIN[sig[j]])} for j in supp],
                "queries": [{"input": list(DOMAIN[j])} for j in queries],
            }
            private = {
                "id": uid, "program": list(program), "semantic_sha256": fingerprint(sig),
                "query_outputs": [list(DOMAIN[sig[j]]) for j in queries],
            }
            tasks.write(serialize(public) + "\n")
            answers.write(serialize(private) + "\n")
    manifest = {
        "schema": SCHEMA, "seed": seed, "count": count,
        "support_size": support_size, "query_size": query_size,
        "domain": {"width": WIDTH, "alphabet": ALPHABET, "inputs": len(DOMAIN)},
        "bucket_counts": counts_for(count),
        "unique_rule_counts": {name: len(rules) for name, rules in pool.items()},
        "identifiability": "Unique in same-level candidate function pool on support examples",
        "tasks_sha256": file_hash(tasks_path),
        "answers_sha256": file_hash(answer_path),
        "integrity_note": "PRIVATE answer file must never be given to model during evaluation; L2 absent from train",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def validate(out: Path) -> dict:
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != SCHEMA:
        raise ValueError("Unsupported dataset schema")
    for field, fname in (("tasks_sha256", "tasks.jsonl"), ("answers_sha256", "answers.PRIVATE.jsonl")):
        if manifest[field] != file_hash(out / fname):
            raise ValueError(f"Integrity mismatch: {fname}")
    pool = make_pool(manifest["seed"])
    valid = {key: {fingerprint(sig): p for p, sig in rules} for key, rules in pool.items()}
    candidate_functions = {level: [sig for key, rows in pool.items() if key.endswith(level) for _, sig in rows] for level in ("l1", "l2")}
    totals: Counter[str] = Counter()
    signatures = {key: set() for key in pool}
    seen_ids: set[str] = set()
    with (out / "tasks.jsonl").open(encoding="utf-8") as tasks, (out / "answers.PRIVATE.jsonl").open(encoding="utf-8") as answers:
        for line_no, (public_line, private_line) in enumerate(itertools.zip_longest(tasks, answers), 1):
            if public_line is None or private_line is None:
                raise ValueError("Unequal public/private record counts")
            public, private = json.loads(public_line), json.loads(private_line)
            uid = public["id"]
            if uid in seen_ids or uid != private["id"] or public["schema"] != SCHEMA:
                raise ValueError(f"Duplicate/misaligned episode at line {line_no}")
            seen_ids.add(uid)
            bucket = f'{public["split"]}_{public["level"].lower()}'
            if bucket not in valid:
                raise ValueError(f"Invalid split/level at {line_no}")
            if "program" in public or "query_outputs" in public or "semantic_sha256" in public:
                raise ValueError("Private labels leaked to public tasks")
            signature_id = private["semantic_sha256"]
            if signature_id not in valid[bucket]:
                raise ValueError(f"Law leaks outside assigned split/level: {uid}")
            program = tuple(private["program"])
            # Equivalent representative is acceptable, but verify its own fingerprint.
            direct_signature = bytes(encode(apply_program(program, d)) for d in DOMAIN)
            if fingerprint(direct_signature) != signature_id:
                raise ValueError(f"Bad oracle/program fingerprint: {uid}")
            if len(program) != (2 if public["level"] == "L1" else 3):
                raise ValueError(f"Invalid program depth: {uid}")
            if len(public["support"]) != manifest["support_size"] or len(public["queries"]) != manifest["query_size"]:
                raise ValueError(f"Wrong number of examples: {uid}")
            if len(private["query_outputs"]) != manifest["query_size"]:
                raise ValueError(f"Wrong number of gold labels: {uid}")
            used: set[tuple[int, ...]] = set()
            for pair in public["support"]:
                inp = tuple(pair["input"])
                if inp not in DOMAIN or inp in used or tuple(pair["output"]) != apply_program(program, inp):
                    raise ValueError(f"Incorrect support or duplicate input: {uid}")
                used.add(inp)
            observations = [(encode(pair["input"]), encode(pair["output"])) for pair in public["support"]]
            ambiguity = sum(1 for candidate in candidate_functions[public["level"].lower()]
                            if all(candidate[x] == y for x, y in observations))
            if ambiguity != 1:
                raise ValueError(f"Support does not uniquely identify law: {uid} ({ambiguity} candidates)")
            for item, expected in zip(public["queries"], private["query_outputs"]):
                inp = tuple(item["input"])
                if inp not in DOMAIN or inp in used or tuple(expected) != apply_program(program, inp):
                    raise ValueError(f"Incorrect oracle or duplicate input: {uid}")
                used.add(inp)
            totals[bucket] += 1
            signatures[bucket].add(signature_id)
    if dict(totals) != manifest["bucket_counts"] or len(seen_ids) != manifest["count"]:
        raise ValueError("Incorrect split totals")
    all_sig_sets = list(signatures.values())
    if any(left & right for i, left in enumerate(all_sig_sets) for right in all_sig_sets[i+1:]):
        raise ValueError("Semantic overlap across splits/levels")
    return {"passed": True, "episodes": len(seen_ids), "support_identifiability": "100% in declared same-level function class", "buckets": dict(totals),
            "distinct_functions_used": {k: len(v) for k, v in signatures.items()},
            "unique_pool": manifest["unique_rule_counts"], "tasks_sha256": manifest["tasks_sha256"],
            "answers_sha256": manifest["answers_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="UNIFOLD deterministic L1/L2 task dataset generator")
    commands = parser.add_subparsers(dest="command", required=True)
    g = commands.add_parser("generate")
    g.add_argument("--output", required=True, type=Path)
    g.add_argument("--count", type=int, default=10000)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--support", type=int, default=12)
    g.add_argument("--queries", type=int, default=16)
    v = commands.add_parser("validate")
    v.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = generate(args.output, args.count, args.seed, args.support, args.queries) if args.command == "generate" else validate(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
