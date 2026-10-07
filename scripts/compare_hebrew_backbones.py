#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    rows=[]
    for p in sorted(Path(args.root).rglob("*.json")):
        try:
            x=json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "model_id" not in x or "full_training" not in x:
            continue
        gens=x.get("generations",[])
        avg_heb=(sum(g["diagnostics"]["hebrew_char_ratio"] for g in gens)/len(gens)) if gens else None
        avg_rep=(sum(g["diagnostics"]["repeated_4gram_ratio"] for g in gens)/len(gens)) if gens else None
        rows.append({
            "model_id":x["model_id"],
            "revision":x["resolved_revision"],
            "parameters":x["parameters"],
            "context_limit":x["context_limit_detected"],
            "vocab_size":x["tokenizer"]["vocab_size"],
            "chars_per_token":x["tokenizer"]["chars_per_token"],
            "bits_per_char":x["language_modeling"]["bits_per_char"],
            "nats_per_char":x["language_modeling"]["nats_per_char"],
            "eval_tokens_per_second":x["language_modeling"]["eval_tokens_per_second"],
            "full_train_tokens_per_second":x["full_training"]["tokens_per_second"],
            "full_train_max_rss_mb":x["full_training"]["max_process_rss_mb"],
            "all_parameters_trainable":x["full_training"]["all_parameters_trainable"],
            "avg_generated_hebrew_char_ratio":avg_heb,
            "avg_generated_repeated_4gram_ratio":avg_rep,
            "source_file":str(p),
        })
    rows.sort(key=lambda r:(r["bits_per_char"], -r["full_train_tokens_per_second"]))
    payload={
        "schema_version":1,
        "comparison_basis":"Same deterministic FineWeb2 Hebrew raw-text slice; cross-tokenizer LM comparison normalized as bits/character. Full-model AdamW training measured at batch=1, seq=128 on GitHub-hosted CPU runner.",
        "automatic_ranking_note":"Automatic ranking is diagnostic only. Final selection must also inspect Hebrew generations manually because base-LM fluency/coherence is not captured by bits/character alone.",
        "rows":rows,
    }
    Path(args.out).write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()
