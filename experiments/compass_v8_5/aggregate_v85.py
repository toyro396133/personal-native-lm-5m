"""Aggregate 3 ranks × 3 seeds of COMPASS V8.5 without cherry-picking runs."""
import argparse
import json
import statistics
from pathlib import Path
from collections import defaultdict

RANKS=(4,16,64)
SEEDS=(11,19,37)
PARAMS={4:7680,16:30720,64:122880}

def aggregate(folder):
    records=[]
    for rank in RANKS:
        for seed in SEEDS:
            path=Path(folder)/f"compass-v85-r{rank}-s{seed}.json"
            if not path.exists():raise FileNotFoundError(f"Missing required arm: {path}")
            a=json.loads(path.read_text(encoding="utf8"))
            assert a["rank"]==rank and a["seed"]==seed
            assert a["trainable_parameters"]==PARAMS[rank]
            assert a["locked_test"]==144
            main=a["scores"]["head_raw"];freeze=a["scores"]["frozen"]
            assert main["total"]==freeze["total"]==144
            expected_count={k:v["total"] for k,v in main["by_policy"].items()}
            assert expected_count=={"rapid":72,"thorough":72},expected_count
            assert set(main["by_language"])=={"en","he"}
            cases=a["decisions"]
            unseen=[x for x in cases if x["kind"]=="source_retracted"]
            old=[x for x in cases if x["kind"]!="source_retracted"]
            assert len(unseen)==16 and len(old)==128,(len(unseen),len(old))
            unseen_right=sum(x["prediction"]["head_raw"]==x["gold"] for x in unseen)
            old_right=sum(x["prediction"]["head_raw"]==x["gold"] for x in old)
            record={
              "rank":rank,"seed":seed,"parameters":PARAMS[rank],
              "raw_correct":main["correct"],"frozen_correct":freeze["correct"],
              "guarded_correct":a["scores"]["head_guarded"]["correct"],
              "fact_pairs":main["contrast"]["correct_fact_flip_pairs"],
              "full_groups":main["contrast"]["all_four_correct"],
              "english_correct":main["by_language"]["en"]["correct"],
              "hebrew_correct":main["by_language"]["he"]["correct"],
              "rapid_correct":main["by_policy"]["rapid"]["correct"],
              "thorough_correct":main["by_policy"]["thorough"]["correct"],
              "new_family_correct":unseen_right,
              "familiar_families_correct":old_right,
              "development_epoch":a["selected_epoch"],
              "feature_sha256":a["matched_feature_digest"],
            }
            records.append(record)
    assert len({x["feature_sha256"] for x in records})==1
    grouped=defaultdict(list)
    for x in records:grouped[x["rank"]].append(x)
    summary=[]
    for rank in RANKS:
        lines=grouped[rank]
        fmt=lambda key:{ "mean":round(statistics.mean(x[key] for x in lines),3),
                          "range":[min(x[key] for x in lines),max(x[key] for x in lines)]}
        summary.append({
          "rank":rank,"parameters":PARAMS[rank],"n_seeds":3,
          "accuracy_correct_of_144":fmt("raw_correct"),
          "fact_pairs_of_72":fmt("fact_pairs"),
          "quadruples_of_36":fmt("full_groups"),
          "english_correct_of_72":fmt("english_correct"),
          "hebrew_correct_of_72":fmt("hebrew_correct"),
          "rapid_correct_of_72":fmt("rapid_correct"),
          "thorough_correct_of_72":fmt("thorough_correct"),
          "novel_retracted_correct_of_16":fmt("new_family_correct"),
        })
    return {
      "experiment":"V8.5 complete nine-arm small/big exact personalized ranker",
      "fixed_unlearned_backbone":"HuggingFaceTB/SmolLM2-360M-Instruct",
      "fixed_feature_hash":records[0]["feature_sha256"],
      "results":records,"by_size":summary,
      "source_testing_note":"Synthetic generator-authored examples are not independent human-reviewed gold.",
      "interpretation_limit":"Multiple seeds and projects improve reproducibility but do not prove deployed personal prioritization.",
    }

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="v85-results")
    p.add_argument("--json",default="compass-v85-summary.json")
    p.add_argument("--markdown",default="compass-v85-summary.md")
    args=p.parse_args()
    data=aggregate(args.input)
    Path(args.json).write_text(json.dumps(data,indent=2),encoding="utf8")
    lines=[
      "# COMPASS V8.5 — nine-arm synthetic validation",
      "",
      "Frozen 360M backbone, EXACT trained personal parameter counts:",
      "",
      "| Rank | Personal weights | Mean correct /144 | Range /144 | Mean fact flips /72 | Mean full groups /36 | Mean Hebrew /72 | Mean newly retracted-source /16 |",
      "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in data["by_size"]:
        v=lambda k:r[k]
        lines.append(
            f"| {r['rank']} | {r['parameters']:,} | "+
            f"{v('accuracy_correct_of_144')['mean']} | {v('accuracy_correct_of_144')['range']} | "+
            f"{v('fact_pairs_of_72')['mean']} | {v('quadruples_of_36')['mean']} | "+
            f"{v('hebrew_correct_of_72')['mean']} | {v('novel_retracted_correct_of_16')['mean']} |"
        )
    lines+=["","## Per-run raw metrics","","| Rank | Seed | Raw correct /144 | Frozen /144 | Guarded /144 | Fact flips /72 | Complete quadruples /36 | Hebrew /72 | New source-family /16 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in data["results"]:
        lines.append(f"| {r['rank']} | {r['seed']} | {r['raw_correct']} | {r['frozen_correct']} | "+
                     f"{r['guarded_correct']} | {r['fact_pairs']} | {r['full_groups']} | "+
                     f"{r['hebrew_correct']} | {r['new_family_correct']} |")
    lines+=["","**Limitations:** Synthetic research-authored scenarios, same 8 scenario families in train and familiar test; "+
            "new source-retraction family is heldout. Two opposite explicitly fictitious user policies are present, "+
            "not individualized real user histories. External permission guards are not learned competence. "+
            "Rank comparisons use the exact same frozen feature cache and learning schedule per seed.",
            ""]
    Path(args.markdown).write_text("\n".join(lines),encoding="utf8")
    print(Path(args.markdown).read_text(encoding="utf8"),flush=True)
