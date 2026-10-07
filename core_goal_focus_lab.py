from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from eval_v17 import load_any
from model_self_variants import SelfVariantPersonalNativeLM
from personal_state import PersonalState
from train_hebrew import load_tokenizer


CORES = [
    "לשמור על יציבות המערכת",
    "לבנות בסיס נתונים אמין",
    "להפחית שגיאות לאורך זמן",
    "לשמור על פשטות המבנה",
    "לשפר את איכות התשובות",
    "לשמור על עקביות התהליך",
    "להעדיף פתרון שניתן לתחזוקה",
    "לשמור על הפרדה ברורה בין אחריות הרכיבים",
]

GOALS = [
    "להשלים מנגנון ייבוא נתונים",
    "לתקן את מנגנון הסינון",
    "להוסיף בדיקת תקינות",
    "לשפר את מסך הניהול",
    "להשלים תהליך מיזוג",
    "לצמצם כפילויות בנתונים",
    "להוסיף שמירת מצב",
    "לתקן את מסלול העדכון",
    "להכין דוח השוואה",
    "להשלים מנגנון הרשאות",
    "לשפר את זמן הטעינה",
    "לייצב את תהליך ההפעלה",
]

FOCI = [
    "בדיקת קובץ התצורה",
    "תיקון שאילתת המסד",
    "קריאת יומן השגיאות",
    "בדיקת נתוני הקלט",
    "עדכון פונקציית הסינון",
    "הרצת בדיקה ממוקדת",
    "השוואת שתי תוצאות",
    "תיקון שם שדה",
    "בדיקת סדר הפעולות",
    "עדכון שכבת התצוגה",
    "בדיקת שמירת המצב",
    "תיקון נתיב הקובץ",
    "בדיקת תגובת השרת",
    "עדכון תיעוד קצר",
    "בדיקת תרחיש קצה",
    "אימות תוצאת החישוב",
]

TRAIN_TEMPLATES = {
    "irrelevant": [
        "נמצא פרט צדדי שאינו משנה את המשימה הנוכחית. המשך ללא שינוי.",
        "התקבל מידע נוסף, אבל אין לו השפעה על המטרה או על המוקד כעת.",
        "נוספה הערה כללית בלבד; אין צורך לשנות את העבודה.",
    ],
    "focus": [
        "כדי להתקדם, המוקד עובר כעת אל: {focus}.",
        "יש לטפל עכשיו בתת המשימה הבאה: {focus}.",
        "השלב הבא בתוך אותה מטרה הוא: {focus}.",
    ],
    "goal": [
        "המטרה הקודמת הוחלפה. המטרה החדשה: {goal}. מתחילים ב: {focus}.",
        "הוחלט לשנות את המטרה ל-{goal}; המוקד הראשון יהיה {focus}.",
        "מעתה היעד הפעיל הוא {goal}, והפעולה הנוכחית היא {focus}.",
    ],
    "goal_done": [
        "המטרה הנוכחית הושלמה. עוברים למטרה הבאה: {goal}. מתחילים ב-{focus}.",
        "היעד הושג; כעת ממשיכים אל {goal}, והמוקד הוא {focus}.",
    ],
    "core": [
        "הוגדר מחדש עקרון העל: {core}. בהתאם, המטרה כעת {goal} והמוקד {focus}.",
        "הליבה השתנתה במפורש ל-{core}; לכן עוברים ל-{goal} ומתחילים ב-{focus}.",
    ],
}

TEST_TEMPLATES = {
    "irrelevant": [
        "הגיע פרט נוסף שאינו קשור להחלטה הנוכחית. אין לשנות דבר.",
        "יש מידע רקע חדש, אך הוא אינו משנה את הליבה, המטרה או המוקד.",
    ],
    "focus": [
        "בלי להחליף את המטרה, יש להתמקד עכשיו ב-{focus}.",
        "המטרה נשארת כפי שהיא; הפעולה שדורשת תשומת לב כעת היא {focus}.",
    ],
    "goal": [
        "יש להפסיק את היעד הקודם ולעבור ל-{goal}; הצעד הראשון הוא {focus}.",
        "נקבע יעד חדש: {goal}. התחל כעת ב-{focus}.",
    ],
    "goal_done": [
        "לאחר השלמת היעד, היעד הבא הוא {goal}; הפעולה הראשונה: {focus}.",
        "המשימה הראשית הסתיימה בהצלחה, ועכשיו עוברים ל-{goal} עם מוקד {focus}.",
    ],
    "core": [
        "שינוי יסודי אושר: מעתה העיקרון המנחה הוא {core}. היעד הוא {goal}, והמוקד {focus}.",
        "המרכז הוגדר מחדש ל-{core}; בהתאם היעד החדש {goal} והפעולה הנוכחית {focus}.",
    ],
}


def sha_model(model: nn.Module) -> str:
    h = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            t = tensor.detach().cpu().contiguous()
            h.update(name.encode())
            h.update(str(tuple(t.shape)).encode())
            h.update(bytes(t.untyped_storage()))
    return h.hexdigest()


def different(rng, seq, current):
    choices = [x for x in seq if x != current]
    return rng.choice(choices)


def make_examples(n: int, seed: int, templates: dict):
    rng = random.Random(seed)
    kinds = ["irrelevant", "focus", "focus", "focus", "goal", "goal_done", "core"]
    out = []
    for i in range(n):
        core = rng.choice(CORES)
        goal = rng.choice(GOALS)
        focus = rng.choice(FOCI)
        kind = rng.choice(kinds)
        next_core, next_goal, next_focus = core, goal, focus

        if kind == "irrelevant":
            target = [0.0, 0.0, 0.0]
        elif kind == "focus":
            next_focus = different(rng, FOCI, focus)
            target = [0.0, 0.0, 1.0]
        elif kind in {"goal", "goal_done"}:
            next_goal = different(rng, GOALS, goal)
            next_focus = different(rng, FOCI, focus)
            target = [0.0, 1.0, 1.0]
        elif kind == "core":
            next_core = different(rng, CORES, core)
            next_goal = different(rng, GOALS, goal)
            next_focus = different(rng, FOCI, focus)
            target = [1.0, 1.0, 1.0]
        else:
            raise RuntimeError(kind)

        template = rng.choice(templates[kind])
        event = template.format(core=next_core, goal=next_goal, focus=next_focus)
        out.append({
            "id": i,
            "kind": kind,
            "core": core,
            "goal": goal,
            "focus": focus,
            "event": event,
            "target": target,
            "next_core": next_core,
            "next_goal": next_goal,
            "next_focus": next_focus,
        })
    return out


@torch.no_grad()
def encode_text(model, tok, text: str, device: str):
    ids = tok.encode(text, bos=True, eos=True)
    ids = ids[:96]
    xids = torch.tensor([ids], dtype=torch.long, device=device)
    neutral = PersonalState("neutral").flatten().unsqueeze(0).to(device)

    if isinstance(model, SelfVariantPersonalNativeLM):
        base = model.base
        x = base.token_embedding(xids)
        pos = torch.arange(xids.shape[1], device=device)
        x = x + base.position_embedding(pos)[None, :, :]
        for block, adapter in zip(base.blocks, model.self_adapters):
            x = block(x, None)
            x = adapter(x, model.self_anchor)
        x = base.final_norm(x)
    else:
        base = model
        x = base.token_embedding(xids)
        pos = torch.arange(xids.shape[1], device=device)
        x = x + base.position_embedding(pos)[None, :, :]
        for block in base.blocks:
            x = block(x, None)
        x = base.final_norm(x)

    return x.mean(dim=1).squeeze(0).cpu()


@torch.no_grad()
def build_feature_cache(model, tok, examples, device):
    texts = set()
    for e in examples:
        texts.update([e["core"], e["goal"], e["focus"], e["event"]])
    cache = {}
    for i, text in enumerate(sorted(texts)):
        cache[text] = encode_text(model, tok, text, device)
        if (i + 1) % 200 == 0:
            print(f"[encode] {i+1}/{len(texts)}", flush=True)
    return cache


def tensorize(examples, cache):
    return {
        "core": torch.stack([cache[e["core"]] for e in examples]),
        "goal": torch.stack([cache[e["goal"]] for e in examples]),
        "focus": torch.stack([cache[e["focus"]] for e in examples]),
        "event": torch.stack([cache[e["event"]] for e in examples]),
        "target": torch.tensor([e["target"] for e in examples], dtype=torch.float32),
        "kind": [e["kind"] for e in examples],
    }


class RelationEncoder(nn.Module):
    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2 * d_model + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

    def forward(self, x, anchor, mode: str):
        x = F.layer_norm(x, (x.shape[-1],))
        if mode == "relational":
            a = F.layer_norm(anchor, (anchor.shape[-1],)).view(1, -1).expand_as(x)
            cos = F.cosine_similarity(x, a, dim=-1, eps=1e-8).unsqueeze(-1)
            z = torch.cat((x - a, x * a, cos), dim=-1)
        elif mode == "direct":
            norm = x.norm(dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
            z = torch.cat((x, x * x, norm), dim=-1)
        else:
            raise ValueError(mode)
        return self.proj(z)


class CoreGoalFocusController(nn.Module):
    def __init__(self, d_model: int, mode: str, hidden: int = 64):
        super().__init__()
        self.mode = mode
        self.relation = RelationEncoder(d_model, hidden)
        self.core_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.goal_head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.focus_head = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, core, goal, focus, event, anchor):
        c = self.relation(core, anchor, self.mode)
        g = self.relation(goal, anchor, self.mode)
        f = self.relation(focus, anchor, self.mode)
        e = self.relation(event, anchor, self.mode)
        core_logit = self.core_head(torch.cat((c, e), dim=-1))
        goal_logit = self.goal_head(torch.cat((c, g, e), dim=-1))
        focus_logit = self.focus_head(torch.cat((c, g, f, e), dim=-1))
        return torch.cat((core_logit, goal_logit, focus_logit), dim=-1)


def ablated_anchor(anchor, mode: str, seed: int = 991):
    if mode == "normal":
        return anchor
    if mode == "zero":
        return torch.zeros_like(anchor)
    if mode == "negate":
        return -anchor
    if mode == "shuffle":
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        idx = torch.randperm(anchor.numel(), generator=g).to(anchor.device)
        return anchor[idx]
    raise ValueError(mode)


@torch.no_grad()
def metrics(controller, data, anchor, device, anchor_mode="normal"):
    controller.eval()
    a = ablated_anchor(anchor, anchor_mode)
    logits = controller(
        data["core"].to(device),
        data["goal"].to(device),
        data["focus"].to(device),
        data["event"].to(device),
        a,
    )
    pred = (torch.sigmoid(logits) >= 0.5).float().cpu()
    y = data["target"]
    exact = (pred == y).all(dim=1).float()
    slot_acc = (pred == y).float().mean(dim=0)

    by_kind = {}
    for kind in sorted(set(data["kind"])):
        idx = torch.tensor([k == kind for k in data["kind"]], dtype=torch.bool)
        by_kind[kind] = float(exact[idx].mean())

    keep_core = y[:, 0] == 0
    keep_goal = y[:, 1] == 0
    change_core = y[:, 0] == 1
    change_goal = y[:, 1] == 1
    change_focus = y[:, 2] == 1

    def safe(mask, expression):
        return float(expression[mask].float().mean()) if bool(mask.any()) else None

    return {
        "exact_match": float(exact.mean()),
        "core_accuracy": float(slot_acc[0]),
        "goal_accuracy": float(slot_acc[1]),
        "focus_accuracy": float(slot_acc[2]),
        "core_false_update_rate": safe(keep_core, pred[:, 0] == 1),
        "goal_false_update_rate": safe(keep_goal, pred[:, 1] == 1),
        "core_change_recall": safe(change_core, pred[:, 0] == 1),
        "goal_change_recall": safe(change_goal, pred[:, 1] == 1),
        "focus_change_recall": safe(change_focus, pred[:, 2] == 1),
        "by_kind_exact": by_kind,
    }


def train_one(train, test, anchor, d_model, mode, seed, device, epochs, lr):
    torch.manual_seed(seed)
    controller = CoreGoalFocusController(d_model, mode=mode).to(device)
    opt = torch.optim.AdamW(controller.parameters(), lr=lr, weight_decay=0.01)

    x = {k: v.to(device) for k, v in train.items() if torch.is_tensor(v)}
    n = x["target"].shape[0]
    batch = 128
    for epoch in range(epochs):
        g = torch.Generator(device="cpu")
        g.manual_seed(seed + epoch)
        order = torch.randperm(n, generator=g)
        controller.train()
        for start in range(0, n, batch):
            idx = order[start:start+batch].to(device)
            logits = controller(x["core"][idx], x["goal"][idx], x["focus"][idx], x["event"][idx], anchor)
            loss = F.binary_cross_entropy_with_logits(logits, x["target"][idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
            opt.step()

    result = {
        "seed": seed,
        "mode": mode,
        "parameter_count": sum(p.numel() for p in controller.parameters()),
        "normal": metrics(controller, test, anchor, device, "normal"),
    }
    if mode == "relational":
        for a_mode in ("zero", "shuffle", "negate"):
            result[f"anchor_{a_mode}"] = metrics(controller, test, anchor, device, a_mode)
    return result


def aggregate(runs):
    normal = [r["normal"] for r in runs]
    keys = [
        "exact_match", "core_accuracy", "goal_accuracy", "focus_accuracy",
        "core_false_update_rate", "goal_false_update_rate",
        "core_change_recall", "goal_change_recall", "focus_change_recall",
    ]
    result = {}
    for key in keys:
        vals = [x[key] for x in normal if x[key] is not None]
        result[key] = {
            "mean": statistics.mean(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }
    if runs[0]["mode"] == "relational":
        result["causal_anchor"] = {}
        base = result["exact_match"]["mean"]
        for am in ("zero", "shuffle", "negate"):
            vals = [r[f"anchor_{am}"]["exact_match"] for r in runs]
            mean = statistics.mean(vals)
            result["causal_anchor"][am] = {
                "exact_match_mean": mean,
                "delta_vs_normal": mean - base,
            }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-examples", type=int, default=3200)
    ap.add_argument("--test-examples", type=int, default=1200)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=0.0015)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt, cfg, model = load_any(args.checkpoint, args.device)
    tok, _ = load_tokenizer(args.tokenizer)
    variant = ckpt.get("variant", "baseline")
    before_sha = sha_model(model)

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    train_examples = make_examples(args.train_examples, 4401, TRAIN_TEMPLATES)
    test_examples = make_examples(args.test_examples, 9923, TEST_TEMPLATES)
    cache = build_feature_cache(model, tok, train_examples + test_examples, args.device)
    train = tensorize(train_examples, cache)
    test = tensorize(test_examples, cache)

    if isinstance(model, SelfVariantPersonalNativeLM):
        anchor = model.self_anchor.detach().clone().to(args.device)
        modes = ["relational", "direct"]
    else:
        anchor = torch.zeros(cfg.d_model, device=args.device)
        modes = ["direct"]

    seeds = [101, 202, 303]
    all_runs = {}
    for mode in modes:
        mode_runs = []
        for seed in seeds:
            print(f"[train] variant={variant} mode={mode} seed={seed}", flush=True)
            mode_runs.append(
                train_one(train, test, anchor, cfg.d_model, mode, seed, args.device, args.epochs, args.lr)
            )
        all_runs[mode] = {
            "runs": mode_runs,
            "aggregate": aggregate(mode_runs),
        }

    after_sha = sha_model(model)
    if before_sha != after_sha:
        raise SystemExit("frozen 15M model changed during CORE/GOAL/FOCUS lab")

    result = {
        "experiment": "v0.16c CORE-GOAL-FOCUS side lab",
        "source_checkpoint": args.checkpoint,
        "source_variant": variant,
        "source_tokens_seen": ckpt.get("tokens_seen"),
        "source_model_frozen": True,
        "frozen_model_sha256_before": before_sha,
        "frozen_model_sha256_after": after_sha,
        "frozen_model_unchanged": before_sha == after_sha,
        "semantics": {
            "SELF": "stable reference supplied by the frozen 15M model when available",
            "CORE": "persistent governing principle; changes only on explicit core revision",
            "GOAL": "active objective; changes on explicit replacement or completion",
            "FOCUS": "current subtask; may change while CORE and GOAL remain stable",
        },
        "dataset": {
            "train_examples": len(train_examples),
            "test_examples": len(test_examples),
            "train_template_family": "train-only wording",
            "test_template_family": "held-out wording",
        },
        "modes": all_runs,
    }

    if "relational" in all_runs:
        rel = all_runs["relational"]["aggregate"]["exact_match"]["mean"]
        direct = all_runs["direct"]["aggregate"]["exact_match"]["mean"]
        result["relational_minus_direct_exact_match"] = rel - direct

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
