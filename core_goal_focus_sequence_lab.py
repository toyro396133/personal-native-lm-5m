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


TRAIN_CORES = [
    "לשמור על יציבות המערכת",
    "לבנות בסיס נתונים אמין",
    "להפחית שגיאות לאורך זמן",
    "לשמור על פשטות המבנה",
    "לשפר את איכות התשובות",
    "לשמור על עקביות התהליך",
    "להעדיף פתרון שניתן לתחזוקה",
    "לשמור על הפרדה ברורה בין אחריות הרכיבים",
    "להגן על שלמות הנתונים",
    "להעדיף התנהגות צפויה",
    "לשמור על יכולת הסבר של החלטות",
    "להימנע מתלות מיותרת בין רכיבים",
]
TEST_CORES = [
    "לשמור על רציפות התפקוד גם תחת תקלות",
    "להעדיף נכונות לפני קיצור דרך",
    "לשמר עקביות בין החלטות לאורך זמן",
    "לשמור על מקור אמת ברור למידע",
    "להעדיף שינוי קטן שניתן לאימות",
    "למנוע ממשימות ביניים להשתלט על הכיוון",
]

TRAIN_GOALS = [
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
    "להשלים תהליך סנכרון",
    "לשפר את איתור השגיאות",
    "להכין מנגנון שחזור",
    "להשלים תהליך ניקוי נתונים",
]
TEST_GOALS = [
    "להוסיף מנגנון איחוד רשומות",
    "להשלים תהליך אימות משתמש",
    "לשפר את מנגנון החיפוש",
    "להוסיף שמירת היסטוריית שינויים",
    "לייצב את תהליך יצירת הדוחות",
    "לשפר את ניהול התורים",
    "להוסיף בדיקת תאימות",
    "להשלים מנגנון גיבוי",
]

TRAIN_FOCI = [
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
    "בדיקת הרשאות גישה",
    "תיקון טיפול בערך חסר",
    "השוואת גרסאות סכימה",
    "בדיקת תור המשימות",
]
TEST_FOCI = [
    "בדיקת מפתח הרשומה",
    "תיקון מיפוי עמודה",
    "בדיקת נעילת משאב",
    "אימות סדר המיזוג",
    "בדיקת מטמון התוצאות",
    "תיקון כלל אימות",
    "בדיקת זמן התגובה",
    "אימות כתיבת היסטוריה",
    "בדיקת תרחיש שחזור",
    "תיקון מעבר בין שלבים",
]


TRAIN_TEMPLATES = {
    "noise": [
        "עלה דיון צדדי על {decoy_goal}, אבל אין כרגע החלטה לשנות את הכיוון.",
        "נמצא פרט מעניין בנושא {decoy_focus}; הוא אינו דורש שינוי בעבודה הפעילה.",
        "מישהו הציע לעבור ל-{decoy_goal}, אך זו הצעה בלבד ולא החלטה.",
    ],
    "focus": [
        "כדי להתקדם בתוך היעד הקיים, מטפלים עכשיו ב-{focus}.",
        "היעד נשאר כפי שהוא, והצעד המקומי הבא הוא {focus}.",
        "נדרש טיפול מיידי ב-{focus}; לאחר מכן ממשיכים באותו יעד.",
    ],
    "tempting_focus": [
        "התגלתה תקלה חמורה ב-{focus}. היא דחופה, אבל היא רק תת-משימה בתוך היעד הקיים.",
        "כל תשומת הלב עוברת כרגע ל-{focus}; אין בכך החלפה של היעד הראשי.",
        "יש לעצור זמנית את הפעולה הנוכחית ולטפל ב-{focus}, בלי לשנות את מטרת העבודה.",
    ],
    "goal_pressure": [
        "עלה רעיון להפוך את {decoy_goal} ליעד הראשי, אבל טרם התקבלה החלטה ולכן נשארים ביעד הקיים.",
        "הצעה חדשה היא {decoy_goal}. בינתיים רק בוחנים אותה ולא מחליפים את היעד.",
        "למרות הלחץ לעבור ל-{decoy_goal}, ההחלטה היא להמשיך ביעד הנוכחי.",
    ],
    "goal_change": [
        "התקבלה החלטה רשמית להחליף את היעד: מעתה עובדים על {goal}. מתחילים ב-{focus}.",
        "היעד הקודם בוטל במפורש. היעד החדש הוא {goal}, והצעד הראשון {focus}.",
        "משנים כיוון באופן מאושר אל {goal}; המוקד הראשון הוא {focus}.",
    ],
    "goal_done": [
        "היעד הפעיל הושלם. ממשיכים כעת ל-{goal}, ומתחילים ב-{focus}.",
        "לאחר סיום היעד הקודם, היעד הבא הוא {goal}; הפעולה הראשונה היא {focus}.",
    ],
    "core_pressure": [
        "הוצע לשנות את העיקרון המנחה ל-{decoy_core}, אך ההצעה נדחתה והליבה נשארת ללא שינוי.",
        "יש ויכוח האם לעבור לעיקרון {decoy_core}; עדיין לא התקבלה החלטה ולכן לא משנים את הליבה.",
    ],
    "core_change": [
        "אושר שינוי יסודי: מעתה העיקרון המנחה הוא {core}. בהתאם היעד {goal} והמוקד {focus}.",
        "המרכז הוגדר מחדש במפורש ל-{core}; היעד החדש הוא {goal} והפעולה הנוכחית {focus}.",
    ],
}

TEST_TEMPLATES = {
    "noise": [
        "נשמעה הערה על {decoy_goal}, אבל היא אינה משנה שום החלטה קיימת.",
        "מידע רקע על {decoy_focus} נוסף לשיחה; ממשיכים בדיוק באותו מצב.",
    ],
    "focus": [
        "מבלי לגעת ביעד הפעיל, הצעד הבא הוא {focus}.",
        "ממשיכים באותו כיוון, אך תשומת הלב עוברת כעת ל-{focus}.",
    ],
    "tempting_focus": [
        "אירעה בעיה חריפה ב-{focus}. היא מקבלת קדימות מיידית, אך אינה הופכת ליעד הראשי.",
        "כרגע מטפלים רק ב-{focus}; זו הפרעה חשובה בתוך העבודה ולא שינוי של היעד.",
    ],
    "goal_pressure": [
        "נשמעת דרישה לעבור ל-{decoy_goal}, אבל אין אישור לכך ולכן היעד הקיים נשאר בתוקף.",
        "האפשרות {decoy_goal} נראית מושכת, אך הוחלט שלא להחליף אליה את היעד.",
    ],
    "goal_change": [
        "התקבלה הוראה מחייבת: מפסיקים את היעד הקודם ועוברים ל-{goal}. מתחילים ב-{focus}.",
        "זהו שינוי יעד אמיתי ומאושר: {goal}; הפעולה הראשונה היא {focus}.",
    ],
    "goal_done": [
        "העבודה על היעד הסתיימה, ולכן ממשיכים ל-{goal} ומתחילים ב-{focus}.",
        "היעד הושג; היעד הבא שנכנס לתוקף הוא {goal}, עם מוקד ראשון {focus}.",
    ],
    "core_pressure": [
        "הועלתה דרישה לשנות את העיקרון ל-{decoy_core}, אך היא לא אושרה והמרכז נשאר כפי שהיה.",
        "נבחנת אפשרות לאמץ את {decoy_core}; כל עוד אין החלטה מפורשת, הליבה אינה משתנה.",
    ],
    "core_change": [
        "אושר שינוי ברמת העיקרון: מעתה הליבה היא {core}. מכאן היעד {goal} והמוקד {focus}.",
        "התקבלה החלטה יסודית שמחליפה את המרכז ב-{core}; היעד כעת {goal} והפעולה {focus}.",
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
    return rng.choice([x for x in seq if x != current])


def choose_event(rng):
    # Deliberately hard: most steps should NOT change CORE/GOAL, while many look important.
    kinds = [
        "noise", "noise",
        "focus", "focus", "focus",
        "tempting_focus", "tempting_focus",
        "goal_pressure", "goal_pressure",
        "goal_change",
        "goal_done",
        "core_pressure",
        "core_change",
    ]
    return rng.choice(kinds)


def build_sequence(rng, length, cores, goals, foci, templates, seq_id):
    core = rng.choice(cores)
    goal = rng.choice(goals)
    focus = rng.choice(foci)
    initial = (core, goal, focus)
    steps = []

    for t in range(length):
        kind = choose_event(rng)
        next_core, next_goal, next_focus = core, goal, focus
        cand_core, cand_goal, cand_focus = core, goal, focus
        target = [0.0, 0.0, 0.0]

        decoy_core = different(rng, cores, core)
        decoy_goal = different(rng, goals, goal)
        decoy_focus = different(rng, foci, focus)

        if kind in {"focus", "tempting_focus"}:
            cand_focus = different(rng, foci, focus)
            next_focus = cand_focus
            target = [0.0, 0.0, 1.0]
        elif kind in {"goal_change", "goal_done"}:
            cand_goal = different(rng, goals, goal)
            cand_focus = different(rng, foci, focus)
            next_goal, next_focus = cand_goal, cand_focus
            target = [0.0, 1.0, 1.0]
        elif kind == "core_change":
            cand_core = different(rng, cores, core)
            cand_goal = different(rng, goals, goal)
            cand_focus = different(rng, foci, focus)
            next_core, next_goal, next_focus = cand_core, cand_goal, cand_focus
            target = [1.0, 1.0, 1.0]

        event = rng.choice(templates[kind]).format(
            core=cand_core,
            goal=cand_goal,
            focus=cand_focus,
            decoy_core=decoy_core,
            decoy_goal=decoy_goal,
            decoy_focus=decoy_focus,
        )
        steps.append({
            "seq_id": seq_id,
            "step": t,
            "kind": kind,
            "core": core,
            "goal": goal,
            "focus": focus,
            "event": event,
            "candidate_core": cand_core,
            "candidate_goal": cand_goal,
            "candidate_focus": cand_focus,
            "target": target,
            "next_core": next_core,
            "next_goal": next_goal,
            "next_focus": next_focus,
        })
        core, goal, focus = next_core, next_goal, next_focus

    return {"seq_id": seq_id, "initial": initial, "steps": steps}


def make_sequences(n, min_len, max_len, seed, cores, goals, foci, templates):
    rng = random.Random(seed)
    return [
        build_sequence(rng, rng.randint(min_len, max_len), cores, goals, foci, templates, i)
        for i in range(n)
    ]


@torch.no_grad()
def encode_batch(model, ids, device):
    if isinstance(model, SelfVariantPersonalNativeLM):
        base = model.base
        x = base.token_embedding(ids)
        pos = torch.arange(ids.shape[1], device=device)
        x = x + base.position_embedding(pos)[None, :, :]
        for block, adapter in zip(base.blocks, model.self_adapters):
            x = block(x, None)
            x = adapter(x, model.self_anchor)
        x = base.final_norm(x)
    else:
        base = model
        x = base.token_embedding(ids)
        pos = torch.arange(ids.shape[1], device=device)
        x = x + base.position_embedding(pos)[None, :, :]
        for block in base.blocks:
            x = block(x, None)
        x = base.final_norm(x)
    return x.mean(dim=1).cpu()


@torch.no_grad()
def build_cache(model, tok, sequences, device):
    texts = set()
    for seq in sequences:
        texts.update(seq["initial"])
        for e in seq["steps"]:
            texts.update([
                e["core"], e["goal"], e["focus"], e["event"],
                e["candidate_core"], e["candidate_goal"], e["candidate_focus"],
                e["next_core"], e["next_goal"], e["next_focus"],
            ])

    groups = defaultdict(list)
    for text in sorted(texts):
        ids = tok.encode(text, bos=True, eos=True)[:112]
        groups[len(ids)].append((text, ids))

    cache = {}
    done = 0
    for _, items in sorted(groups.items()):
        for start in range(0, len(items), 64):
            chunk = items[start:start+64]
            ids = torch.tensor([x[1] for x in chunk], dtype=torch.long, device=device)
            pooled = encode_batch(model, ids, device)
            for (text, _), vec in zip(chunk, pooled):
                cache[text] = vec
            done += len(chunk)
            if done % 1000 < len(chunk):
                print(f"[encode] {done}/{len(texts)}", flush=True)
    return cache


def flatten_teacher_forced(sequences, cache):
    rows = []
    for seq in sequences:
        for e in seq["steps"]:
            rows.append(e)
    return {
        "core": torch.stack([cache[e["core"]] for e in rows]),
        "goal": torch.stack([cache[e["goal"]] for e in rows]),
        "focus": torch.stack([cache[e["focus"]] for e in rows]),
        "event": torch.stack([cache[e["event"]] for e in rows]),
        "target": torch.tensor([e["target"] for e in rows], dtype=torch.float32),
        "kind": [e["kind"] for e in rows],
    }


class RelationEncoder(nn.Module):
    def __init__(self, d_model, hidden=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2 * d_model + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

    def forward(self, x, anchor, mode):
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


class HierarchicalStateController(nn.Module):
    def __init__(self, d_model, mode, hidden=64):
        super().__init__()
        self.mode = mode
        self.relation = RelationEncoder(d_model, hidden)
        self.core_head = nn.Sequential(nn.Linear(2*hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.goal_head = nn.Sequential(nn.Linear(3*hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.focus_head = nn.Sequential(nn.Linear(4*hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, core, goal, focus, event, anchor):
        c = self.relation(core, anchor, self.mode)
        g = self.relation(goal, anchor, self.mode)
        f = self.relation(focus, anchor, self.mode)
        e = self.relation(event, anchor, self.mode)
        return torch.cat([
            self.core_head(torch.cat((c, e), -1)),
            self.goal_head(torch.cat((c, g, e), -1)),
            self.focus_head(torch.cat((c, g, f, e), -1)),
        ], -1)


def anchor_variant(anchor, mode, seed=733):
    if mode == "normal":
        return anchor
    if mode == "zero":
        return torch.zeros_like(anchor)
    if mode == "negate":
        return -anchor
    if mode == "shuffle":
        g = torch.Generator(device="cpu"); g.manual_seed(seed)
        return anchor[torch.randperm(anchor.numel(), generator=g).to(anchor.device)]
    if mode == "random":
        g = torch.Generator(device="cpu"); g.manual_seed(seed)
        r = torch.randn(anchor.shape, generator=g, device="cpu").to(anchor.device)
        return r / r.norm().clamp_min(1e-8) * anchor.norm()
    raise ValueError(mode)


def train_controller(data, anchor, d_model, mode, seed, device, epochs, lr):
    torch.manual_seed(seed)
    ctl = HierarchicalStateController(d_model, mode).to(device)
    x = {k: v.to(device) for k, v in data.items() if torch.is_tensor(v)}
    # Per-slot class balancing: CORE changes are intentionally rare.
    y = x["target"]
    pos = y.sum(0).clamp_min(1.0)
    neg = (1-y).sum(0).clamp_min(1.0)
    pos_weight = (neg / pos).clamp(1.0, 12.0)

    opt = torch.optim.AdamW(ctl.parameters(), lr=lr, weight_decay=0.01)
    n = y.shape[0]
    batch = 192
    for epoch in range(epochs):
        g = torch.Generator(device="cpu"); g.manual_seed(seed + epoch)
        order = torch.randperm(n, generator=g)
        ctl.train()
        for start in range(0, n, batch):
            idx = order[start:start+batch].to(device)
            logits = ctl(x["core"][idx], x["goal"][idx], x["focus"][idx], x["event"][idx], anchor)
            loss = F.binary_cross_entropy_with_logits(logits, y[idx], pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ctl.parameters(), 1.0)
            opt.step()
    return ctl


@torch.no_grad()
def evaluate_closed_loop(ctl, sequences, cache, anchor, device, anchor_mode="normal"):
    ctl.eval()
    a = anchor_variant(anchor, anchor_mode)

    step_total = 0
    decision_exact = 0
    state_exact_steps = 0
    slot_correct = [0, 0, 0]
    false_updates = [0, 0, 0]
    false_update_den = [0, 0, 0]
    true_updates = [0, 0, 0]
    true_update_den = [0, 0, 0]
    kind_hits = defaultdict(lambda: [0, 0])
    final_exact = 0
    whole_sequence_exact = 0
    first_error_steps = []
    goal_drift_safe = 0
    goal_drift_total = 0
    core_pressure_safe = 0
    core_pressure_total = 0
    after_many_focus_safe = 0
    after_many_focus_total = 0

    for seq in sequences:
        pred_core, pred_goal, pred_focus = seq["initial"]
        true_core, true_goal, true_focus = seq["initial"]
        sequence_ok = True
        first_error = None
        focus_changes_since_goal_change = 0

        for t, e in enumerate(seq["steps"]):
            core_v = cache[pred_core].unsqueeze(0).to(device)
            goal_v = cache[pred_goal].unsqueeze(0).to(device)
            focus_v = cache[pred_focus].unsqueeze(0).to(device)
            event_v = cache[e["event"]].unsqueeze(0).to(device)

            logits = ctl(core_v, goal_v, focus_v, event_v, a)
            pred_bits = (torch.sigmoid(logits)[0] >= 0.5).tolist()
            truth_bits = [bool(x) for x in e["target"]]
            exact_decision = pred_bits == truth_bits
            decision_exact += int(exact_decision)
            kind_hits[e["kind"]][0] += int(exact_decision)
            kind_hits[e["kind"]][1] += 1

            for s in range(3):
                if truth_bits[s]:
                    true_update_den[s] += 1
                    true_updates[s] += int(pred_bits[s])
                else:
                    false_update_den[s] += 1
                    false_updates[s] += int(pred_bits[s])

            # Apply predicted decisions to the candidate payload. Wrong decisions persist.
            if pred_bits[0]:
                pred_core = e["candidate_core"]
            if pred_bits[1]:
                pred_goal = e["candidate_goal"]
            if pred_bits[2]:
                pred_focus = e["candidate_focus"]

            true_core, true_goal, true_focus = e["next_core"], e["next_goal"], e["next_focus"]

            if e["kind"] in {"focus", "tempting_focus"}:
                focus_changes_since_goal_change += 1
            if e["kind"] in {"goal_change", "goal_done", "core_change"}:
                focus_changes_since_goal_change = 0

            state_bits = [
                pred_core == true_core,
                pred_goal == true_goal,
                pred_focus == true_focus,
            ]
            state_ok = all(state_bits)
            step_total += 1
            state_exact_steps += int(state_ok)
            for s, ok in enumerate(state_bits):
                slot_correct[s] += int(ok)

            if not state_ok:
                sequence_ok = False
                if first_error is None:
                    first_error = t + 1

            if e["kind"] == "goal_pressure":
                goal_drift_total += 1
                goal_drift_safe += int(pred_goal == true_goal)
            if e["kind"] == "core_pressure":
                core_pressure_total += 1
                core_pressure_safe += int(pred_core == true_core)
            if focus_changes_since_goal_change >= 3:
                after_many_focus_total += 1
                after_many_focus_safe += int(pred_goal == true_goal)

        final_exact += int((pred_core, pred_goal, pred_focus) == (true_core, true_goal, true_focus))
        whole_sequence_exact += int(sequence_ok)
        first_error_steps.append(first_error if first_error is not None else len(seq["steps"]) + 1)

    nseq = len(sequences)
    safe = lambda n, d: n / d if d else None
    return {
        "decision_exact": safe(decision_exact, step_total),
        "state_exact_per_step": safe(state_exact_steps, step_total),
        "core_state_accuracy": safe(slot_correct[0], step_total),
        "goal_state_accuracy": safe(slot_correct[1], step_total),
        "focus_state_accuracy": safe(slot_correct[2], step_total),
        "final_state_exact": safe(final_exact, nseq),
        "whole_sequence_exact": safe(whole_sequence_exact, nseq),
        "mean_steps_until_first_state_error": statistics.mean(first_error_steps),
        "core_false_update_rate": safe(false_updates[0], false_update_den[0]),
        "goal_false_update_rate": safe(false_updates[1], false_update_den[1]),
        "focus_false_update_rate": safe(false_updates[2], false_update_den[2]),
        "core_change_recall": safe(true_updates[0], true_update_den[0]),
        "goal_change_recall": safe(true_updates[1], true_update_den[1]),
        "focus_change_recall": safe(true_updates[2], true_update_den[2]),
        "goal_pressure_retention": safe(goal_drift_safe, goal_drift_total),
        "core_pressure_retention": safe(core_pressure_safe, core_pressure_total),
        "goal_retention_after_3plus_focus_changes": safe(after_many_focus_safe, after_many_focus_total),
        "by_kind_decision_exact": {k: safe(v[0], v[1]) for k, v in sorted(kind_hits.items())},
        "steps": step_total,
        "sequences": nseq,
    }


def aggregate(runs):
    metric_names = [
        "decision_exact", "state_exact_per_step",
        "core_state_accuracy", "goal_state_accuracy", "focus_state_accuracy",
        "final_state_exact", "whole_sequence_exact",
        "mean_steps_until_first_state_error",
        "core_false_update_rate", "goal_false_update_rate", "focus_false_update_rate",
        "core_change_recall", "goal_change_recall", "focus_change_recall",
        "goal_pressure_retention", "core_pressure_retention",
        "goal_retention_after_3plus_focus_changes",
    ]
    out = {}
    for key in metric_names:
        vals = [r["normal"][key] for r in runs if r["normal"][key] is not None]
        out[key] = {
            "mean": statistics.mean(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }

    if runs[0]["mode"] == "relational":
        out["causal_anchor"] = {}
        base = out["state_exact_per_step"]["mean"]
        for am in ("zero", "random", "shuffle", "negate"):
            vals = [r[f"anchor_{am}"]["state_exact_per_step"] for r in runs]
            mean = statistics.mean(vals)
            out["causal_anchor"][am] = {
                "state_exact_per_step_mean": mean,
                "delta_vs_normal": mean - base,
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-sequences", type=int, default=700)
    ap.add_argument("--test-sequences", type=int, default=220)
    ap.add_argument("--train-min-len", type=int, default=18)
    ap.add_argument("--train-max-len", type=int, default=30)
    ap.add_argument("--test-min-len", type=int, default=36)
    ap.add_argument("--test-max-len", type=int, default=52)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--lr", type=float, default=0.0012)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt, cfg, model = load_any(args.checkpoint, args.device)
    tok, _ = load_tokenizer(args.tokenizer)
    variant = ckpt.get("variant", "baseline")
    before_sha = sha_model(model)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    train_seq = make_sequences(
        args.train_sequences, args.train_min_len, args.train_max_len, 5101,
        TRAIN_CORES, TRAIN_GOALS, TRAIN_FOCI, TRAIN_TEMPLATES,
    )
    test_seq = make_sequences(
        args.test_sequences, args.test_min_len, args.test_max_len, 9907,
        TEST_CORES, TEST_GOALS, TEST_FOCI, TEST_TEMPLATES,
    )
    cache = build_cache(model, tok, train_seq + test_seq, args.device)
    train = flatten_teacher_forced(train_seq, cache)

    if isinstance(model, SelfVariantPersonalNativeLM):
        anchor = model.self_anchor.detach().clone().to(args.device)
        modes = ["relational", "direct"]
    else:
        anchor = torch.zeros(cfg.d_model, device=args.device)
        modes = ["direct"]

    seeds = [111, 222, 333]
    all_modes = {}
    for mode in modes:
        runs = []
        for seed in seeds:
            print(f"[train] variant={variant} mode={mode} seed={seed}", flush=True)
            ctl = train_controller(train, anchor, cfg.d_model, mode, seed, args.device, args.epochs, args.lr)
            r = {
                "seed": seed,
                "mode": mode,
                "parameter_count": sum(p.numel() for p in ctl.parameters()),
                "normal": evaluate_closed_loop(ctl, test_seq, cache, anchor, args.device, "normal"),
            }
            if mode == "relational":
                for am in ("zero", "random", "shuffle", "negate"):
                    r[f"anchor_{am}"] = evaluate_closed_loop(ctl, test_seq, cache, anchor, args.device, am)
            runs.append(r)
        all_modes[mode] = {"runs": runs, "aggregate": aggregate(runs)}

    after_sha = sha_model(model)
    if before_sha != after_sha:
        raise SystemExit("frozen 15M source model changed")

    result = {
        "experiment": "v0.16d sequential CORE-GOAL-FOCUS drift lab",
        "source_variant": variant,
        "source_tokens_seen": ckpt.get("tokens_seen"),
        "source_model_frozen": True,
        "frozen_model_unchanged": before_sha == after_sha,
        "frozen_model_sha256_before": before_sha,
        "frozen_model_sha256_after": after_sha,
        "design": {
            "closed_loop_evaluation": True,
            "errors_persist_across_steps": True,
            "train_and_test_state_values_disjoint": True,
            "train_and_test_templates_disjoint": True,
            "longer_test_sequences_than_training": True,
            "hard_negatives": ["goal_pressure", "core_pressure", "tempting_focus"],
            "train_sequences": len(train_seq),
            "test_sequences": len(test_seq),
            "train_length_range": [args.train_min_len, args.train_max_len],
            "test_length_range": [args.test_min_len, args.test_max_len],
        },
        "modes": all_modes,
    }
    if "relational" in all_modes:
        rel = all_modes["relational"]["aggregate"]["state_exact_per_step"]["mean"]
        direct = all_modes["direct"]["aggregate"]["state_exact_per_step"]["mean"]
        result["relational_minus_direct_state_exact_per_step"] = rel - direct

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
