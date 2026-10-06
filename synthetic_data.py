"""
Synthetic personalization curriculum.

Purpose:
- prove that the same prompt can yield different targets based on PersonalState;
- teach the reader that irrelevant/low-confidence state must not dominate;
- keep this dataset separate from real language pretraining.

This is NOT intended to make the 5M model generally intelligent.
"""
import random
import torch
from personal_state import PersonalState, StateSlot, CORE_DIM, POLICY_DIM, WORLD_DIM, ROUTING_DIM

def _vec(dim, index, value=1.0):
    v = [0.0] * dim
    v[index % dim] = value
    return v

def make_state(user_id, style_index, world=None, confidence=1.0):
    s = PersonalState(
        user_id=user_id,
        core=StateSlot(_vec(CORE_DIM, style_index), confidence=confidence, evidence_count=5),
        policies=StateSlot(_vec(POLICY_DIM, style_index + 7), confidence=confidence, evidence_count=5),
        routing=StateSlot(_vec(ROUTING_DIM, style_index + 3), confidence=confidence, evidence_count=5),
    )
    if world:
        s.worlds[world] = StateSlot(
            _vec(WORLD_DIM, style_index + 11),
            confidence=confidence,
            evidence_count=5,
        )
    return s

# Same question; correct continuation is state-dependent.
PERSONAL_PAIRS = [
    ("ענה בקיצור: בחר א או ב. תשובה: ", "א", "ב"),
    ("Choose A or B. Answer: ", "A", "B"),
    ("סגנון מועדף: קצר או מפורט? ", "קצר", "מפורט"),
]

def sample_example(tokenizer, max_len=96):
    prompt, ans0, ans1 = random.choice(PERSONAL_PAIRS)
    style = random.randint(0, 1)
    state = make_state(f"u{style}", style, world="demo", confidence=1.0)
    answer = ans0 if style == 0 else ans1

    ids = tokenizer.encode(prompt + answer)
    ids = ids[:max_len]
    x = ids[:-1]
    y = ids[1:]

    # pad is handled in collate
    return x, y, state.flatten("demo")

def collate(batch, pad_id):
    max_t = max(len(x) for x, _, _ in batch)
    xs, ys, ss = [], [], []
    for x, y, s in batch:
        pad = max_t - len(x)
        xs.append(x + [pad_id] * pad)
        ys.append(y + [-100] * pad)
        ss.append(s)
    return (
        torch.tensor(xs, dtype=torch.long),
        torch.tensor(ys, dtype=torch.long),
        torch.stack(ss),
    )
