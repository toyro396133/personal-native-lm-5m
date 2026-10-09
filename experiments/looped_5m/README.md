# Looped Transformer 5M — isolated canary

Status: **implementation prepared; no training or benchmark result claimed**.

## Hypothesis

Repeating the existing six physical blocks at the same parameter count may
increase language-model quality or SELF behavior. This costs additional compute
and can be unstable. This experiment does not replace the existing SELF runs.

## Arms

| Arm | Passes | Injection |
| --- | ---: | ---: |
| baseline | 1 | 0 |
| loop2 | 2 | 0 |
| loop4 | 4 | 0 |
| loop8 | 8 | 0 |
| loop4-inject | 4 | 0.1 |

Use exactly the same tokenizer, training and held-out text splits, data order,
seed, optimizer, physical-block configuration and exposure (tokens) for all arms.
Report parameter count, train/eval compute, peak RAM and wall time.
Compare against an **equal-compute** control as well as equal tokens, since
extra loop passes multiply FLOPs. Train each arm **from scratch** for the first
comparison; using a pretrained baseline as the initial checkpoint would be a
different research question. Do not compare perplexity across tokenizers.

## Readiness

Run `python -m pytest -q tests/test_looped_model.py`.
An input/output/gradient smoke test is not a training result.

Before promoting: implement a dedicated training runner with checkpoints,
held-out nats/character, eval after fixed token milestones, repetition and
generation samples, seeds >= 3 for close results. Test SELF anchor ablation
and matched-capacity controls; a language gain alone is not proof of SELF.

## Implementation

`looped_model.LoopedPersonalNativeLM(cfg, loops=4, input_injection=0.0)`
has the same parameter names as `PersonalNativeLM` and intentionally shares
all physical block weights across passes. `loops=1` must match the existing
model exactly, including conditional personal-state routing.
