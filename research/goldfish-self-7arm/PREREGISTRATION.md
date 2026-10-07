# Goldfish 124.8M seven-arm SELF transfer — preregistration

Status: preregistered before the first seven-arm run.

Backbone: `goldfish-models/heb_hebr_1000mb` at revision
`425eee09d1b9ff32ffa21d6cfb300a74302a0799`.

## Purpose

This is a transfer/feasibility experiment for the SELF research. Language
ability is treated as pre-existing infrastructure, not as the research target.
The question is whether the existing seven-arm SELF mechanisms can co-adapt
inside a pretrained Hebrew causal LM while all pretrained weights remain
trainable.

The run uses the same stable blind-safe V1–V7 codes as the existing blind
study. This document deliberately does not publish the code-to-architecture
mapping.

## Fixed design

- seven arms in parallel;
- identical pretrained backbone revision in every arm;
- every backbone parameter remains trainable;
- identical FineWeb2 Hebrew token stream, validation split, seed and block
  order;
- sequence length 128 for this canary;
- 1,000,000 continuation tokens per arm;
- measurements at 0 / 250K / 500K / 1M;
- base-model learning rate: 1e-5;
- SELF adapter learning rate: 1e-4;
- slow variants retain the prior 0.1 anchor learning-rate multiplier;
- SELF rank 8 and projection rank 4, matching the existing seven-arm mechanism;
- final model + AdamW state is archived to private Hugging Face storage so a
  later run can resume without optimizer reset;
- GitHub artifacts contain metrics only, not multi-gigabyte checkpoints.

The higher SELF-adapter LR is intentional: the backbone is already pretrained
while the new SELF path begins at a zero-output adapter initialization. The
base LR is kept low to protect the pretrained Hebrew substrate.

## What is measured

Language loss is a health/retention metric. It is not the primary SELF score.

At each milestone, SELF arms record primitive intervention measurements for:

- zero;
- negate;
- shuffle;
- random same-norm replacement;
- orthogonal same-norm replacement.

For every intervention we separately record:

- replacement norm;
- displacement norm from the learned anchor;
- cosine to the learned anchor;
- validation NLL change on a fixed probe;
- final-hidden-state displacement;
- final-hidden-state cosine.

We also record the learned anchor norm and per-layer adapter diagnostics.

## Blind follow-up corrections incorporated

The independent blind follow-up at results commit
`b1276e3510480fa6037c3ef2a9c11adea0a6523d` changed the interpretation
rules for this transfer.

1. Intervention magnitude and intervention mode are separate explanatory
   variables. Raw negate damage must not be called sign sensitivity without
   magnitude control.
2. SELF effect strength and learned-value specificity are distinct dimensions.
   A strong intervention effect does not establish that the learned SELF value
   is a privileged reference.
3. Simple scalar gate/path norms are not assumed to causally explain
   next-layer SELF effects.
4. Development is allowed to be non-monotonic. A 250K/500K/1M trajectory is
   not forced into a single-stage maturation story.
5. Conditional or geometric differences involving CORE/GOAL/FOCUS are not
   called causal without direct component intervention.
6. Composite metrics are secondary. Primitive measurements remain the
   evidential base.

The blind follow-up also found a robust D-vs-C challenge asymmetry and
multi-axis variant behavior. Those observations motivate later structural
assays, but this 1M transfer canary does not pretend to provide direct B/C/D
causal evidence.

## Success criteria for this canary

A successful infrastructure/transfer result requires:

- all seven arms complete full-parameter training;
- no arm catastrophically destroys the pretrained language substrate;
- metrics remain numerically stable;
- SELF arms develop measurable, variant-dependent intervention effects;
- final checkpoints and optimizer state are resumably archived.

This canary alone cannot establish semantic SELF, universal learned-reference
behavior, or a causal SELF/CORE/GOAL/FOCUS topology.

## Next gate

If the transfer succeeds, the next experiment should use the archived 1M
states for direct structural/causal assays, including magnitude-matched
interventions and an opaque V1–V7 runner compatible with the blind-runner
contract. Multi-seed replication comes before any strong architecture-level
claim.
