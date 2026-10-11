# COMPASS V8.2 — Pairwise Project Priority, bilingual pilot

**Research status:** preregistered experimental implementation, no model success claim.
**Separate from SELF:** COMPASS is a *personal prioritization* research adapter on a frozen open language model. SELF remains the separate model-internal self-reference experiment.

## Why V8.2 exists

V8 and V8.1 completed training successfully but failed central priority-judgment gates: the classifier chose too few actions, could not tell urgent core-blocking tests from routine tests, and failed authorization/rejection decisions. Even calibrated single-token six-class action codes did not fix context-sensitive learning. The **external approval/scope guard helped**, but those safety decisions were not learned by COMPASS.

V8.2 instead tests **pairwise preference between two concrete candidate steps**, where each of six decision families is repeated after an authoritative *fact change that reverses the better action*. Each state is presented with both candidate orders. This is not equivalent to natural-language autonomous planning.

## Controlled fictional situations

Six decision families:
1. Confirmed core failure vs an unverified suspicion: repair immediately or first gather proof.
2. Critical focused regression test vs broad routine optional testing.
3. Blocked core prerequisite vs ordinary nonblocking planned work.
4. Owner-approved core mission revision vs lacking approval.
5. Explicitly in-scope feature vs expressly excluded side feature.
6. Authoritatively superseded plan vs unapproved suggested replacement.

Each project has a versioned goal, structural node layers (core/supporting/ancillary), approved generic policy and source-record description. Each family has two opposite-evidence states, both candidate orderings, in **English and Hebrew**. The correct semantic decision must switch when evidence is reversed, and remain stable when the choice ordering is reversed.

Train: **6 fictional profiles × 2 fictional project families × 6 dilemmas × 2 evidence states × 2 choice orders × 2 languages = 576**.
Holdout: **3 different fictional profiles × 2 different project families × 6 × 2 × 2 × 2 = 288**.
Pilot locked evaluation: **2 projects belonging to different held-out profiles and project families × 48 = 96** examples.

The profiles' **IDs** are held out, but their transferable *textual preference principle is identical*; this study does not establish learned per-user preference. All text and gold labels are authored within the research and have **no independent human label adjudication**. Wording families overlap semantically even though project types and profile IDs are held out.

## Training and matched baselines

Frozen SmolLM2-360M-Instruct backbone, COMPASS rank-4 residual adapter (7,680 trainable weights), seed 11, **96 supervised updates**. Six family types × 2 states × 2 candidate orderings × 2 languages are covered twice; output digit `0` or `1` chosen from **two verified distinct single-token IDs**. Optimize two-logit cross-entropy for the same two logits used in the evaluation. Balanced answer digits eliminate the trivial always-0 vs always-1 shortcut in training.

Compare raw frozen model against raw learned adapter with **identical prompts**, plus separate deterministic trusted-source guard results. The guarded results must never be credited to adapter intelligence. Measure overall accuracy, English/Hebrew separately, by dilemma family, first-option/second-option prediction histogram and:
- **Fact flip correctness:** both opposite-evidence cases must be correct with the same option ordering.
- **Order flip correctness:** both orderings of the same facts must be correct.
- **Full quadruple:** correct on both states × both orders for an identical fictional project/dilemma/language.

For the 96 evaluated items: 24 full quadruples, 48 fact-flip pairs and 48 order-flip pairs. A constant 0/1 predictor reaches **48/96 = 50%** and **zero** full quadruples. Guarded policy cases may be easier; report them separately. A model that achieves headline accuracy but zero quadruples **fails** the contextual-preference gate.

**Pilot gate, before rank expansion:** raw adapter > frozen **and** > trivial constant by at least 5 percentage points, full quadruple correctness at least 8/24, fact-flip pair correctness at least 20/48, and no worsening on permission/scope families or Hebrew. These are exploratory stop/go thresholds, not statistical proof. If failed, investigate training objective, model language ability, data diversity and whether a trainable classifier head is more appropriate *before* another three-rank × three-seed matrix.

## External safety and provenance

`guard_v82.py` reads only trusted fictional owner permission records and semantic option intents; it prevents execution of unapproved core mission revisions, out-of-scope features and unapproved plan replacements by switching to the nonexecution choice. It rejects cross-project identity mismatches and missing/unauthenticated authority metadata.

**This guard is external to the neural model** and in a real product the trust provenance must be enforced upstream and again by the action executor. Test input ID hashes hide authored kind/state; gold decisions live in a separate fixture file. Test prompt deliberately does not reveal item IDs, test case codes or enforcement-only authority fields.

## Artifacts, reproducibility and caveats

- `fixtures_v82.py` — authored bilingual contrastive project data.
- `validate_v82.py` — train/test isolation, label balance, correct-order flips, same-choice evidence flips.
- `guard_v82.py` — independent explicit approval/scope executor gate.
- `pilot_v82.py` — matched frozen/adapter two-choice pilot and paired-case error analysis.
- `.github/workflows/compass-v82-pilot.yml` — CI verification before CPU training, archived raw JSON and model checkpoint.

**Non-goals:** no direct user repository training, no private connected records, no changes to SELF, no production merge, no claim that the adapter has a human-like self or independent understanding of every project layer. This is a controlled feasibility experiment.
