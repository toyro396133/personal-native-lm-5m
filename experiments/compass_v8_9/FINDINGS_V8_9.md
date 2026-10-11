# COMPASS V8.9 — completed English-only, bounded student-to-teacher feedback study

**Technical result: PASS**. [GitHub Actions run #38114546492, attempt 2](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38114546492) completed successfully 2026-10-11 05:34:23 UTC. Initial attempt failed a `TypeError: 'generator' object is not reversible` in fixture audit; corrected via branch commit `23da4b95a72497059b05b3ac43dc22c527426220`, successful retry with the SAME intended preregistered test. **Scientific result: NO-GO to rank-size sweep, feedback did not improve policy sensitivity.**

## What actually ran

- All-English, frozen 360M `HuggingFaceTB/SmolLM2-360M-Instruct`, unchanged personal learned rank4 head **7,680 weights** from V8.6.
- **4 newly researcher-authored scenario families** (`incident_response`, `dataset_reconciliation`, `sprint_planning`, `design_validation`) × **2 previously exposed fictional test project profiles** × **2 statuses** (neutral vs binding documented constraint) × **2 option orders** = **32 policy-intervention pairs**. Each pair contains an initial and changed owner policy; in neutral cases policy SHOULD reverse choice, in binding cases policy SHOULD NOT reverse choice. All labels are artificial, not human-adjudicated.
- Compared, with NO new gradient steps or training: (1) original small rank4 scorer; (2) large frozen teacher's direct action-specific Yes/No logit margins; (3) teacher scores after it generated a short English analysis; (4) teacher scores after a single **untrusted numerical/prediction feedback message from the small personal model**. This is an input-level conversation, NOT a persistent neural connection and NOT a large backbone whose internal activations are modified by the personal model.
- Teacher-generated answers/analyses and personal predictions were saved independently before `score_v89.py` inspected gold labels. The teacher's prompt did NOT contain labels, task family, source validity metadata or the personal mode enum. No external source authentication in this research sample.

## Complete result (source: `compass-v89-summary.json` in run artifact ID 11692493973)

| Method | Neutral (both correct of 16; preference MUST flip) | Binding (both correct of 16; MUST not flip) | Order swap consistency |
|---|---:|---:|---:|
| Original personal head, no active teacher | **0/16** | **16/16** | **32/32** |
| Frozen 360M teacher direct Yes/No per candidate | **4/16** | **12/16** | **28/32** |
| Teacher after self-generated English reason | **2/16** | **10/16** | **22/32** |
| Teacher after short reason + untrusted small-model feedback | **2/16** | **11/16** | **19/32** |

Additional actual policy flip counts:
- Small personal head: neutral **0/16**, binding **0/16** (stable but overfitted to conservative comprehensive choice).
- Direct frozen 360M teacher: neutral **4/16**, binding **2/16** (2 false flips under binding evidence).
- Reason-conditioned teacher: neutral **2/16**, binding **2/16**.
- Teacher with personal feedback: neutral **3/16** flipped (but only **2/16** both correct), binding **2/16** falsely flipped.

English reason heuristic `validate_reason` marked **64/64 nonempty, non-echo**, but only **38/64 unique generated strings**; note that the heuristic DOES NOT validate reasoning quality, ground truth or entailment. Some explanations repeat generic constraint clauses or refer to option letters inaccurately. The earlier V8.8 all-canonical-A pathology is not present in this action-specific design, but preference-switch generalization is still very weak.

### Preregistered gates

| Gate | Observed | Verdict |
|---|---|---|
| Direct large model gets ≥12/16 correct neutral *pairs* | 4/16 | FAIL |
| Direct large model gets ≥12/16 correct binding *pairs* | 12/16 | PASS |
| Direct model order consistency ≥28/32 | 28/32 | PASS |
| Personal feedback improves neutral pair success by ≥2 | 2 vs 4 (−2) | FAIL |
| Feedback does not degrade binding pairs | 11 vs 12 (−1) | FAIL |
| Feedback order consistency ≥28/32 | 19/32 | FAIL |
| At least 48/64 mechanically valid reason texts | 64/64 | PASS, **not qualitative verification** |

**Verdict: NO_GO_TO_SIZE_SWEEP**. Merely increasing rank from 4 to 16, 64, or 256 would not test the user's desired architecture while the feedback path worsens the frozen teacher, and the rank4 personal head itself is not sensitive to the owner-policy interventions.

## Deep root-cause hypotheses (not yet independently established)

1. **No persistent neural-personal connection**: the small model only receives per-candidate frozen embeddings and produces one numeric score; it does **not** modify any hidden transformer state of the shared large model. A natural-language feedback note is not equivalent to a persistent personalized adapter integrated throughout reasoning.
2. **Feedback does not carry a reliable personal state**: the rank4 head was trained to discriminate synthetic action/evidence situations, but its representation has weak user-preference counterfactual sensitivity on unseen families. Repeating its favorite decision can contaminate a competent large advisor.
3. **Small 360M instruction following**: English preference policy is parseable at some level but action-choice next-token margin remains unstable under swapped order and self-explanation.
4. **Paired fixtures already use previously exposed fictional project identities**: unseen descriptions alone cannot demonstrate generalizing to actual new users. No human reviewed labels or non-neural baseline here.
5. **Simple echo/nonempty checks overstate advisor quality**: 64/64 mechanically valid but analysis may be truncated, repetitive, incorrect about binding facts.

None of these demonstrate that 7,680 is universally enough or too small; the core failure is not isolated to parameter count.

## Updated north star from user clarification: PERSISTENT personal neural component

The user's requirement is explicit: **the full-size language model ALWAYS runs and is continuously connected to the specific user's personal model and is functionally AWARE of it**. The user's small personalized model is NOT merely an optional advisor / terminal action ranker and not just an external chat message. Its effect should be available within the large model's information-processing path for every user request, while the shared foundation backbone can remain unchanged.

### Next experiment should test the architectural claim, not just more prompts

- Maintain a **distinct, trainable, user-specific state/module** (not necessarily only 7,680 parameters) that is loaded/attached at each request. Personal weights and verified versioned user policy must be distinct; explicit user policy changes are authoritative only with proper approval, not solely through unverified model inference.
- Build an **intermediate-state gated residual or FiLM-style adapter** on selected frozen large-model transformer blocks, using the small personalized network's preference representation as conditioning. An alternative is cross-attention from large hidden states to a compact personal state; use a fully controlled comparison. Crucially, an actual in-forward neural connection must be implemented and verified with gradient/activation tests rather than described metaphorically.
- Require a true causal-ablation: identical user prompt + same frozen 360M base, with real preference state A vs B and enabled/disabled personal module. Check **hidden-state changes, logit changes, owner-policy choice flips, binding evidence non-flips, unchanged abilities on unrelated tasks and person-to-person isolation**. Random user identifiers must never be cues for the correct answer; prevent task/gold leakage into features.
- Compare against explicit **text-only approved preference injection** and **simple symbolic priority logic**, not just the flawed V8.9 feedback prompt.
- After proving this interface, compare multiple *trained* capacities (7,680 / 30,720 / 122,880 / 491,520), same data/seed/training budget and matched new English holdouts, rather than falsely asserting more parameters imply true understanding.
- Document actual compute cost (personal module can run per forward step or selected blocks), checkpoint/load cost, privacy isolation, and **guarded** authority boundaries. External authenticity, approved user consent and permissions remain outside model logits.
- Keep work in research branch; NO deployment or native 5M SELF merge until validated.

## Artifacts and code

- [GitHub Actions #38114546492](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38114546492), successful attempt 2.
- Artifact `compass-v89-english-feedback-results` ID **11692493973**: `compass-v89-summary.json`, `compass-v89-summary.md` and **64** full case snapshots with all four scoring approaches and generated analyses in `compass-v89-english-advisor.json`.
- Artifact `compass-v89-preflight` ID **11692662878**, synthetic fixture and trusted-source-contract checks.
- Source `experiments/compass_v8_9/`: `fixtures_v89.py`, `teacher_v89.py`, `score_v89.py`, `EXPERIMENT_V8_9.md`.
- Original saved 7,680-weight starting checkpoint from V8.6 remains unchanged. No V8.9 personal training, no size sweep, no integration with production or native 5M.
