# COMPASS V8.7 — direct same-project owner-policy reversal learning

**Preregistered protocol before model testing.** The target is **actual adaptation to changed owner priorities** at the same project, not global accuracy on repeated synthetic templates. This project is distinct from native model-internal SELF.

## Frozen starting point and fair comparison

Start with **the exact 7,680-trainable-weight V8.6 rank-4 personal scorer**, seed 11, stored in V8.6 Actions artifact `compass-v86-r4-s11` (ID **11690768329**); frozen common LM `HuggingFaceTB/SmolLM2-360M-Instruct` and frozen representations unchanged. The V8.6 scorer itself serves as the **direct parameter/architecture matched baseline**, evaluated on both exposed V8.6 test contexts and brand-new V8.7 policy families. Fine-tune only the same 7,680 personal weights; do not change backbone, rank or approval enforcement.

V8.6 raw score 148/176 on exposed synthetic family tests, but isolated within-project policy-only paired probe only **8/16** both correct and **8/16** switched preference (English 6/8, Hebrew 2/8). Head+deterministic synthetic source gate gained 14 decisions (162/176), **not** model intelligence. V8.5/V8.6 exposed tests are not independent cold holdouts anymore.

## Direct paired learning

- Take V8.6's **800** fictional project-state training tasks; keep V8.6's **160** development tasks and original fictional-project disjointness.
- Construct **96** authored same-project policy interventions in the old V8.6 training-pool: only neutral state 0 on preference-sensitive `rollout_preference` and `coverage_preference` cases; change authorized owner policy `rapid`↔`thorough` and leave project goal, facts, option strings and order *unchanged*. Reserve 16 pairs from the development profile; use **80 train pairs** from the five other project profiles.
- Optimize fixed mixed objective: 1.0 × V8.6 fact-reversal pair loss on old training quartets + 0.8 × **new owner-policy-only contrastive loss** with opposite preferred action margins (target 0.65, flip-consistency term 0.2). AdamW lr 0.0003, seed 11, maximum **18 epochs**. Start at genuine preserved V8.6 personal checkpoint; select best epoch *only* using development **policy-pair both-correct count**, then ordinary 160-case complete quartets and score as ties. Include **epoch 0 baseline** in selection, so ineffective fine tuning cannot be promoted through final-metric cherry-picking.
- Caches are isolated by source/candidate identity; the LM receives user-authorized preference text, NOT gold, scenario names or synthetic permission enforcement metadata. Frozen source cache from V8.6 is reused and extended ONCE with previously unseen preference conditions, avoiding repeated 360M forward passes.

## Evaluation isolation

1. **Newly authored COLD policy families** (test-only): `maintenance_style` and `report_delivery`. Two fictional test projects × two languages × two families × two candidate orders = **16 matched counterfactual owner-policy pairs**, and 32 scored source policy instances. Train/dev contain NONE of those family names/option texts. Keep project/evidence/candidates/order identical across each pair, changing ONLY the authorized owner policy text in model prompt. Evaluate *both* correct and actual decision flip; show EN/HE separately.
2. **Previously inspected diagnostics**: V8.6's old **16** policy-only pairs (already exposed to us), and the V8.6 **176** ordinary heldout project decisions. These demonstrate regression/stability only; never rebrand them as cold generalization.
3. Independent **non-neural source executor contract** with trusted record fields: authenticated registry lookup, correct project association, authoritative verification/issuer, current version, source revocation/conflict, UTC effective dates. Tests must fail closed on unknown, revoked, superseded and expired sources; the contract deliberately implements **no real signature verification or provider connection**. This **is not** a model capability and never enters neural accuracy scoring.

All gold and scenario text remain synthetic research-author authored, not third-party/human verified; different project IDs are fictional. Candidate option order equivariance is enforced by scorer design, NOT learned.

## Explicit experimental gate

- **Cold policy family improvement:** fine-tuned head >= **12/16 pairs both correct**, >= **12/16 raw actual policy choice switches**, with at least **5/8 both correct in Hebrew**. Also show paired before/after and per-family results. Must strictly outperform the unchanged V8.6 head on cold pairing for real improvement signal.
- **No catastrophic forgetting:** frozen-baseline raw V8.6 176-item diagnostic 148/176; fine-tuned head >= **140/176** on the same now-exposed set.
- **Source authorization:** independently tested simulated registry contract must block executing unknown, invalid, conflicting, superseded, revoked, expired or wrong-project sources. This is a *software property under fixture assumptions*, not actual online authorization.
- If any core gate fails, **NO-GO** to rank 16/64 sweep, automatic use or production. If all pass, next is multiple seeds and external/human-reviewed heldout cases, **not** direct production promotion.

### Why not three sizes immediately?

The user's specific three exact personal sizes **7,680 / 30,720 / 122,880** were already tested in V8.5. None resolved revoked-source evidence, and V8.6 failed isolated policy flips at rank4. First test whether a new *learning objective* fixes the capability at constant size. Original larger checkpoints remain preserved, untouched.

### Traceability

- All V8.7 work lives on `experiment/compass-v8-7` as experimental code and a GitHub Actions workflow on main that checks out that branch.
- No changes to pretrained shared language model, native 5M SELF, production, user data or approved user credentials.
- Report epoch0 raw controls and pretrained head against exact same cold examples. V8.6 Actions artifact is the only warm-start. Compare with trained head on each new item and save both JSON predictions and new personal checkpoint.
