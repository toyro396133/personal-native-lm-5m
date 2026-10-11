# COMPASS V9.10 — final results: decisive failure beyond three-event, one-principle histories

**Original (preregistered) GitHub Actions:** https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38128954898
**Run status:** `success` in CI for syntax, new fixture audit, seed 11, seed 19, seed 23 and all-seed aggregate.
**Scientific verdict:** **`NO_GO_TO_LM_ADAPTER_INTEGRATION`**, every seed.
**Independent artifact:** three per-seed detailed JSON files and aggregation on that run. All contain per-case predictions and fitted small personal/project readout weights.

## Actual locked-condition results

The pretrained English MiniLM encoder, 48D personal/project GRU and scope-isolated action readouts were TRAINED **only on the original V9.4 three-event English histories**. They were then evaluated WITHOUT FINE-TUNING on 128 newly researcher-authored bilingual EN/HE paired dilemmas, each with FIVE sequential events per scope and a conditional secondary principle. A positive pair requires both the original decision and the decision after changing the three relevant preference sentences to be correct.

| Seed | Candidate BOTH-correct /128 | English /64 | Hebrew /64 | Active verified mandate /32 | TRAIN-only numeric prototype /128 |
|---|---:|---:|---:|---:|---:|
| 11 | **8** | **0** | 8 | 8 | 14 |
| 19 | **10** | **0** | 10 | 8 | 14 |
| 23 | **6** | **0** | 6 | 6 | 14 |
| Across 3 trials (NOT independent human examples) | **24/384** | **0/192** | **24/192** | **22/96** | **42/384** |

The candidate scored **0/32** on the neutral 'open' regime in every seed; it also scored **0/32** on 'revoked' and **0/32** on 'expired' in all seeds. Almost all its paired successes were in the active mandate regime where **both histories have the same correct answer**.

Crucial counterfactual indicator: the intended preference flip for all 96 non-active pairs per seed was almost never reproduced. In seed 11 the candidate changed its answer on **0/128** pairs; in seed 19 on **3/128**; seed 23 on **5/128**. (These raw counts include possible wrong-direction changes, not proofs of semantic intent tracking.) This is a serious failure to preserve sensitivity to primary preference under longer context, regardless of surface language.

## Broken down interpretation: avoid attributing everything to Hebrew

The failure is even present in ENGLISH: 0/64 both-correct on English tasks in every seed. The test changed several factors at once: (1) sequence length from 3 to 5, (2) appending two conditional principles at the end, (3) new task descriptions, and (4) Hebrew on half the data. It is **not legitimate to claim that Hebrew alone caused it**.

Strong hypothesis (not yet causally established): the trained GRU sees two later identical secondary-principle statements across counterfactuals and attenuates the earlier mutable primary preference state. A separate, clearly labeled POST-HOC inference-only study was launched on a new branch to compare sequence prefixes, suffixes and reordered five-event histories. Results from that diagnostic may identify recency interference but can never change this original NO-GO.

The English-centric frozen MiniLM remains a plausible additional cause of Hebrew-specific weakness. To isolate language, future work needs a genuinely multilingual frozen text encoder and independent matched controls, not merely a translated data generator.

## Valid but narrow subcomponents

- Mock UI source routing 128/128; missing/corrupted/conflicting source ASK 32/32 per seed.
- Unrelated user/project history changes did not alter the decision **0/256** per seed.
- Option order symmetry 128/128 per seed.
- The structured privileged control that received ground-truth-like active-status and required operation achieved **32/32 in the active regime**, but only because it received extra policy facts; it is NOT proof of an LLM understanding authority and it does not fix preference-memory collapse.
- Original source-routed neural model has **no policy metadata and does not know which textual commitment is still active**. Passing fake-signed UI ownership is not comparable to authentic real-world authority.

## Strict preregistered gates

For every seed, FAIL:
- All >=88/128
- English >=48/64
- Hebrew >=40/64
- Active conditional >=24/32
- Personal >=40/64
- Project >=40/64

For every seed, PASS:
- Mock origin routes 128/128
- Unknown origin triggers ASK 32/32
- Non-authoritative scope-history invariance at least 95%
- Candidate option ordering at least 95%

Prototype control beat the neural candidate by +6,+4,+8 pairs across seeds, so even cheap numeric-reference improvement gates are not met.

## Scientific decision and follow-up

**Do NOT integrate this small memory readout into the frozen 360M LM** based on V9.9 near-perfect synthetic scores. The V9.10 result demonstrates how brittle the supposed personal/project transfer is to longer, mixed-principle histories.

Recommended next model experiment (only after post-hoc diagnosis):
1. Replace a single last-hidden-state preference summary with **typed, persistent multiple concept records**: source, active workspace, relevance/applicability, reason, validity period, accepted/rejected corrections. Do not discard an old but relevant principle when a new unrelated statement arrives.
2. Train with mixed-order 3–8+ history events and explicit counterfactual corrections; test on held-out lengths and unseen principled conflicts, not merely three-sentence paraphrases.
3. Evaluate Hebrew using a multilingual semantic encoder, same-data English control and independently authored human bilingual dilemmas; test irrelevant-history changes and actual applicability conditions.
4. Retain V9.7's separate policy provenance/authority controller; do not make a text-only preference scorer responsible for safety validity.
5. Test actual frozen 360M full-history prompted baseline on matched new cases before claiming the compact learned memory outperforms it, and keep the true SELF adapter out of this proof-of-concept until objective gates pass.

**Limit:** the 128 original tasks and labels are still synthetic and researcher-authored. This is a negative counterexample in a toy environment, not a measured real-world failure rate.
