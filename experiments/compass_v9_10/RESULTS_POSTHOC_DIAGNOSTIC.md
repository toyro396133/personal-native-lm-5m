# COMPASS V9.10-D — post-hoc memory ordering explains much of the collapse

**Original cold V9.10:** https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38128954898 — preregistered NO-GO.
**This *retrospective* diagnostic:** https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38129235547
**All three-seed data artifact:** https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38129235547/artifacts/11698910925
**Protocol:** [POSTHOC_DIAGNOSTIC_PROTOCOL.md](POSTHOC_DIAGNOSTIC_PROTOCOL.md)
**Scientific verdict remains:** `NO_GO_TO_LM_ADAPTER_INTEGRATION`. This intervention is NOT newly blind testing and must not change the V9.10 preregistration outcome.

## Six inference-only manipulations, no model training on V9.10 test examples

These results score the SAME 128 already-observed English/Hebrew paired cold dilemmas with the same old English-trained MiniLM/GRU + separate heads. Each variant changes only the order/length/content of the personal and project history sequence. All gold answers, options, task text and signed mock source metadata are left untouched.

| Sequence passed to GRU | Seed 11 BOTH /128 | Seed 19 BOTH /128 | Seed 23 BOTH /128 |
| --- | ---: | ---: | ---: |
| **Original five**: three preference statements, then TWO other conditions | **8** | **10** | **6** |
| First THREE preference statements ONLY | **57** | **55** | **49** |
| First FOUR: three primary statements then one secondary condition | **24** | **20** | **24** |
| Last THREE: one primary statement plus both secondary conditions | **8** | **11** | **6** |
| Same FIVE, but **conditions moved to the front**, preference observations last | **56** | **55** | **49** |
| Five events, last two conditions **replaced by repeats of preference** | **57** | **58** | **58** |

The MOST diagnostic comparison is **same five statements in different order**. Moving the identical two secondary statements ahead of the three preference statements restores from **8/10/6** to **56/55/49** paired outcomes, without changing weights or the semantic content. This is powerful evidence that event order/recency matters in the small 48D recurrent-state memory. Note that mere five-event length cannot explain this controlled difference: both versions still have FIVE statements.

Removing just ONE secondary statement (first4) yields an intermediate 20–24 pairs; retaining last three gives effectively the same collapse as full five. This supports progressive swamping of earlier state by later context. State-vector distance metrics and raw per-case outputs are preserved in artifacts to investigate whether this is encoder representation vs downstream scorer sensitivity; not quantified into causal certainty here.

## Language and conditional-policy weaknesses remain even after the order intervention

| Scope metric | Seed 11: original -> secondary-first | Seed 19 | Seed 23 |
| --- | ---: | ---: | ---: |
| English BOTH-correct /64 | **0 -> 45** | **0 -> 43** | **0 -> 38** |
| Hebrew BOTH-correct /64 | **8 -> 11** | **10 -> 12** | **6 -> 11** |
| Active binding BOTH-correct /32 | **8 -> 5** | **8 -> 4** | **6 -> 6** |
| Expected change/no-change agreement /128 | **32 -> 74** | **33 -> 72** | **35 -> 66** |

**No near-complete recovery.** Even after moving the history conditions first, Hebrew scores remain ~11/64, far below the original required 40/64. The existing English-centric encoder was not designed to transfer Hebrew meaning. The recovered preference memory still cannot enforce an active task-specific obligation, because that job belongs to a provenance-aware external policy controller rather than a mere preference readout. The original test intentionally did not pass the policy manifest to the neural scorer.

## Interpretation and concrete architectural change

V9.9's near-perfect results on short synthetic histories hid a brittle internal assumption: that the final GRU state is sufficient to encode everything that still matters. When longer histories contain a second condition, the final vector can drift away from earlier relevant global or project principles. Last-message dominance is useful for **actual corrections** but dangerous for **unrelated, non-revoking facts**.

**Next research V9.11 should not only tune a bigger GRU.** Use explicit multi-slot principle memory with provenance:
- Individual principle record: owner/scope, content/concept, reason/evidence, effective interval, revocation or supersession, confidence, applicability predicates, and immutable source ID.
- Distinguish **correction** (explicitly supersedes same principle) from **new independent condition** (must not erase old principle).
- Use a retriever/attention selector over still-relevant active principle slots; avoid collapsing all historical events into only the most recent recurrent state.
- Preserve separate personal-global and project-local slots, with explicit request ownership; no sharing of hard project constraints into global personal preference.
- Bilingual EN/HE encoder and independently written test tasks. Train on mixed event orders/lengths, multiple principles, corrections, revocations and real applicability conflicts.
- Compare with original GRU, simple retrieval, frozen full-history LM, and reliable provenance/controller; require counterfactual flips even after irrelevant facts are appended.

**Caveat:** this study is retrospective and only probes already observed researcher-authored synthetic samples. Results diagnose a likely failure mechanism, not robust real-world accuracy and not a new model success.
