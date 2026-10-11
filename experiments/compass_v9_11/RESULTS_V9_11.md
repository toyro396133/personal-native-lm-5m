# COMPASS V9.11 — Results: stable slot isolation, no learning gain versus simple prefix shortcut

**Workflow:** https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38129839017
**State:** CI completed SUCCESS, second attempt after correcting a tensor-length mismatch; seed 11/19/23 and aggregate completed.
**Preregistration:** [EXPERIMENT_V9_11.md](EXPERIMENT_V9_11.md) committed before any cold scores.
**Scientific verdict under predeclared synthetic checks:** `PROVISIONAL_SYNTHETIC_COMPASS_SLOT_MEMORY_PASS` in all three seeds.
**Production-readiness:** `NO_GO_COMPASS_REAL_WORLD`. Not production-ready; no real historical work, independent human evaluation or human-verified updates yet.
**This is COMPASS work, NOT SELF training.** No native neural adapter and no 360M weights changed.

## Main 128-paired fresh researcher-authored examples per seed

Same frozen English-centric text encoder, TRAIN-only 48D GRU, separate personal and project intent readouts, same (mock) typed policy checker for all policy-enabled comparisons. Both counterfactual decisions must be correct. All 16 new task families have English/Hebrew, personal/project and active/revoked/expired/open statuses.

| Method | Seed 11 /128 | Seed 19 /128 | Seed 23 /128 |
|---|---:|---:|---:|
| Old single continuous 5-event GRU, no policy checker | 12 | 12 | 10 |
| **Old continuous 5-event GRU + SAME policy checker** | **32** | **32** | **32** |
| Versioned COMPASS principle slot only, no policy checker | 58 | 57 | 59 |
| **Versioned COMPASS principle slot + SAME policy checker** | **86** | **85** | **82** |
| **Simple first-three-position truncation + SAME policy checker** | **86** | **85** | **82** |

The stable structured slot retrieves three evidence statements belonging to the active personal or project principle instead of feeding them together with later unrelated conditions into a single recurrent sequence. This preserves sensitivity to old relevant principles and prevents unrelated context from overwriting memory. The common policy checker enforces active typed obligations on both systems; **32/32 active** paired dilemmas are trivially correct in both policy-enabled models because the controller receives a researcher-authored structured contract and matching action operation IDs. Do NOT attribute the 32 obligation successes to the neural model.

**Crucial matched control:** The new versioned slot candidate obtained **exactly the same accuracy** as merely taking the first three preference statements by their fixed position. Consequently, V9.11 supports the architectural utility of preserving versioned, auditable, separately scoped principles, but it **does not provide evidence that versioned records teach the neural learner more than a simple fixed-position shortcut** on these fixtures. The test's author supplied exact role/status/scope and positions; source labels do a large part of the work.

## English and Hebrew

| Seed | COMPASS slot+policy English /64 | Hebrew /64 | User /64 | Project /64 | Neutral statuses open/revoked/expired /96 |
|---|---:|---:|---:|---:|---:|
| 11 | 59 | 27 | 36 | 50 | 54 |
| 19 | 59 | 26 | 35 | 50 | 53 |
| 23 | 55 | 27 | 35 | 47 | 50 |

The English-centric frozen encoder and English-only TRAIN examples still leave Hebrew substantially weaker than English. V9.11 did not solve multilingual model comprehension or human-intent grounding.

## Protocol invariance and uncertainty gates

Across all seeds:
- Verified mock source owner declaration accepted on 128/128; no source/owner inferencing by model.
- Missing/invalid source event -> ASK **32/32**.
- Tampered active slot signature or foreign declared scope -> ASK **32/32**.
- Shuffling the slot order or moving the superseded historic record to another position changed **0 out of 512 individual decisions**. Old version identity and provenance are explicit, and old record still contains a reversed copy of current evidence — so this test is about ordering/marking obsolete data, not independently detecting semantic contradictions.
- Changes to unrelated personal/project history -> **0/128** decision changes.
- Option-permutation equivariance -> **128/128**.
- Every active mock policy obligation -> **32/32**, with or without slots.
- All declared numerical gates passed. No thresholds were changed after cold predictions.

## First-run technical failure, correction and verification

Preflight passed. The first training attempt failed when it tried to stack a 3-event active preference stream alongside 5-event irrelevant streams. The existing V9.4 `FrozenFeatureCache` implementation expects equal-length stacked user+project histories. Fixed the *data shape only* in commit `79cec87df48fa275512cfe9123cc40895a5b6d39`: trim irrelevant streams to three for the isolated readout while preserving the authoritative versioned slot evidence. The corrected attempt passed; case texts, policy labels, preregistration gates and model weights were unchanged. This correction does not improve the robustness of the unstructured baseline and should not be interpreted as scientific tuning.

## Important limits

1. All 128 task pairs and mock policy records are **researcher-authored synthetic data**. No independent human judges, no real projects, no real accountability for source authority.
2. A versioned principle slot is assumed to already exist with truthful `kind=preference`, `owner_type`, `scope_id`, `version`, `supersedes`, `active`, `evidence` and `reason`. The model does not yet infer whether a new chat message corrects a past principle or is only contextual. That's the central *COMPASS learning* problem.
3. Signed keys are public test keys. Signature verification only demonstrates software wiring and does not verify real-world identity, consent or contractual authority.
4. The V9.11 protocol treats equal-information policy-enabled comparisons seriously, but the policy contract contains fixture-generated correct action operation tags. Structured hard-control success != neural interpretation of law or intent.
5. The last active preference evidence is always extractable as the first three entries in synthetic histories, so the retrieval demonstration is artificially simple. The separate `superseded` mock record contains reversed copies of current evidence; there is not yet a truly changing preference across real historical states.
6. Both readout and encoder remain unchanged English-trained small external networks. No SELF/360M model modification or integration.

## V9.12 — recommended next COMPASS experiment, NOT RUN

**Focus: learning and maintaining the actual multi-slot principle memory** rather than just reading perfectly annotated slots.

- Build sequential change events: explicit corrections, new independent principles, validity changes, justifications and scope changes; cases deliberately permute chronology versus message arrival/order, and sometimes show an earlier principle under a new phrasing. The system should decide which records to create/update/revoke and why, with test-time uncertain/ASK instead of silently selecting the most recent utterance.
- Remove privileged `kind/active/supersedes` fields at the learning stage; provide only realistic origin (actual user vs project activity) and text/timestamps. Test predicted slot edits against independently curated labeled updates, including false positives (overwriting valid principles), false negatives and conflicts.
- Deliberately make a positional first-three/last-three shortcut fail by interleaving principle evidence and context unpredictably; keep a position-free retrieval baseline and existing source-only small GRU comparison on identical inputs.
- Preserve **COMPASS personal ↔ COMPASS project** architecture, including controlled transfer of a general principle to other projects only when applicability conditions allow it. No project-specific rule should accidentally become a global user preference.
- Evaluate novel genuine Hebrew and English multi-principle histories with a multilingual encoder, equal-input retrieval/prompt baselines, and human-authored independent challenging tasks. Prefer long-running project evolution over single synthetic binary choice.
- Keep V9.7 provenance-based external binding enforcement separate; require approval on uncertain/invalid claims. Explain **why** every principle was learned or superseded, with traceable source and decision rationale.
- Only after real-world COMPASS component gates pass should we contemplate an optional connection to any LM, as a separate integration project. There is no goal here to train SELF itself.

**Do not merge experimental V9.11 into main** as if ready for production. The workflow file alone is on main as a trigger; the V9.11 experiment and findings remain on `experiment/compass-v9-11`.
