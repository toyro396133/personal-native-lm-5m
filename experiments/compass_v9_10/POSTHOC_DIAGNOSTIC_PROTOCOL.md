# V9.10-D — explicitly POST-HOC memory-order diagnostic

**Not a new blind benchmark.** We have already observed V9.10 cold answers (8/128, 10/128, 6/128 for seeds 11/19/23). No experimental acceptance gates will be repaired, changed or claimed as newly met. V9.10 stays `NO_GO_TO_LM_ADAPTER_INTEGRATION`.

Hypothesis suggested after seeing the failure: an English-trained V9.3 GRU has seen only three short history events; V9.10 appends two common conditional-principle sentences AFTER the three mutable personal/project preference observations. This may suppress earlier state through recency/length shift. Hebrew encoding and genuine multi-principle interference are competing explanations.

## Frozen-weight interventions on the SAME V9.10 cases

Retrain baseline numeric GRU and isolated heads from original V9.4 TRAIN only, with exactly the original optimizer, seed and within-TRAIN family holdout. On V9.10 cold cases perform **inference-only** history transformations; no tuning or label-dependent choice:

- `full5_original`: original three preference observations plus two conditional-principle explanations.
- `first3_preference_only`: remove the two appended explanations. Removes content and reduces length, confounded by both effects.
- `first4_one_secondary`: remove only last explanation.
- `last3_secondary_dominant`: retain final primary preference observation plus both secondary explanations.
- `secondary_first_five`: move both secondary explanations ahead of the three preference observations; text and length exactly unchanged (order test).
- `repeat_primary_tail_five`: replace the two secondary explanations with repetitions of the last primary preference observation. Length fixed but meaning/content changes.

For each method compute BOTH-correct pair outcomes, expected counterfactual flip agreement, original/changed *numeric memory state* distance per scope, and breakdown by Hebrew vs English, neutral vs active. This will show whether the lost flip is associated with loss of change-sensitive numeric representation or with the action scorer. No variant is selected as a new model checkpoint and **no test labels are used to train**.

A strong improvement from order-only change would support recency interference (still not prove a single cause); no improvement could indicate encoder/conditional task shift or neural decision-head mismatch. This is hypothesis-generating, not a new generalization result. Preserve all variants and all three seeds, no cherry-picking.

Mock UI source metadata remains consistent across variants and remains a synthetic proxy. The 360M native LM was not modified.
