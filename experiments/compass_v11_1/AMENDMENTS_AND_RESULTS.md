# V11.1 — transparent protocol amendments (engineering, NOT a new cold scientific test)

Original `PROTOCOL.md` was committed at `c3fec8ddfef4391a7c688c7995a441454df1290b` before the first V11.1 code tests. The implemented engineering study evolved in the following explicitly documented ways:

1. **Human interpretation review:** added `CONFIRM_CORRECTION` and `KEEP_UNRESOLVED` offline actions. A source-bounded user-confirmed before/after + rationale may train model weights, but does **not** approve a project's principle. Review references SHA-256 of the exact input event; reviewer identity is self-asserted, not authenticated. This was introduced after initial free-text parser tests.
2. **Local chat input:** added `chat:session_id:message_id` as a valid evidence pointer and the `chat_ingest.py` reader for consecutive USER–ASSISTANT–USER triplets from normalized local role-tagged exports. `source_verified=True` means the provided local source has matching role and IDs, **not** independent proof that the user actually authored that message. Input source authenticity remains a future limitation.
3. **Synthetic retrospective grading only:** the pair tested on a correction turn is formed after the user's reply. Its pre-update ranking is useful to test whether earlier feedback changes weights, **not** a genuinely prospective user-choice prediction. No real accuracy or research gain has been measured.
4. **Corpus audit:** the one-phrase high-precision scan identified only one potential explicit correction (T37) in 52 real private planning sections. No independently confirmed supervised pair exists yet.

These changes **did not** alter the original scientific gate or retroactively convert tests into an evidence-backed result. All new tests are lab fixtures. For a true cold benchmark a new protocol will have to freeze data intake, annotation, model, baseline and task order BEFORE collecting any new feedback/answer.

## Result and limitations

The current [CI run](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38186804378) passed **47 fictional engineering tests on Python 3.11, 3.12, 3.13 (141 total executions)**. That includes actual optimizer updates, ambiguous abstention, per-project scope, local export UI XSS escaping, review-hash binding and role-tagged chat parsing. It proves only tested mechanics, not that COMPASS inferred the user's project WHY.

**Scientific status: `FREEFORM_CORRECTION_FEEDBACK_PIPELINE_WORKING`, `NO_INDEPENDENT_LONGITUDINAL_INTENT_LEARNING_PROVEN`.**
