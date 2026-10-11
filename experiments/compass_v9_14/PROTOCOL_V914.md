# COMPASS V9.14 — Evidence-first project-decision mining on REAL Chacham Lev

**Protocol frozen before V9.14 predictions.** V9.13 real private-doc retrieval (6/6 for both TF-IDF and trivial lexical overlap) is **not a breakthrough**. Here COMPASS goes past search to proposing *typed, scoped memory updates*, not approving them.

## Output and no-go boundary

- On the user's **private local clone only**, mine exact decision/evidence units from a historical tree at pinned Git ref. Capture file path, Git SHA, heading hierarchy, line range, short source quote, claimed decision status, confidence/reasons, and contradictions/supersession references. Do **not** copy source docs into the public model repo or CI artifacts.
- Each proposed record remains **PENDING_REVIEW**. If the evidence cannot determine authority, project scope or whether a statement is historical/proposed/current: `ASK` / `UNRESOLVED`. Source wording is NOT proof of author authority or actual acceptance.
- Personal COMPASS is out of scope for automatic cross-project promotion. All mined changes are PROJECT-scoped only. The accepted v9.12 SQLite store is the only persistent human-approved destination.
- No changes to the Chacham Lev repository. No 360M SELF or any neural language model trained: this is the COMPASS understanding pipeline.

## Planned extractor

1. Split Markdown by headings, preserve hierarchy (h1–h6) and line numbers. A heading beginning `החלטה מאושרת` or `הכרעה מחייבת` is a higher precision authority signal than the global document title `הצעת מודל`. A subheading `הצעה`, `מוצע` or `לא אושר` must not inherit a parent approval.
2. Differentiate `APPROVED_CLAIM`, `PROPOSED_CLAIM`, `HISTORICAL_EVIDENCE`, `UNRESOLVED` and `REVOKED_OR_SUPERSEDED_CLAIM`. Heading/paragraph cues are weak evidence only, not user-authenticated truth.
3. Use a **separate temporal revision graph**: stable explicit Tnn and D-nnn identifiers, `מחליפה`/`מתקנת`/`מבוטלת` signals, source links and chronological document commit. A new paragraph or different implementation is not automatically a supersession event.
4. For each status claim, expose the exact local cited line excerpt and basis used; never assume authority simply because the heading says "approved" in a user-editable document.
5. Compare status detections against a trivial whole-document keyword baseline and first-matching-keyword paragraph baseline on the same sections. Report coverage/precision proxies and abstention. Weak document-heading labels are **NOT human-independent gold**. No tuning any thresholds after looking at the private outcome.
6. Use **V9.13 historical checkpoint** `55b0ac175513` for existing knowledge and later main commit `fe9863278a04dedf27a0c4fcb8cca56b91ec4544` for chronological holdout; reject future-file leakage.
7. Test on a private-doc **real case inventory** (at least five files: an updated importance-map, interview protocol, telephone close-out, telephone status proposal with authoritative subheadings, and telephone architecture plan). For scientific credit, request **independent user/human labels** for the cases; until then only call it an exploratory silver audit.
8. Build unit and integration tests against entirely invented Markdown and local Git repos: inherited headings, approved subheading beneath proposal document, proposed subheading beneath approved section, supersession, missing reasons, same decision in different years, unknown provenance, anti-leakage, no autoapproval and reproducible line provenance. All CI data fictional.

## Locked acceptance

**Engineering:** 100% test and CLI pass across Python 3.11–3.13; every generated record has verifiable source and no automatic approval; contradictory/unverified case results are visible as uncertainty. Replays deterministic.

**Research:** real historical private docs evaluated from a read-only clone with grounded case references, vs matched equal-input baselines; **no claims of independent accuracy or neural language-learning gain without a separate human-labeled blind benchmark and a real improvement over simpler methods**.

Scientific statuses: `REAL_SOURCE_STRUCTURED_PROPOSAL_PILOT` if operational; `NO_PROVEN_AUTONOMOUS_INTENT_LEARNING` until external evaluation.
