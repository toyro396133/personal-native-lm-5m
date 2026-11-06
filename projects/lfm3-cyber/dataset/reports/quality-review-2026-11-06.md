# LFM3-Cyber dataset quality review — 2026-11-06

## Executive result

**Disposition: structurally improved; not yet approved for training.** The initial 100-example baseline was consistently split and has no detected duplicate or cross-split similarity issue in the checks below. Three JSON Schema violations were found and corrected. A gap in the validator was also fixed: it now applies the dataset JSON Schema to each record, rather than merely checking that the schema file parses as JSON. The latest CI rerun is pending.

This is an automated and model-assisted quality audit, not an independent human subject-matter certification. All 200 current examples remain `pending`; none is represented as expert-reviewed.

## Dataset inventory

- Total: 100
- Split: 80 train / 10 validation / 10 test (80/10/10)
- Unique example IDs: 100
- Unique scenario IDs: 100
- Exact duplicate conversations detected: 0
- Cross-split near-duplicate pairs detected: 0 at the validator's 0.88 SequenceMatcher threshold
- Source families used: 7; all family keys resolve in the 9-entry source registry
- Manual expert review: 0/100

### Domain distribution

| Domain | Count |
|---|---:|
| SOC | 17 |
| Network security | 15 |
| Cloud infrastructure | 15 |
| Application security | 13 |
| Incident response | 13 |
| Threat intelligence | 10 |
| Fundamentals | 9 |
| Penetration testing | 8 |

### Difficulty and orientation

- Difficulty: 10 beginner, 49 intermediate, 41 advanced.
- Orientation: 88 defensive, 6 authorized-offensive lab, 6 conceptual.
- Every record has three messages (system, user, assistant).
- Assistant answers: 61–111 words; mean 79.5 words.
- Only two distinct system-prompt variants were found.

## Findings and actions

### F-01 — JSON Schema was not being applied to records (High; fixed in branch)

The previous validator checked required fields and selected metadata rules but did not apply `dataset/schema.json` recursively. The workflow's schema step only parsed the schema files; it did not validate dataset records against them.

**Action:** added a dependency-free validator for the JSON Schema keywords used by this project's schema, wired it into the dataset validator, and reused it in the report generator. Added unit tests for nested type violations and unexpected properties.

### F-02 — Three records violated the declared schema (High; fixed in data)

The following `expected_skills` values were strings rather than arrays:
- `lfm3c-pilot-080`
- `lfm3c-pilot-092`
- `lfm3c-pilot-093`

**Action:** normalized each value to a one-element array. No prompt or answer text was changed.

### F-03 — Technical correctness still lacks independent review (High; open)

All 100 records remain `verification.status=pending` and `quality.review_status=pending`. Structural checks cannot establish that recommendations are correct for every product/version or operational context.

**Action required:** domain reviewers should verify technical statements, prerequisites, caveats, and remediation advice before any record is promoted to `reviewed` or used in training.

### F-04 — Source traceability is too broad for a mature dataset (Medium; open)

The registry contains source-family landing pages or general terms pages for many examples (for example broad NIST/CISA/OWASP/MITRE pages), rather than a specific document, section, or version supporting each concept. No external dataset rows, PCAPs, raw Splunk LFS payloads, or Sigma rule text are represented as copied content; the examples are marked original synthetic.

**Action required:** attach a precise source reference (document/page, version or retrieval date, and relevant section) to each example where a concrete external claim is made. Retain the current licensing caveats; keep OTRF excluded until its conflicting license notices are resolved.

### F-05 — Training variety is still narrow (Medium; open)

All examples use a three-message pattern and only two system-prompt variants. Answers are concise and consistently structured, which is useful for a pilot but may teach a narrow response style. Difficulty is weighted toward intermediate/advanced (90%), and only 8% of examples are explicitly authorized-offensive lab scenarios.

**Action required:** before expanding toward 500, diversify interaction patterns and evidence formats: sanitized synthetic logs, configuration fragments, detection-rule review, incident timelines, short multi-turn clarification, report writing, and beginner-level explanations. Keep offensive practice explicitly scoped to isolated authorized labs.

### F-06 — Dataset breadth is reasonable for a pilot, not yet evidence of balanced coverage (Medium; open)

All eight planned domains are represented, but counts are uneven and some domains have limited task variety. Domain count alone does not demonstrate coverage of the important subskills within each domain.

**Action required:** define a subskill coverage matrix per domain and use it to plan the next batch rather than adding examples only to the domains with the lowest counts.


### F-07 — Lab metadata overstated the authorization context (High; fixed)

Two scenarios described production endpoints or passive-only production review but were marked `authorized_lab`, whose metadata promised an isolated training VM/network. Two additional lab scenarios did not make the isolated-lab boundary explicit in the user prompt.

**Action:** reclassified the production/passive-only scenarios `lfm3c-pilot-031` and `lfm3c-pilot-096` as defensive, including their orientation and authorization metadata; clarified the isolated authorized-lab context in `lfm3c-pilot-033` and holdout record `lfm3c-pilot-035`. The corrected environment distribution is 88 defensive / 6 authorized_lab / 6 conceptual. The Test record remains in Test and has not been used for training or tuning.

## Safety and split-isolation review

- The six authorized-lab examples carry explicit lab scope and a permitted-target boundary.
- The examples reviewed here emphasize defensive investigation, containment, configuration review, and bounded lab practice; no third-party target instructions or destructive payload procedures were identified in the sampled/static checks.
- Test remains a holdout. Do not use its contents for training, prompt design, or iterative tuning.
- No training run was started and no model weights were downloaded.

## CI status

The JSON Schema validator and tests were added to the branch. The first run exposed a defect in the new test fixture (the fixture accidentally declared the supposedly unexpected key as allowed); that fixture was corrected and the unit-test suite then passed. The next run exposed an integration mismatch: the audit passed a source-registry argument that the validator signature did not yet accept. That signature and registry check were corrected in the latest branch commit. At the time of the initial 100-example audit, the `scaffold` job had passed its checks while architecture inspection was still running. Following expansion to 200 examples, the new CI run is queued/pending; final schema, near-duplicate and workflow results must be confirmed against that run.



## Expansion review — 200-example milestone

The dataset was expanded from 100 to 200 original synthetic examples. The new batch adds 80 defensive training scenarios, 10 validation holdouts and 10 test holdouts. The existing six authorized-lab examples and six conceptual examples were retained unchanged.

### Updated inventory

- Total: 200
- Split: 160 train / 20 validation / 20 test (80/10/10)
- Unique IDs: 200
- Unique scenario IDs: 200
- Exact duplicate conversations: 0 in the static inventory scan
- Environment/orientation/authorization scope mismatches: 0
- Manual expert review: 0/200
- All 200 records remain pending; no training has started.

### Domain distribution

| Domain | Count |
|---|---:|
| SOC | 33 |
| Network security | 29 |
| Cloud infrastructure | 29 |
| Application security | 28 |
| Incident response | 25 |
| Threat intelligence | 20 |
| Fundamentals | 19 |
| Penetration testing | 17 |

### New findings and limitations

- Six generated IDs initially collided with pre-existing validation/test IDs. The six new training records were renumbered to unique IDs and scenario IDs before finalizing the split.
- The new examples are predominantly defensive decision-making and operational analysis. They improve breadth but do not yet solve the earlier format-diversity limitation: all examples still use three messages and two system-prompt variants.
- The updated source-family distribution is NIST 77, CISA 54, MITRE 27, OWASP 27, Sigma 12, Splunk Attack Data 2 and CIC-IDS2017 1. These are conceptual references; no external dataset rows or rule text were copied.
- Cross-split near-duplicate scanning and record-level schema validation must be confirmed by the post-expansion CI run. The report JSON marks those checks as pending until CI completes.
- All 200 examples still require independent technical review. Automated structural validity is not professional certification and is not sufficient authorization to begin SFT.

### Next quality gate

Before training, add a subskill coverage matrix and diversify examples with sanitized synthetic event snippets, configuration fragments, multi-turn clarification, incident timelines and structured reports. Then complete independent expert review, resolve document-level source traceability gaps, and run the full acceptance suite.
