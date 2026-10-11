# COMPASS V8.8 — frozen language model as ACTIVE project-thinking partner

## Design decision

The user explicitly requested that the **large language model actively participate in reasoning**, instead of merely producing one 960-dimensional numeric vector for a tiny personal scorer. V8.8 tests an active frozen 360M-LM advisor that reads the project objective, current approved owner policy, evidence and BOTH candidate actions, **generates a concise natural-language analysis**, and then supplies an independent **A/B candidate recommendation**. The unchanged tiny personalized scorer sees this teacher-generated advice at inference and is fine-tuned to make owner-policy-sensitive decisions. Its **EXACTLY 7,680 trainable personal parameters** stay the same; the frozen SmolLM2-360M model is **NOT trained**. Teacher analysis is not trusted as fact or authority.

### Three separate responsibilities

1. **Explicit authenticated user intent/state (not trained):** accept proposed policy changes as versioned preferences only after user confirmation/authorization. This experiment assumes the fictional owner's current preference is already approved; it does NOT have a live user-intent update feature.
2. **Active project-analysis teacher (360M, frozen):** reason over the approved preference plus evidence and both candidate actions, generate a short textual explanation and A/B vote **before labels or decisions are consulted**. Deterministically generate with `do_sample=False` and up to 28 tokens. Canonically sort options in the teacher prompt, then translate the vote to an action text; swapping option presentation to the student cannot trivially change the teacher's result. Record teacher explanation, vote and confidence for every canonical context. Teacher decisions may be wrong.
3. **Small personalized decision scorer (7,680 learned weights):** encode the candidate + teacher explanation and suggested action through the unchanged shared 360M frozen encoder, then train the rank-4 two-matrix shared option scorer with the same supervised owner-policy pairwise objectives used by V8.7. At evaluation generate teacher advice again in advance for each new blind synthetic case (not from gold). The user should never regard the teacher as a production authorization source.

The **non-neural fail-closed source authorization contract** introduced in V8.7 stays separate and does not enter any claimed learned model score. Unknown, revoked, conflicted or expired sources require externally authenticated rejection, not teacher opinion.

## Critical scientific controls

- Frozen warm start: same V8.6 rank4 personal checkpoint, 7,680 weights, seed 11, **18 training epochs**.
- **Control A**: original V8.6 personal head without extra active teacher analysis.
- **Control B**: original V8.6 personal head with teacher analysis injected (isolates effect of teacher input without learning).
- **Control C**: exact same V8.7-style policy-pair fine-tuning but WITHOUT teacher analysis (equal optimizer, schedule and gold labels).
- **Experimental arm**: matched 7,680-weight head receiving independently generated advisor explanations and suggested actions in its frozen encoder, with the **same** training labels/loss and optimizer budget as Control C.
- **Teacher alone**: compare 360M advisor's canonical A/B votes against hidden gold on held-out cases; report teacher's correct policy-only flips, including EN vs HE. A model trained from a bad teacher may perform worse.
- All 80 policy-pair training examples from V8.7, and 16 development policy pairs; use exactly the same 800 old synthetic ordinary training and 160 development items to avoid forgetting. Select epoch using DEVELOPMENT ONLY (policy correct pairs first, correct four-way groups second, dev accuracy third). No test-label tuning.
- **BRAND NEW scripted synthetic policy-only holdout**: families `deployment_communications` and `backlog_triage`, 16 original/change-policy pairs total (8 English/8 Hebrew). The families' exact option/evidence sentences are unseen in training and development. All texts and labels remain researcher authored; it is **not** independent external/human-reviewed generalization.
- V8.6 exposed 16 policy pairs and V8.7 exposed 16 pairs can be used only as **diagnostics**, not pristine holdouts. The original 176 task cases are also exposed diagnostics.
- Expensive frozen embeddings from V8.7 reused. Additional contexts are encoded once. **Teacher inference adds real inference latency and compute on every new real decision**, so even perfect accuracy would require cost/latency review.

## Registered comparative gate — before inspecting V8.8 training outputs

- In the 16 new cold-policy pairs: active-teacher-assisted rank4 must achieve **>=12/16 both old/new decisions correct**, **>=12/16 changed action**, and **>=5/8 both-correct Hebrew pairs**.
- It must outperform equal-training text-only Control C by **at least +2 both-correct pairs** to attribute gains to active teacher participation rather than extra supervised owner-policy training.
- Do not cause major forgetting: test on the previously exposed 176 ordinary decisions should retain **>=140/176** correct as a diagnostic.
- Teacher's own policy-pair performance and whether its generated reasons are intelligible must be reported. Passing gate without meaningful teacher understanding cannot justify deploying.
- Outside source-revocation verification must stay outside language-model decisions and pass fail-closed synthetic unit tests.

**No automatic expansion to 30,720 / 122,880 personalized weights, no production deployment and no native SELF/5M merge** until this rank4 controlled experiment establishes external usefulness. The already-completed V8.5 large heads remain preserved.

## Honest limits

The active advisor is `SmolLM2-360M-Instruct`, which may not follow instructions reliably, especially in Hebrew. A teacher text explanation does not prove genuine internal reasoning. The teacher's chosen answer may simply reflect position, lexical bias or hallucination. The synthetic policy gold remains authored by the experimenter. Source validity is a separate external authorization requirement. The study is about architecture feasibility and failure modes, not proof of personal human-level judgment.
