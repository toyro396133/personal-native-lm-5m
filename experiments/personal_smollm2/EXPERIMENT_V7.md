# SELF V7 preregistered style study

Purpose: test whether a 7,680-weight personal residual adapter adds value beyond an explicit style preference prompt on frozen SmolLM2-360M-Instruct, without degrading general knowledge.

Three fictional user styles (two bullets, three numbered steps, one sentence) x seeds 11 and 19 = six training jobs. Twelve original V6 held-out advice tasks plus six new tasks, distinct from twelve training tasks. Forty-eight optimizer steps per trained adapter. All personal test conditions use the same 160 generated-token budget.

Six arms: frozen base; frozen+explicit preference prompt; zero-residual untrained adapter+prompt; trained adapter only; trained adapter+prompt; gated trained adapter+prompt for practical tasks, frozen backbone alone for general questions.

Preregistered metrics: original strict format, alternate permissive format (two numbered points accepted as two bullets; one sentence <=50 words), literal topic-keyword anchor coverage (weak content proxy), joint format+anchor, repeated outputs and budget exhaustion. A matched 76-token control compares prompt alone against trained adapter+prompt. Eight generic QA prompts per job check regression and routing. Raw outputs and a predetermined six-question qualitative review set are saved; no human review claims until someone assesses them.

V6 showed 51/72 strict format for trained adapter+prompt versus 12/72 explicit prompt, but grading and output truncation were confounded. V7 attempts to falsify the effect. It is fictional, structured, English-only; no real user data or reliable deletion of audit logs/learned weights. Do not merge experimental branch into native 5M SELF core or claim production readiness without verified post-run audit.
