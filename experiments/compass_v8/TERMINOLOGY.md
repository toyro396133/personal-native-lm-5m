# COMPASS terminology and ownership boundaries

| Term | Definition | Owns | Does NOT own |
|---|---|---|---|
| **SELF** | Model's internal self-reference / modeled "I" mechanism under independent research | Internal SELF-model experiments | The user's project prioritization or personal history |
| **COMPASS (מצפן)** | Small trainable personal decision-policy adapter | Transferable user-specific importance, triage and decision preferences | Immutable personal facts, canonical project state, project's entire identity |
| **Project Memory** | Versioned, inspectable per-project knowledge graph | Core purpose, layers, dependencies, decisions, source, rationale, validity and current state | A new assistant's hidden weights or subjective authority |
| **Priority Engine** | Decision system combining frozen LLM, Project Memory, optional COMPASS and explicit rules | Ranked next actions, rationale, evidence and abstention | Replacing the user's authority on ambiguous strategic decisions |
| **Workflow Checks** | Tests, QA, checks and status metrics | Evidence about whether core work is healthy | An independent justification to displace the project mission |

**Important distinction:** A node can be *peripheral in architecture* and *urgent in execution* because it protects a failing core dependency. COMPASS must reason over both dimensions.

**Preference update:** Project maps and their effective dates come from an explicit versioned external source. COMPASS never overrides a newly approved project decision simply because it learned an older preference from training.

**Human authority:** Approved user decisions outrank inferred assistant assumptions. Insufficient evidence requires investigation or clarification, not fabricated certainty.

**Data boundary:** Multiple projects may share transferable high-level prioritization policy, but must not share actual secret records, goals, donor data, policies, or logs unless deliberately authorized. Research examples remain fully synthetic.

**Legacy naming:** Earlier \`personal_smollm2\` experiments (V1–V7) used the term "personal adapter" and sometimes called that SELF. They are now historical predecessor experiments of the COMPASS research lane. This naming correction does **not** alter or replace the separate existing SELF-mechanism experiments or the native 5M SELF model.
