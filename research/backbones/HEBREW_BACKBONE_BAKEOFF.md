# Hebrew pretrained backbone bake-off

Goal: choose the smallest pretrained Hebrew causal LM that already has
reasonable Hebrew while remaining practical for **full-parameter training** on
GitHub-hosted runners. Language quality is infrastructure for the SELF/CORE
research, not the research target.

## Measured candidates

### goldfish-models/heb_hebr_1000mb

- parameters: 124,770,816
- detected context: 512
- vocab: 51,200
- chars/token on shared Hebrew slice: 4.3018
- bits/character: **1.5026**
- eval throughput: 745.0 tokens/s
- full AdamW forward+backward throughput: **142.6 tokens/s**
- peak process RSS during full-training canary: **3.47 GB**
- all parameters confirmed trainable: yes
- benchmark completed successfully

### Norod78/distilgpt2-base-pretrained-he

- parameters: 81,912,576
- detected context: 1024
- vocab: 50,257
- chars/token on shared Hebrew slice: 3.8691
- bits/character: **1.8869**
- eval throughput: 700.2 tokens/s
- full AdamW forward+backward throughput: **156.9 tokens/s**
- peak process RSS during full-training canary: **2.64 GB**
- all parameters confirmed trainable: yes
- benchmark completed successfully

### Slasky/HebrewGPT-296M

Status: **not comparable / disqualified for current workflow due upstream
tokenizer-package inconsistency**.

Three attempts failed before a valid language/full-training benchmark could be
completed.

Observed packaging conflict at current Hub revision:
- HF config loaded by the model declares vocab_size=8192.
- repository README / standalone generate.py still describes a 32K
  SentencePiece tokenizer.
- tokenizer_config states that the real tokenizer is cl100k_base remapped to
  8192 IDs and that tokenizer.model is reference-only.
- the repository does not expose the required remapping through the ordinary
  AutoTokenizer path.
- direct SentencePiece therefore emits token IDs outside the model's 8192-row
  embedding table and fails with IndexError.

This is an upstream reproducibility problem, not evidence about the model's
quality or GitHub RAM requirements. We should not spend research time
reverse-engineering the tokenizer unless no cleaner backbone passes the Hebrew
quality threshold.

## Generation inspection

Both successful models are base LMs rather than instruction-tuned assistants.

Goldfish generations were generally more locally coherent Hebrew and less
web-page-like on the fixed prompt bank, although still weak at explicit
instruction following and reasoning.

DistilGPT2 frequently drifted into unrelated web/Wikipedia/blog fragments and
showed more repetition/topic drift.

## Current decision

**Temporary winner: goldfish-models/heb_hebr_1000mb.**

Why:
- materially better normalized Hebrew LM score than DistilGPT2
  (1.50 vs 1.89 bits/char);
- only ~9% slower in the measured full-parameter AdamW canary;
- still comfortably fits GitHub runner memory (~3.5 GB peak in the canary);
- standard GPT-2-compatible architecture and clean tokenizer path;
- all parameters can be trained, satisfying the SELF research requirement.

Next validation before adopting it as the primary SELF backbone:
1. run a longer full-training canary (not just two measured steps);
2. verify checkpoint save/resume with optimizer state;
3. add SELF module to a copy while preserving the base path;
4. compare normal language retention baseline vs SELF over a small matched
   continuation budget;
5. only then scale the structural/causal SELF assays.
