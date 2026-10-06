# Personal-Native LM 5M — Prototype v0.2

This repository is the first working prototype of a language model designed to
consume a persistent personal model as a native computational input.

## What is already implemented

1. **Canonical PersonalState**
   - Core
   - Policies
   - Worlds
   - Routing
   - confidence/evidence metadata

2. **T_read / PersonalStateReader**
   Converts PersonalState into learned latent prefix tokens.

3. **~5M decoder-only Transformer**
   The same text can produce different hidden states/logits for different users.

4. **Safe state update boundary**
   The LM cannot directly write persistent personal state.
   Update proposals must pass evidence/confidence gating.

5. **Synthetic personalization curriculum**
   A small smoke-training loop that tests whether state can control a target.

6. **Real text pretraining entry point**
   `train_corpus.py` accepts a UTF-8 corpus.

## Run

```bash
python -m pip install -r requirements.txt
python train_smoke.py --steps 40
```

Create a sample personal state:

```bash
python make_example_state.py
```

Run tests if pytest is installed:

```bash
pytest -q
```

## Important limitation of v0.1

The tokenizer is byte-level on purpose, so the architecture has no tokenizer
dependency and can be validated immediately. It is **not** the tokenizer I
would use for the real Hebrew model.

The next production step is a Hebrew-aware BPE/Unigram tokenizer (roughly
4K–8K vocabulary), followed by:
- Hebrew general-language pretraining;
- personalization curriculum with matched prompt / different-user examples;
- negative examples where irrelevant personal state must be ignored;
- confidence calibration;
- a learned T_write proposal model;
- reader portability tests across two different LM backbones.

## Training objective

The final training mix should contain three families:

### A. General language
Neutral state. Learn ordinary language modeling.

### B. Personal contrastive examples
Same prompt, different PersonalState, different correct continuation.

### C. Personal irrelevance examples
Different PersonalState, same correct continuation.
This is crucial: the model must learn *not* to personalize when personalization
is irrelevant.

The central architectural property is:

`PersonalState -> T_read -> latent prefix -> LM`

not:

`PersonalState -> prose prompt -> LM`


## v0.2: Hebrew BPE profile

For the real Hebrew run, use:

```bash
python train_tokenizer.py corpus.txt --vocab-size 4096
python train_hebrew.py corpus.txt --tokenizer hebrew-bpe-4096.json
```

The BPE profile uses:
- vocabulary: 4,096
- d_model: 224
- 8 attention heads
- 6 Transformer blocks
- FFN: 896
- 4 latent personal-prefix tokens
- PersonalState ABI: 228 dimensions

Total trainable parameter count is approximately 4.92M, including `T_read`.
