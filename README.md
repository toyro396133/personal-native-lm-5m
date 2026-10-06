# Personal-Native LM 5M — Prototype v0.6

A research prototype for training **both** a shared language model and a persistent personal model, then freezing the shared model while each user's personal model can keep learning and influence the shared model only at runtime.

## Core lifecycle

### 1. Train the Main LM

The shared Main LM is trained normally. In v0.6 the Hebrew profile contains **4,891,731 parameters** and was trained on natural Hebrew before personalization.

### 2. Train the personal interface

Once language ability exists, v0.6 freezes the ordinary language backbone and trains only the personal interface inside the Main LM together with the shared personal-learning stack.

The personal interface consists of:

- the request-conditioned `ContextualPersonalController`;
- each Transformer block's `cond_delta` and `cond_gate` projections.

The ordinary token embeddings, attention, MLP, final normalization and LM head stay frozen during this integration phase.

### 3. Freeze the released Main LM

After the system is closed/frozen:

`frozen Main LM + changing per-user Personal Micro-Model -> temporary runtime conditioning`

The user's personal model may continue learning. It changes activations for the current request, **not the Main-LM weights**.

## Architecture

### Main LM

- decoder-only Transformer;
- ~5.23M parameters in the byte bootstrap profile;
- **4,891,731 parameters** in the 4,096-vocabulary Hebrew profile;
- 6 Transformer blocks;
- 8 attention heads;
- request-conditioned activation modulation inside every block.

### Shared personal-learning system

- ~91K parameters;
- `PersonalHistoryEncoder`: turns interaction history into an initial personal latent;
- `PersonalStateDecoder`: turns the compact latent into the canonical 228-dimensional Personal State ABI.

### Per-user Personal Micro-Model

- **116 trainable parameters per user**;
- initialized from interaction history;
- can keep learning after the Main LM is frozen;
- decoded into Core / Policy / World / Routing state through the shared decoder.

## Request routing

The controller performs a five-way request-dependent route:

1. Core
2. Policy
3. World
4. Routing
5. None

`None` lets factual or otherwise irrelevant requests explicitly ignore personal state.

## v0.6: natural Hebrew + personalization without forgetting

v0.6 moved from generated bootstrap text to a public natural-Hebrew corpus derived from Hebrew Wikipedia.

The source corpus contains 50,000 sentences. The cleaning pipeline retained **33,604** sentences, split deterministically into:

- **30,941 training sentences**
- **2,663 validation sentences**

The natural training split encoded to **1,141,490 model tokens** with the dependency-free `HybridHebrewTokenizer`.

### Natural Hebrew pretraining

Configuration:

- Main LM: **4,891,731 parameters**
- vocab: 4,096
- sequence length: 128
- batch size: 12
- **800 update steps**
- CPU GitHub Actions runner

Training loss moved from **8.4902** on step 1 to **2.3017** on step 800.

On a held-out validation set of 24,576 evaluated tokens:

- validation loss: **2.1799389155**
- perplexity: **8.8457659030**

This is a real natural-language training run, but the model is still only ~4.9M parameters and the corpus is modest. Generated Hebrew remains primitive; these numbers should not be interpreted as fluent conversational ability.

### The failed v0.6 experiment that changed the design

The first natural-Hebrew run allowed unrestricted joint fine-tuning of the whole Main LM during personalization.

It failed:

- personal benchmark accuracy reached only **62.5%**;
- held-out language loss worsened from about **2.18 to 3.19**;
- perplexity worsened from about **8.86 to 24.40**;
- the online World adaptation did not succeed.

That run demonstrated catastrophic forgetting rather than a successful milestone.

### Corrected v0.6 design

The correction freezes the ordinary language backbone after Hebrew pretraining and opens only the personal interface for the system-level personalization stage.

During the corrected integration run:

- **228,947 Main-LM personal-interface parameters** were trainable;
- **4,662,784 ordinary language parameters** were frozen;
- the shared Personal Learning System remained trainable;
- the per-user Personal Model remained part of the learning path.

The 32-case longitudinal benchmark reached:

**100% accuracy**

The ordinary language backbone was verified **bit-for-bit unchanged**.

### No-forgetting result

The exact same held-out Hebrew validation was evaluated before and after personalization.

Before personalization:

`loss = 2.1799389154960713, perplexity = 8.845765902995316`

After personalization:

`loss = 2.1799389154960713, perplexity = 8.845765902995316`

Therefore:

**language loss delta = 0.0**

This exact equality is expected from the architecture: unconditioned language inference bypasses the personal controller and modulation projections, while every parameter used by ordinary language inference is frozen during the personal-interface stage.

### Frozen per-user adaptation

After system training, the Main LM was fully frozen.

For one user, before online adaptation:

`Core=2, Policy=2, World=2, Fact=5`

Only that user's personal World latent was updated.

After adaptation:

`Core=2, Policy=2, World=1, Fact=5`

The Main LM remained **bit-for-bit unchanged** and the adaptation test passed.

## Reproducible v0.6 pipeline

The GitHub Actions workflow `.github/workflows/v06-natural-hebrew.yml` reproduces the full run:

`download natural corpus -> clean/split -> tokenizer -> Hebrew pretraining -> held-out evaluation -> personal-interface training -> held-out evaluation -> freeze adaptation -> tests -> artifacts`

The pipeline fails if the personal benchmark is below its gate, the frozen adaptation fails, or the protected language backbone changes.

## Local commands

Natural corpus preparation:

```bash
python prepare_natural_corpus.py corpus.txt \
  --train-out data/hebrew-natural-train.txt \
  --val-out data/hebrew-natural-val.txt
```

Tokenizer:

```bash
python train_natural_tokenizer.py data/hebrew-natural-train.txt \
  --vocab-size 4096 \
  --out hebrew-natural-hybrid-4096.json
```

Natural Hebrew training:

```bash
python train_hebrew.py data/hebrew-natural-train.txt \
  --tokenizer hebrew-natural-hybrid-4096.json \
  --seq-len 128 \
  --batch-size 12 \
  --max-steps 800 \
  --save hebrew-natural-v0.6.pt
```

Personal integration:

```bash
python train_longitudinal_v06.py \
  --base hebrew-natural-v0.6.pt \
  --tokenizer hebrew-natural-hybrid-4096.json \
  --personal-warmup 220 \
  --joint-steps 250 \
  --adapt-steps 60 \
  --save personal-natural-v0.6.pt
```

## Tests

```bash
pytest -q
```

Current result: **9/9 passing**.

## Important boundaries

- The Main LM **is trained** during model development; it is not permanently static by design.
- The language backbone is protected during the later personal-interface integration stage to prevent catastrophic forgetting.
- After release/freeze, the Main LM no longer changes per user.
- The per-user Personal Micro-Model can continue learning after that freeze.
- Personal influence is temporary runtime conditioning, not a permanent Main-LM weight update.
- Raw facts/events should remain in an auditable evidence/memory layer; the Personal Model stores learned aggregates.
- Real deployment still needs privacy, deletion/export, versioning, rollback and evidence/contradiction policies.

See [`PERSONAL_STATE_ABI.md`](PERSONAL_STATE_ABI.md) for the personal-model boundary.
