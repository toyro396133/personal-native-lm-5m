# Personal-Native LM 5M — Prototype v0.5

A research prototype for training **both** a shared language model and a persistent personal model, then freezing the shared model while each user's personal model can keep learning and influence the shared model only at runtime.

## Core lifecycle

### 1. Joint/system training

During development, both sides are trainable:

`history -> Personal Learner -> Personal Model/State -> request-conditioned controller -> Main LM`

Gradients can update the Main LM and the shared personal-learning stack so they learn a compatible interface.

### 2. Freeze the Main LM

After the Main LM is closed/frozen:

`frozen Main LM + changing per-user Personal Micro-Model -> temporary runtime conditioning`

The user's personal model may continue learning. Its effect on the Main LM is ephemeral: it changes activations for the current request, **not the Main LM weights**.

## Architecture

### Main LM

- decoder-only Transformer
- ~5.23M parameters in the byte bootstrap profile
- ~4.89M parameters in the 4,096-vocabulary Hebrew profile
- 6 Transformer blocks
- 8 attention heads
- request-conditioned activation modulation inside every block

### Shared personal-learning system

- ~91K parameters
- `PersonalHistoryEncoder`: turns a sequence of past interactions into an initial personal latent
- `PersonalStateDecoder`: turns the compact latent into the canonical 228-dimensional Personal State ABI

### Per-user Personal Micro-Model

- **116 trainable parameters per user** in v0.5
- initialized from interaction history
- may continue training after the Main LM is frozen
- decoded into Core / Policy / World / Routing state through the shared decoder

The 116-parameter model is deliberately tiny. v0.5 is proving the lifecycle and weight separation before scaling personal-model capacity.

## Request routing

The controller performs a five-way request-dependent route:

1. Core
2. Policy
3. World
4. Routing
5. None

`None` is important: a factual or irrelevant request can explicitly ignore personal state.

## v0.5 milestone 1: longitudinal personal learning

The earlier prototype used very short histories. v0.5 adds eight synthetic users with longer histories containing:

- about 14–17 events per user;
- repeated evidence;
- irrelevant/noise events;
- a contradictory event;
- old -> new Core preference changes for some users;
- separate Core, Policy and World preferences.

The curriculum evaluates four request families: Core, Policy, World and a factual question that should ignore personalization.

Latest run:

- initial longitudinal evaluation: **37.5%**;
- after personal warm-up + first joint stage: **96.9%**;
- after focused continuation training: **100%** on the 32-case longitudinal benchmark;
- request-routing loss in the continuation stage fell to about **0.07**.

### Frozen Main-LM adaptation test

A user initially produced:

`Core=2, Policy=2, World=2, Fact=5`

Then only the user's **World** latent was trained while the Main LM and shared personal stack were frozen.

After adaptation:

`Core=2, Policy=2, World=1, Fact=5`

The Main LM remained **bit-for-bit unchanged**. This verifies that a user-specific model can learn a new context-specific preference without permanently modifying the shared model or contaminating the tested unrelated behaviors.

Run:

```bash
python train_longitudinal.py \
  --personal-warmup 180 \
  --joint-steps 40 \
  --adapt-steps 30
```

## v0.5 milestone 2: Hebrew language bootstrap

v0.5 also starts actual Hebrew-language training of the shared model.

Because the optional external `tokenizers` package is not always available in filtered/offline environments, the repository now includes a dependency-free `HybridHebrewTokenizer`:

- frequent Hebrew words/punctuation can receive whole-token IDs;
- every unknown item falls back losslessly to UTF-8 bytes;
- no `UNK` token is required;
- the model can still reserve a 4,096-token vocabulary.

A clean generated Hebrew bootstrap corpus was used for the first run:

- **20,000 lines**;
- about **1.7 MB** of UTF-8 text;
- **368,186 model tokens** with the hybrid tokenizer;
- Main LM: **4,891,731 parameters**;
- sequence length: 64;
- batch size: 2;
- **300 update steps** on CPU.

Training loss fell from **8.2834** on step 1 to **1.8999** on step 300.

This is **not** broad Hebrew pretraining yet. The corpus is intentionally small, generated and neutral, and text generation after 300 steps is still primitive. The milestone proves that the full Hebrew training path runs and produces a usable checkpoint; a much larger licensed natural-language corpus is still required for real fluency.

Generate the bootstrap corpus:

```bash
python generate_hebrew_bootstrap_corpus.py --lines 20000 --out hebrew_bootstrap_corpus.txt
```

Train the dependency-free tokenizer:

```bash
python train_hybrid_tokenizer.py hebrew_bootstrap_corpus.txt \
  --vocab-size 4096 \
  --out hebrew-hybrid-4096.json
```

Run Hebrew bootstrap training:

```bash
python train_hebrew.py hebrew_bootstrap_corpus.txt \
  --tokenizer hebrew-hybrid-4096.json \
  --seq-len 64 \
  --batch-size 2 \
  --max-steps 300 \
  --save hebrew-bootstrap-v0.5.pt
```

The existing optional BPE path remains available through `bpe_tokenizer.py` when the external `tokenizers` package is installed.

## Tests

```bash
pytest -q
```

Current result: **9/9 passing**.

The tests cover:

- tensor shapes and language-model loss;
- runtime personal-state influence;
- request-dependent conditioning;
- inference not mutating Main-LM weights;
- 116-parameter personal micro-model isolation;
- five-way competitive routing including `None`;
- gradient flow into only the personal micro-model when the Main LM is frozen;
- dependency-free Hebrew tokenizer round-trip without unknown tokens;
- longitudinal histories with noise/change and explicit World state.

## Important boundaries

- Raw facts/events should remain in an auditable memory/evidence layer.
- The Personal Model stores learned aggregates, not the only copy of user history.
- The Main LM may be trained during system development, but after release/freeze its weights do not change per user.
- Per-user adaptation should be routed to the relevant personal slot whenever evidence identifies that slot.
- Real deployment still needs privacy, deletion/export, versioning, rollback and evidence/contradiction policies.

See [`PERSONAL_STATE_ABI.md`](PERSONAL_STATE_ABI.md) for the personal-model boundary.