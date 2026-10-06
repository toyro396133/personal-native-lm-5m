# Personal-Native LM 5M — Prototype v0.4

A research prototype for training **both** a shared language model and a personal model, then freezing the shared model while each user's personal model can continue learning and influence the shared model only at runtime.

## The lifecycle

### 1. Joint/system training

During development, the shared LM and the personal-learning stack are trainable together:

`history -> Personal Learner -> Personal Model/State -> request-conditioned controller -> Main LM`

Gradients can update both sides so they learn a compatible interface.

### 2. Freeze the shared model

After the shared LM is closed/frozen, its parameters stop changing:

`frozen Main LM + changing per-user Personal Micro-Model -> temporary runtime conditioning`

The user's personal model may keep learning. Its effect on the Main LM is ephemeral: it changes activations for the current request, **not the Main LM weights**.

## v0.4 architecture

### Main LM

- decoder-only Transformer
- **5,231,059 trainable parameters** in the byte-level bootstrap profile
- 6 Transformer blocks
- 8 attention heads
- request-conditioned activation modulation inside every block

### Shared personal-learning system

- **91,476 parameters**
- `PersonalHistoryEncoder`: converts a sequence of interactions into an initial personal latent
- `PersonalStateDecoder`: converts the compact latent into the canonical 228-dimensional Personal State ABI

### Per-user Personal Micro-Model

- **116 trainable parameters per user** in v0.4
- seeded from the user's interaction history
- can continue training after the Main LM is frozen
- produces canonical Core / Policy / World / Routing state through the shared decoder

This 116-parameter micro-model is deliberately tiny: v0.4 proves the lifecycle and separation of weights. Later versions can increase its capacity without changing the Main LM contract.

## Competitive request routing

The controller now performs a five-way request-dependent route:

1. Core
2. Policy
3. World
4. Routing
5. None

`None` is important: factual or otherwise irrelevant requests can explicitly ignore personal state.

The selected personal signal modulates hidden activations only for the current forward pass.

## Training experiment

The synthetic bootstrap uses four profiles with two independent preferences and five request variants:

- two Core requests;
- two Policy requests;
- one factual request that must ignore personalization.

Training is performed in three phases:

1. **Personal warm-up** — learn to infer the canonical personal state from interaction history.
2. **Joint training** — train the Main LM + personal-learning system + router together.
3. **Freeze/adapt** — freeze the Main LM and shared personal stack, then update only one user's 116 parameters.

### Latest v0.4 result

The final run reached:

- joint evaluation: **100%** on the 20-case synthetic benchmark;
- all **111/111** Main-LM parameter tensors changed during joint training, confirming the Main LM really was trained;
- after freeze, the Main LM remained **bit-for-bit unchanged**;
- the shared personal-learning system also remained unchanged;
- only the new user's **116 personal parameters** were trained;
- before adaptation: `Core=2, Policy=2, Fact=4`;
- after learning a changed Core preference: `Core=1, Policy=2, Fact=4`.

That last test is the key v0.4 milestone: the personal model changed one user-specific behavior while the frozen Main LM stayed untouched and unrelated behavior remained stable.

> These are architectural sanity checks on a tiny synthetic curriculum, not a claim of general language capability.

## Plasticity

Persistent state is intentionally kept away from hard ±1 saturation. The state decoder uses a bounded scale so an old preference can still be revised later. This was added after an earlier experiment showed that saturated `tanh` state was too difficult to update online.

## Run the v0.4 experiment

```bash
python -m pip install -r requirements.txt
python train_joint_personal.py \
  --personal-warmup 220 \
  --joint-steps 30 \
  --adapt-steps 40
```

The script can also continue from an existing checkpoint with `--base`.

## Hebrew path

The real Hebrew profile remains around five million parameters and uses a 4,096-token ByteLevel BPE tokenizer.

Train the tokenizer:

```bash
python train_tokenizer.py corpus.txt --vocab-size 4096
```

Start general Hebrew pretraining:

```bash
python train_hebrew.py corpus.txt --tokenizer hebrew-bpe-4096.json
```

The Hebrew pretraining stage is not yet complete; v0.4 focused on proving the two-model training/freeze lifecycle first.

## Tests

```bash
pytest -q
```

Tests cover runtime conditioning, request dependence, frozen-LM invariants, Personal Micro-Model size/routing behavior, and parameter isolation.

## Current limitations

- Training data is still synthetic and tiny.
- The Main LM has not yet undergone large-scale Hebrew pretraining.
- The per-user micro-model is intentionally only 116 trainable scalars in v0.4.
- World selection is still simplified to one canonical World slot in the 228-d ABI.
- Long-term evidence consolidation and a learned write/update policy need a larger longitudinal curriculum.
- Real deployment will need privacy, deletion, export, versioning and rollback rules for personal models.

See [`PERSONAL_STATE_ABI.md`](PERSONAL_STATE_ABI.md) for the model boundary.
