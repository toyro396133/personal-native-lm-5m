# Personal-Native LM 5M — Prototype v0.3

A research prototype for a language model whose **runtime reasoning can be
conditioned by a persistent personal model without permanently changing the
language-model weights**.

## Core idea

For user `u` and current request `q`:

`PersonalState S_u + q -> ContextualPersonalController -> C_u(q) -> LM layers`

`C_u(q)` is temporary. It modulates activations inside the Transformer during
this request only. When inference ends, the LM weights remain unchanged.

This separates three things:

1. **Shared LM weights** — global language capability.
2. **Persistent PersonalState** — durable user-specific state.
3. **Ephemeral request conditioning** — the part of that state relevant to the
   current request.

## Implemented

- Canonical `PersonalState` with Core, Policies, Worlds, Routing and
  confidence/evidence metadata.
- `ContextualPersonalController` that fuses the personal state with the current
  query.
- Layer-wise gated activation modulation in every Transformer block.
- A safety boundary: inference cannot mutate LM weights, and persistent personal
  updates must pass `EvidenceConsolidator`.
- ~5M decoder-only Transformer profiles.
- Dependency-free byte tokenizer for architecture experiments.
- Hebrew-aware ByteLevel BPE training path with a default 4,096-token vocab.
- General Hebrew pretraining entry point.
- Request-specific personalization bootstrap training and tests.

## Parameter profiles

### Byte bootstrap

- ~5.23M trainable parameters
- `d_model=256`
- 6 Transformer blocks
- 8 attention heads
- `controller_dim=128`

### Hebrew BPE profile

- ~4.89M trainable parameters at vocab 4,096
- `d_model=224`
- 6 Transformer blocks
- 8 attention heads
- FFN 896
- `controller_dim=112`
- context window 512

## Bootstrap conditioning experiment

The first training experiment deliberately tests the architecture before
expensive language pretraining. Four synthetic user profiles contain two
independent preferences. Three query families are used:

- one query should use preference A;
- another query should use preference B;
- a factual query must ignore both preferences.

On the deterministic 12-case bootstrap evaluation, the byte-level 5.23M model
improved from **33.3% before training to 100% after 100 balanced update steps**.
This is an architectural sanity check, not a claim of general language ability.

Run it:

```bash
python -m pip install -r requirements.txt
python train_personal_conditioning.py --steps 100 --batch-size 12 --lr 0.0005
```

## Hebrew training

Train a tokenizer:

```bash
python train_tokenizer.py corpus.txt --vocab-size 4096
```

Start general-language pretraining with neutral personal state:

```bash
python train_hebrew.py corpus.txt --tokenizer hebrew-bpe-4096.json
```

The intended full training mix is:

- **general language:** neutral personal state;
- **personal contrast:** same request, different relevant PersonalState,
  different correct continuation;
- **personal irrelevance:** different PersonalState, same correct continuation;
- **context selection:** one user has several independent preferences and the
  current query determines which one should affect computation.

## Tests

```bash
pytest -q
```

Current tests verify that:

- tensor shapes and loss are valid;
- different PersonalState can change runtime computation;
- a different current query changes the conditioning for the same user;
- inference does **not** mutate any LM weight.

## Personal-state update boundary

The LM never writes directly to persistent personal state. A future learned
writer (`T_write`) may propose an update, but an independent consolidator must
approve it based on evidence and confidence.

See [`PERSONAL_STATE_ABI.md`](PERSONAL_STATE_ABI.md).

## Current limitations

- The model has not yet undergone large-scale Hebrew pretraining.
- The 100% result above is only a tiny synthetic architectural benchmark.
- `PersonalState.flatten(scope)` still selects one named world externally. A
  future ABI revision should expose a bank of worlds so the controller itself
  can choose the relevant world from the request.
- A learned `T_write` and real longitudinal user-learning dataset are not yet
  implemented.

## Files

- `model.py` — Transformer + contextual personal controller.
- `personal_state.py` — persistent PersonalState and update consolidator.
- `train_personal_conditioning.py` — request-specific conditioning bootstrap.
- `train_tokenizer.py` / `bpe_tokenizer.py` — Hebrew BPE path.
- `train_hebrew.py` — general Hebrew pretraining entry point.
- `test_prototype.py` — architecture invariants.
