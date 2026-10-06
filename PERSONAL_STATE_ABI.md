# Personal State ABI v0.3

The system contains two separately trainable levels:

1. a shared Main Language Model;
2. a persistent Personal Model for each user.

During system training, both sides may be trained together. After the Main LM is frozen, its parameters must remain fixed; the Personal Model may continue learning and affect the Main LM only through request-specific runtime conditioning.

## Canonical state contract

The current canonical state is 228 dimensions:

- `core[64]`
- `policies[64]`
- `world[64]`
- `routing[32]`
- four confidence values

The ABI is independent of Main-LM vocabulary, token IDs and hidden width.

## Personal model v0.4

The per-user trainable micro-model contains a compact 116-dimensional latent:

- Core latent: 32
- Policy latent: 32
- World latent: 32
- Routing latent: 16
- Confidence latent: 4

A shared `PersonalStateDecoder` maps this personal latent into the canonical 228-dimensional ABI.

A shared `PersonalHistoryEncoder` can initialize the user's latent from a sequence of interactions. After initialization, the user's latent is persistent and trainable independently.

## Read / runtime contract

For persistent user model `P_u` and request `q`:

`S_u = PersonalStateDecoder(P_u)`

`route(q) -> {core, policy, world, routing, none}`

`C_u(q) = PersonalController(S_u, q, route(q))`

`C_u(q)` is ephemeral. It modulates hidden activations inside the Transformer for the current request only.

No inference operation writes into Main-LM weights.

## Training lifecycle

### Joint phase

The Main LM, Personal History Encoder, Personal State Decoder and request router may all receive gradients.

### Frozen phase

The Main LM is frozen. In the strongest v0.4 test, the shared personal-learning stack is frozen too. Only the new user's 116 personal parameters receive gradients.

A routed update mask can restrict an online learning event to one personal slot (for example Core) so unrelated slots remain stable.

## Plasticity rule

Canonical state values should not be pushed unnecessarily to hard saturation. The v0.4 decoder uses a bounded scale below 1 so old beliefs/preferences remain revisable.

## Write / persistence boundary

Raw facts and events should remain in an auditable memory/evidence layer. The Personal Model stores learned aggregates.

Persistent updates should eventually pass an evidence-aware write policy with support for confidence, contradiction, decay, rollback and user deletion/export. The existing `EvidenceConsolidator` is an early deterministic version of that boundary.

## Portability

A future larger Main LM can learn its own controller against the same canonical ABI. The user's Personal Model does not need to be rebuilt merely because the shared language model is replaced.
