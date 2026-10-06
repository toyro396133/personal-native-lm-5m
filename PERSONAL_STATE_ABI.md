# Personal State ABI v0.2

The persistent personal model is separate from the language model. Its job is
not to permanently rewrite LM weights. Instead, it supplies durable user state
that is converted into **temporary request-specific conditioning** at runtime.

## Canonical slots

- `core[64]`: relatively stable user-level latent state.
- `policies[64]`: preferences about how to act/work/respond.
- `worlds[name][64]`: context-specific latent state.
- `routing[32]`: routing/context-selection state.
- confidence + evidence count for every slot.

Explicit factual memory remains outside this latent state in an auditable memory
layer. The personal state stores learned aggregates rather than raw chat logs.

## Runtime read contract

For a user request `q` and persistent user state `S_u`:

`C_u(q) = PersonalController(S_u, q)`

`C_u(q)` is ephemeral. It is used to modulate hidden activations inside the
Transformer layers for this request only. When the request ends, that runtime
conditioning disappears. The language-model weights are unchanged.

The v0.3 prototype uses a query encoder + state encoder + fusion controller and
injects a gated activation delta into each Transformer block.

## Write contract

The LM cannot directly mutate persistent personal state. A writer may create an
`UpdateProposal`, but an independent `EvidenceConsolidator` gates and bounds
persistent changes using confidence and evidence thresholds.

## Portability

The canonical personal state is independent of LM vocabulary, token IDs and
hidden width. A future larger LM can learn a new controller for the same state,
so the user's long-term model does not need to be rebuilt from scratch.

## Current limitation

`PersonalState.flatten(scope)` can still select one named world externally. A
future ABI revision should expose multiple world slots to the controller so that
world selection itself is request-conditioned rather than caller-selected.
