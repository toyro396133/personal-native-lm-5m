# Personal State ABI v0.1

The personal model must survive replacement of the language model. Therefore
the stored state is independent of any LM vocabulary, token IDs or hidden size.

## Canonical slots

- `core[64]`: relatively stable user-level latent state.
- `policies[64]`: preferences about how to act/work/respond.
- `worlds[name][64]`: context-specific state for a project/domain.
- `routing[32]`: context-selection/routing latent state.
- confidence + evidence count for every slot.

## Boundary

Explicit factual memory is **not** stored here. Facts/events remain in an
auditable memory layer. The personal state stores learned aggregates.

## Read contract

`T_read(state, scope) -> K x d_model personal prefix tokens`

In v0.1 K=4 and d_model=256.

The prefix is inserted before ordinary tokens, so all text tokens can attend to
the user state without converting it to prompt prose.

## Write contract

The LM cannot mutate the state.

A future `T_write` may create an `UpdateProposal`, but an independent
`EvidenceConsolidator` gates and bounds every persistent update.

## Portability

A different LM can use the same ABI by training a new reader for its own hidden
dimension. The per-user state need not be retrained from scratch.
