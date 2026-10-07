# Code-named model variants

This directory is a blind-safe index of the seven model variants used in the
v0.17/v0.18 comparison line.

- Variant identities are intentionally represented only as **V1..V7** here.
- Technical implementation names are deliberately omitted from this directory.
- Checkpoint binaries remain in GitHub Actions artifacts; they are not copied
  into Git history.
- Each variant directory contains a `manifest.json` with the exact artifact ID
  and source run for every preserved checkpoint.
- Current complete coverage: **25M, 50M, 75M, 100M, 110M, 120M, 130M, 140M**
  for all seven variants.
- The mapping is stable and matches the V1..V7 identifiers used by the blind
  four-component research.

This layout is append-only: when later checkpoints become available, add them
to the same V1..V7 manifests rather than creating new identities.
