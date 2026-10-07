# Hugging Face permanent checkpoint archive

The checkpoint archive is designed to preserve the code-named V1..V7 model
checkpoints independently of GitHub Actions artifact retention.

## Current scope

The manifests currently index 56 checkpoints:

- V1..V7
- 25M, 50M, 75M, 100M
- 110M, 120M, 130M, 140M

Checkpoint binaries remain in GitHub Actions until they are synchronized.

## Target layout on Hugging Face

```
checkpoints/
  V1/
    025M/
      checkpoint.pt
      metadata.json
    ...
  V7/
    140M/
      checkpoint.pt
      metadata.json
archive-index.json
README.md
```

Each metadata file records:

- code-name variant
- token count
- source GitHub repository
- source workflow run
- immutable Actions artifact ID
- original checkpoint filename
- SHA-256
- byte size
- archive timestamp

## Authentication

Create a Hugging Face access token with permission to create/write the target
model repository and save it in this GitHub repository as an Actions secret:

`HF_TOKEN`

Do not commit the token to the repository.

## Workflow

Workflow:

`.github/workflows/archive-checkpoints-to-hf.yml`

Script:

`scripts/archive_checkpoints_to_hf.py`

The workflow derives the default Hugging Face namespace from the token owner,
creates the target model repository if it does not exist, and defaults to
**private** visibility.

Recommended first run:

- repo_name: `personal-native-lm-checkpoints`
- namespace: blank
- visibility: `private`
- force: `false`
- limit: `1`

After verifying the canary, run again with:

- limit: `0`

The archiver is resumable. A remote `archive-index.json` records successfully
archived artifact IDs, so a repeated run skips checkpoints already stored.

## Future checkpoints

When a new checkpoint is added to any
`models/code-variants/V*/manifest.json`, the same archiver can sync it without
changing the archive layout or the V1..V7 identities.
