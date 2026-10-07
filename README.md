# Personal-Native LM 5M — Prototype v0.8

> **Current research status:** see [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for the canonical experiment ledger, active runs, conclusions, confounds and next decisions.

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


## v0.7: broader corpus, longer training, observable I/O

v0.7 adds a second natural-Hebrew source and makes actual input/output samples a required artifact of every training run.

### Corpus

The run combined:

- Hebrew Wikipedia sentence corpus: CC BY-SA 3.0;
- Knesset Meetings Corpus 2004-2005: Public Domain.

After decoding, cleaning, deduplication and deterministic source caps:

- **142,000 unique selected lines**;
- **133,536 training lines**;
- **8,464 validation lines**;
- **9,426,552 tokenizer tokens** in the training file.

The source mix was deliberately bounded so the much larger transcript corpus did not completely replace the encyclopedic source.

### Longer language run

The same ~4.89M Hebrew Main LM was trained for **3,200 steps** at sequence length 128 and batch size 12.

- first training loss: **8.4602**
- last training loss: **1.7999**
- held-out validation loss: **2.0168712377**
- held-out perplexity: **7.5147761650**

The held-out set is broader than v0.6, so this number is not a strict apples-to-apples comparison with the v0.6 validation set.

### Personal integration still passes

After language training, the protected personal-interface procedure was run again.

- longitudinal personalization accuracy: **100%**
- language backbone unchanged: **true**
- language loss delta after personalization: **0.0**
- frozen per-user adaptation: **success**
- tests: **9/9 passing**

The example frozen adaptation remained:

`Core=2, Policy=2, World=2, Fact=5`

to:

`Core=2, Policy=2, World=1, Fact=5`

with the Main LM unchanged.

### Real generation samples

v0.7 now stores `demo_language.json` and `demo_final.json`.

Representative greedy outputs from the final checkpoint:

| Input | Output |
|---|---|
| `ישראל היא מדינה` | ` שמועדרים ומועדים ומשינים ` |
| `המחשב יכול` | ` להתקורים במורים והמוצאים ` |
| `המחקר מראה כי` | ` המוצאים במורים ומוציבים ו` |

These outputs are **not fluent Hebrew**. The larger corpus and longer run improved quantitative language modeling, but the 4.9M model with the current hybrid tokenizer is still not a useful free-form generator.

This is an important negative result rather than something to hide. In particular, sampled outputs can contain the Unicode replacement character because the current hybrid tokenizer falls back to individual UTF-8 byte tokens; autoregressive generation can choose an invalid/incomplete byte sequence.

### v0.7 conclusion

The personal architecture continues to work strongly, while language generation is now the limiting subsystem.

The next language experiment should therefore change the tokenizer/objective before simply adding more training steps. A Unicode-safe BPE/Unigram tokenizer is the highest-priority candidate, followed by a controlled comparison at the same parameter budget.


## v0.8: Unicode-safe tokenizer A/B

v0.8 isolates the tokenizer as the experimental variable.

The following were held fixed relative to v0.7:

- Main-LM architecture and approximately **4.89M parameters**;
- vocabulary size: **4,096**;
- the exact same cleaned raw corpus and validation split;
- sequence length: 128;
- batch size: 12;
- training seed: 71;
- learning rate: 0.00028;
- **3,200 update steps**.

The v0.7 hybrid lexicon + UTF-8 byte fallback tokenizer was replaced with a
Unicode-safe **SentencePiece Unigram** tokenizer. Byte fallback is disabled.

### Tokenizer audit

On a 500,000-character sample:

- vocabulary: **4,096**;
- tokens: **156,210**;
- tokens per 100 characters: **31.242**;
- tokens per Hebrew word: **1.6505**;
- unknown tokens: **0**;
- Unicode replacement characters after decode: **0**;
- benchmark labels `1`, `2`, `5` are each exactly one token.

SentencePiece normalization means byte-for-byte/text-for-text round-trip is not
guaranteed for whitespace/normalization details, but the tokenizer no longer
constructs invalid partial UTF-8 sequences.

### Why per-token perplexity is not the A/B metric

v0.7 and v0.8 tokenize the same text at very different granularities. Therefore
their raw token-level cross-entropy/perplexity values are **not directly
comparable**.

For example, on the same 300,000-character validation prefix:

| Metric | v0.7 hybrid | v0.8 Unigram |
|---|---:|---:|
| encoded tokens | 262,928 | 92,391 |
| tokens / 100 chars | 87.643 | 30.797 |
| nats / character | 1.62783 | **1.43010** |
| bits / character | 2.34846 | **2.06320** |
| nats / UTF-8 byte | 0.92879 | **0.81597** |

The tokenizer-neutral result is therefore:

**v0.8 improves nats-per-character by 12.15% relative to v0.7.**

### Generation quality

The qualitative change is much larger than the token-level perplexity number
would suggest.

Representative v0.7 output:

`ישראל היא מדינה -> שמועדרים ומועדים ומשינים`

Representative v0.8 outputs:

- `ישראל היא מדינה` -> `יהודית, וכמובן, וכמובן ... 150 מיליון שקל ...`
- `המחשב יכול` -> `להיות שווי רחק ... 150 מיליון שקל ...`
- `בשנים האחרונות` -> parliamentary-style Hebrew beginning with
  `שאול יהלום ... אדוני היושב-ראש, חברי הכנסת ...`
- `המחקר מראה כי` -> a structured but domain-biased legislative continuation.

The text is still not reliably semantically correct and is strongly biased
toward the Knesset portion of the training corpus. It also repeats phrases.
However, it is now composed mostly of recognizable Hebrew words and
sentence-like structures rather than byte-fragment pseudo-words.

Across the ten fixed greedy/sampled demo continuations:

- v0.7 Unicode replacement characters: **6**
- v0.8 Unicode replacement characters: **0**
- v0.8 generated unknown tokens: **0**

### Personal model after tokenizer replacement

The tokenizer replacement did **not** break the personal architecture:

- longitudinal benchmark: **100%**;
- language backbone unchanged during personal-interface integration: **true**;
- ordinary-language validation loss delta after personalization: **0.0**;
- frozen per-user adaptation: **success**;
- tests: **9/9 passing**.

The same user-specific adaptation still changes only:

`World: 2 -> 1`

while Core, Policy, factual output and frozen Main-LM weights remain unchanged.

### v0.8 conclusion

The A/B experiment confirms that the old tokenizer was a material bottleneck.
A Unicode-safe subword model improves both character-normalized likelihood and
visible generation quality.

The remaining bottleneck is no longer malformed UTF-8 generation. The next
issues are model capacity, corpus balance/domain bias, repetition, and semantic
coherence.


## v0.9: balanced-source oversampling — negative result

v0.9 tested whether the remaining Knesset-domain bias could be fixed **without changing model size or tokenizer**, by reweighting the same unique training sources.

Held fixed relative to v0.8:

- Main LM: **4,891,731 parameters**
- SentencePiece Unigram vocabulary: 4,096
- same unique Wikipedia + Knesset source material
- same validation split
- sequence length 128
- batch size 12
- seed 71
- learning rate 0.00028
- **3,200 training steps**

### What was changed

Before weighting, the Unigram tokenizer produced:

- Wikipedia training source: **394,644 tokens**
- Knesset training source: **2,992,486 tokens**

To create an approximately 50/50 token mixture, v0.9 deterministically oversampled Wikipedia by about **7.579×** while keeping the Knesset source at 1×.

After weighting:

- Wikipedia: **2,990,940 tokens**
- Knesset: **2,992,486 tokens**
- Wikipedia share: **49.987%**

### Result

This did **not** improve the model.

On the same 300,000-character validation prefix:

| Metric | v0.8 | v0.9 balanced |
|---|---:|---:|
| nats / character | **1.43010** | 1.59708 |
| bits / character | **2.06320** | 2.30410 |

v0.9 worsened nats-per-character by **11.68%** relative to v0.8.

Generation also became substantially more repetitive. Representative greedy outputs included:

- `המחשב יכול` -> `להיות רחרחרחרחרחר...`
- `בשנים האחרונות` -> a long repetition of `1`
- `המחקר מראה כי` -> repeated `המחקרים`
- another prompt -> repeated `וכולנו`

Unicode safety remained intact: **0 replacement characters**.

### Personal architecture remained stable

The negative language result did not break personalization:

- longitudinal personal benchmark: **100%**
- protected language backbone unchanged: **true**
- frozen Main LM unchanged during per-user adaptation: **true**
- targeted World adaptation: **success**
- Core / Policy / Fact remained stable
- tests: **9/9 passing**

### v0.9 conclusion

**v0.8 remains the preferred language checkpoint.**

The experiment shows that domain balance cannot be fixed by aggressively repeating a small general-domain source. The 7.58× Wikipedia repetition reduced effective data diversity and caused severe repetition/overfitting.

The next data step should add **new, unique, diverse Hebrew text** rather than duplicating existing Wikipedia sentences. Until that additional data exists, the v0.8 checkpoint is the language baseline to beat.


## v0.10: unique-data balance — partial recovery, still below v0.8

v0.10 tested the same ~5M Main LM and the exact v0.8 Unicode-safe Unigram tokenizer, but replaced v0.9's repeated Wikipedia sentences with **genuinely new, deduplicated Wikipedia sentences**.

### Data mix

- old 50K Wikipedia source excluded from the new stream;
- **86,234 unique new Wikipedia lines** selected;
- **103,452 Knesset training lines**;
- token mass:
  - new Wikipedia: **2,942,285**
  - Knesset: **2,992,486**
- Wikipedia share: **49.58%**
- oversampling factor: **1.0**

So unlike v0.9, no source sentence was intentionally repeated to reach 50/50.

### Result

On the same 300,000-character validation prefix:

| Metric | v0.8 | v0.9 | v0.10 |
|---|---:|---:|---:|
| nats / character | **1.43010** | 1.59708 | 1.52040 |
| bits / character | **2.06320** | 2.30410 | 2.19347 |

v0.10 is **4.80% better than v0.9** in nats-per-character, confirming that unique data is materially better than aggressive repetition.

However, it is still **6.31% worse than v0.8**, so a strict 50/50 domain balance is not optimal for this 5M model and validation distribution.

### Generation

The severe collapse seen in v0.9 was reduced, but generation is still unstable and domain-biased. Representative outputs include:

- `ישראל היא מדינה` -> `, ואנשים שופגופם, 200 מטר חופשיים, 2005, 2022...`
- `המחשב יכול` -> `להיות שופגר את ההופעות של ההופעות של האלבום של 1968...`
- `המחקר מראה כי` -> `1944 – 1948 – רוסיה רוסיה, סופר, סופר, חתן פרס נובל...`

Unicode safety remained intact: **0 replacement characters**.

### Personal architecture

The personal subsystem again remained stable:

- longitudinal benchmark: **100%**
- frozen language backbone unchanged: **true**
- frozen Main LM unchanged during per-user adaptation: **true**
- targeted World adaptation: **success**
- language metric before/after personal integration: unchanged
- tests: **9/9 passing**

### v0.10 conclusion

v0.10 confirms two separate effects:

1. v0.9's aggressive duplication caused real damage;
2. even with unique data, **50% Wikipedia / 50% Knesset is not the best training mix** for this model.

The next controlled experiment is therefore a **source-ratio sweep** using unique data only, while holding tokenizer, model size, seed, optimizer, validation set and training budget fixed.


## v0.11: unique-data source-ratio sweep

v0.11 tested four source ratios using **unique data only** and no oversampling:

- 12% Wikipedia / 88% Knesset
- 20% / 80%
- 30% / 70%
- 40% / 60%

All candidates used the same ~4.89M Main LM, the same Unicode-safe 4,096-token Unigram tokenizer family, the same seed, LR and external validation set.

### 800-step screening

| Requested Wikipedia share | Actual share | nats / char | bits / char |
|---|---:|---:|---:|
| 12% | 11.83% | **1.72968** | **2.49540** |
| 20% | 19.74% | 1.73619 | 2.50479 |
| 30% | 29.64% | 1.74249 | 2.51388 |
| 40% | 39.57% | 1.74627 | 2.51933 |

The **12%** mixture won the screening and was retrained from scratch for the full 3,200-step budget.

### Full winner result

The 12% mixture reached:

- nats / character: **1.48902**
- bits / character: **2.14820**
- Unicode replacement characters: **0**

Comparison:

| Version | bits / char |
|---|---:|
| **v0.8 baseline** | **2.06320** |
| v0.11 12% unique Wikipedia | 2.14820 |
| v0.10 50% unique Wikipedia | 2.19347 |
| v0.9 50% oversampled Wikipedia | 2.30410 |

So v0.11 improves on v0.10 by about **2.06% in nats/character**, but remains about **4.12% worse than v0.8**.

### Personal architecture

The personal subsystem remained fully intact:

- longitudinal benchmark: **100%**
- protected language backbone unchanged: **true**
- frozen Main LM unchanged during per-user adaptation: **true**
- targeted World adaptation: **success**
- language score before/after personal integration: unchanged
- tests: **9/9 passing**

### v0.11 conclusion

The sweep shows that manually balancing Wikipedia and Knesset is reaching diminishing returns. Lower Wikipedia shares are better in this setup, but none beat the original v0.8 baseline.

Rather than continue hand-tuning two narrow sources, the next language stage moves to a corpus designed explicitly for language-model pretraining: **FineWeb2 Hebrew (`heb_Hebr`)**, streamed and sampled at controlled token budgets.


## v0.12: FineWeb2 Hebrew with the frozen v0.8 tokenizer — negative result

v0.12 replaced the hand-built Wikipedia/Knesset training mixture with a streamed sample from **FineWeb2 Hebrew (`heb_Hebr`)**.

### Data

The sampler targeted:

- 10,000,000 training tokens
- 500,000 held-out FineWeb2 validation tokens

It produced:

- **10,000,130 sampled training tokens**
- **500,057 validation tokens**
- **12,020 FineWeb2 documents scanned**
- **109,343 accepted chunks**
- **2,621 duplicate chunks removed**
- **8,206 chunks rejected because the frozen v0.8 tokenizer emitted `<unk>`**

The training script ultimately encoded **9,883,710 model tokens** and completed **6,385 steps** in one epoch.

### Result

The model did not beat v0.8.

| Evaluation | nats / char | bits / char |
|---|---:|---:|
| FineWeb2 held-out | 1.74448 | 2.51675 |
| external Wikipedia/Knesset validation | 1.79137 | 2.58440 |
| **v0.8 external baseline** | **1.43010** | **2.06320** |

On the external validation distribution, v0.12 is about **25.26% worse in nats/character** than v0.8.

Generation remained unstable and repetitive, for example:

- `ישראל היא מדינה` -> repeated `ראייה`
- `המחשב יכול` -> repeated `בנוסף`
- other prompts still mixed malformed words with domain fragments

Unicode safety remained intact: **0 replacement characters**.

### Why this is not yet a clean capacity test

v0.12 deliberately reused the frozen tokenizer from v0.8. That tokenizer was learned on Wikipedia/Knesset rather than FineWeb2.

On the tokenizer audit:

- FineWeb2: **37.28 tokens / 100 characters**
- old external validation: **30.80 tokens / 100 characters**
- **8,206 candidate chunks** were rejected because of tokenizer unknowns

Therefore v0.12 confounds two questions:

1. can a ~5M model learn broad FineWeb2 Hebrew?
2. can it do so through a tokenizer optimized for a different corpus?

The next experiment isolates this by keeping the 5M architecture, 4,096 vocabulary size and 10M-token budget fixed while training a **new Unicode-safe Unigram tokenizer on FineWeb2 training text itself**.

### Personal architecture

The language regression did not affect the personal system:

- longitudinal benchmark: **100%**
- language backbone unchanged during personal integration: **true**
- frozen Main LM unchanged during per-user adaptation: **true**
- targeted World adaptation: **success**
- language score before/after personalization: unchanged
- tests: **9/9 passing**


## v0.13: FineWeb2-native tokenizer — tokenizer mismatch was not the main bottleneck

v0.13 repeated the v0.12 FineWeb2 experiment while training a new Unicode-safe 4,096-token Unigram tokenizer from **FineWeb2 training text only**.

The tokenizer corpus contained **8,000,044 train characters** across **31,516 accepted train chunks**, while **1,671 validation chunks were explicitly skipped** to prevent tokenizer leakage.

### Result

| Evaluation | v0.12 | v0.13 |
|---|---:|---:|
| FineWeb2 held-out nats/char | 1.74448 | **1.73508** |
| FineWeb2 held-out bits/char | 2.51675 | **2.50319** |
| external nats/char | **1.79137** | 1.84177 |
| external bits/char | **2.58440** | 2.65711 |

The FineWeb2-native tokenizer improved FineWeb2 held-out nats/character by only **0.54%** relative to v0.12, while external generalization became **2.81% worse**.

Compared with the best v0.8 external baseline, v0.13 remains about **28.8% worse in nats/character**.

Tokenizer efficiency improved only modestly:

- v0.12 FineWeb2: ~36.99 tokens / 100 chars
- v0.13 FineWeb2: ~35.65 tokens / 100 chars

So tokenizer-domain mismatch was real, but **not the dominant remaining bottleneck**.

### Generation

Generation remained repetitive and semantically weak, for example:

- `ישראל היא מדינה` -> repeated `ארץ-ישראל`
- `המחשב יכול` -> repeated `ההוראה`
- `בשנים האחרונות` -> repeated `טכנולוגיות`

Unicode safety remained intact: **0 replacement characters**.

### Personal architecture

The personal system remained stable again:

- longitudinal benchmark: **100%**
- protected language backbone unchanged: **true**
- frozen Main LM unchanged during per-user adaptation: **true**
- targeted World adaptation: **success**
- language metrics unchanged after personal integration
- tests: **9/9 passing**

### v0.13 conclusion

After isolating corpus choice and tokenizer adaptation, the next controlled variable is **model capacity**.

v0.14 therefore keeps the FineWeb2-native tokenizer, data budget, seed, optimizer, sequence length and step budget fixed while increasing the Main LM from ~4.9M to ~10M parameters.


## v0.14: 10M capacity A/B — capacity alone is not the main bottleneck

v0.14 kept the FineWeb2-native tokenizer, FineWeb2 data budget, seed, learning rate,
sequence length and **6,385-step training budget** fixed, while increasing the Main LM
from **4,891,731** to **9,771,445 parameters**.

### Result

| Evaluation | v0.13 ~5M | v0.14 ~10M |
|---|---:|---:|
| FineWeb2 held-out nats/char | 1.73508 | **1.71686** |
| FineWeb2 held-out bits/char | 2.50319 | **2.47690** |
| external nats/char | 1.84177 | **1.82921** |
| external bits/char | 2.65711 | **2.63899** |

The near-doubling in parameter count improved:

- FineWeb2 held-out nats/char by only **1.05%**
- external nats/char by only **0.68%**

The external score is still about **27.9% worse in nats/char** than the v0.8 baseline.

Generation also remained repetitive and weakly coherent, including repeated words such
as `תרבות`, `יראה`, `הוראה`, and `קוראים לעצמי`.

### Personal architecture

The personal subsystem remained stable:

- longitudinal benchmark: **100%**
- protected language backbone unchanged: **true**
- frozen Main LM unchanged during per-user adaptation: **true**
- targeted World adaptation: **success**
- language metrics unchanged after personal integration
- Unicode replacement characters: **0**
- tests: **9/9 passing**

### v0.14 conclusion

Model capacity helps slightly, but **capacity alone is not the dominant bottleneck** at
this stage. Corpus choice, tokenizer adaptation and a near-2x capacity increase have
all produced only limited improvements relative to the best v0.8 baseline.

The next research direction should therefore examine the **learning objective and
knowledge organization**, including explicit structured knowledge, stable identity
anchors and relational/world-model representations, rather than continuing to scale
parameter count blindly.
