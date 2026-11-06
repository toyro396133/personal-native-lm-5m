# Next validation gates

1. Confirm the CPU architecture-inspection job passes on the pinned upstream revision.
2. Confirm PEFT can inject LoRA into the reviewed module suffixes without allocating the base checkpoint.
3. If injection passes, prepare a small approved dataset and run one short training smoke test on an already-available CUDA machine.
4. Save/reload the adapter and compare it with the unmodified base on the held-out English and Hebrew slices.
5. Only then expand the pilot toward 500 curated examples.

No paid GPU, external dataset upload, or full training run is authorized by this scaffold.
