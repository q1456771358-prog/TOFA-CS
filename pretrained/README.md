# TOFA-CS pretrained model

The released seed-0 bundle contains:

```text
pretrained/seed_0/
├── trained_model.pt
├── frozen_reference.pt
└── model_manifest.json
```

- `trained_model.pt` is the source-trained model after 10 epochs.
- `frozen_reference.pt` is the corresponding epoch-0 reference state used by the preserved representation branch.
- `model_manifest.json` records model hashes, training settings, and fixed-checkpoint verification results.

Both model files were produced with joint source training on Texas, Photo, and DBLP using seed 0. The same `trained_model.pt` is used for every unseen target; no target-specific checkpoint is included.

The fixed epoch-10 model produced the following verification results:

| Target | F1 (%) | Jaccard (%) |
|---|---:|---:|
| Wisconsin | 36.98 | 22.68 |
| Cora | 59.39 | 42.24 |
| Citeseer | 47.76 | 31.37 |

These are fixed-checkpoint results. The paper's main table instead reports target-wise best epochs within each independently trained seed, followed by the mean and sample standard deviation across three seeds.

Run the model with:

```bash
python run_experiment.py pretrained \
  --data-root data \
  --model-root pretrained/seed_0 \
  --targets wisconsin,cora \
  --device cuda:0
```

