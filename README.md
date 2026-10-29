# Target-Shift-Aware One-for-All Community Search on Unseen Graphs

This repository provides the implementation and main-experiment pipeline for **TOFA-CS**, a target-shift-aware one-for-all community search framework. A single model is trained jointly on the three source graphs **Texas, Photo, and DBLP**, then applied to unseen target graphs without target-label training.

The public entry point supports two paper protocols:

- `pretrained`: evaluate the released source-trained model on one or more unseen targets;
- `train`: retrain the shared model on the three fixed source graphs and evaluate selected targets.

## Repository layout

```text
.
├── run_experiment.py        # user-facing experiment entry point
├── configs/                 # paper method and training settings
├── data/                    # bundled Wisconsin and Cora target data
├── pretrained/              # verified source-trained model bundle
├── ofacs/                   # method implementation
├── senior_core/             # graph Transformer backbone
├── tools/                   # training, inference, and evaluation engines
└── results/                 # generated outputs (not versioned)
```

## Environment

Python 3.10 or 3.11 is recommended. Install a PyTorch build compatible with the local CUDA runtime first, then install the remaining packages:

```bash
pip install -r requirements.txt
```

## Evaluate the released model

The repository includes one verified seed-0 model bundle under `pretrained/seed_0/`. Run it on Wisconsin with:

```bash
python run_experiment.py pretrained \
  --data-root data \
  --model-root pretrained/seed_0 \
  --targets wisconsin \
  --device cuda:0
```

The same source-trained model can evaluate multiple unseen targets:

```bash
python run_experiment.py pretrained \
  --data-root data \
  --model-root pretrained/seed_0 \
  --targets wisconsin,cora \
  --device cuda:0
```

Target ground truth is read only by the final offline evaluator. It is not provided to representation calibration, candidate scoring, or community retrieval. The released checkpoint is fixed across target graphs.

## Retrain the paper model

Download and preprocess Texas, Photo, and DBLP as described in [DATASETS.md](DATASETS.md), then place them under `data/`. A one-seed paper-protocol run is:

```bash
python run_experiment.py train \
  --data-root data \
  --targets wisconsin \
  --seeds 0 \
  --device cuda:0
```

Several targets and independent seeds can be selected in one command:

```bash
python run_experiment.py train \
  --data-root data \
  --targets wisconsin,cora,citeseer \
  --seeds 0,1,2 \
  --device cuda:0
```

The source set is fixed to `texas,photo,dblp` to match the paper protocol. Each seed saves epochs 0--10 and evaluates every saved checkpoint. The reported paper result selects the best epoch separately for each target and metric within each seed, then reports the mean and sample standard deviation over three seeds.

## Datasets

Wisconsin and Cora are bundled so the released model can be evaluated immediately on selected main-table targets. The other targets use the same documented layout. Source graphs are not bundled because they are needed only for retraining.

See [DATASETS.md](DATASETS.md) for upstream dataset references, file layout, and preprocessing requirements.

## Main protocol

- source graphs: Texas, Photo, DBLP;
- unseen targets: Wisconsin, Cora, Citeseer, CoCS, Physics, Reddit;
- aligned feature dimension: 64;
- hidden dimension: 512;
- local tokens: 10;
- maximum propagation order: 4;
- diffusion training/inference steps: 200/50;
- training epochs: 10;
- learning rate: `5e-4`;
- boundary truncation ceiling: 0.90;
- community-core refinement strength: 1.25;
- reported result: mean and sample standard deviation over three independently trained seeds.

The machine-readable configuration is [configs/paper_main.json](configs/paper_main.json).

## Reference results

The paper reports the following three-seed mean results (percent):

| Target | Wisconsin | Cora | Citeseer | CoCS | Physics | Reddit |
|---|---:|---:|---:|---:|---:|---:|
| F1 | 39.21 | 58.74 | 49.93 | 47.29 | 68.53 | 36.90 |
| Jaccard | 24.39 | 41.59 | 33.27 | 30.97 | 52.13 | 22.62 |

These values use independent seeds and target-wise best epochs for F1 and Jaccard. A single fixed pretrained checkpoint is intended for reproducible inference and therefore is not expected to equal every target-wise best result simultaneously. Its verified results are recorded in [pretrained/seed_0/model_manifest.json](pretrained/seed_0/model_manifest.json).

## Hardware

Recommended configuration for retraining:

- one NVIDIA GPU with 24 GB VRAM;
- 8 or more CPU cores;
- 32 GB or more RAM;
- 50 GB free SSD space;
- Linux with a CUDA-compatible PyTorch installation.

Fixed-checkpoint inference uses substantially less memory than training. Run independent seeds sequentially on one GPU or in parallel on separate GPUs.

## Integrity check

Verify source files, bundled datasets, JSON configurations, and the released model bundle with:

```bash
python tools/verify_release.py --require-model
```

