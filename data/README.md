# Data directory

The bundled targets use the project format below:

```text
data/
├── <graph>_pyg.pt or <graph>_dgl.pt
├── <graph>_d64_khop4_Nl10_pos10_neg10.pt
└── <graph>/
    ├── <graph>.query
    └── <graph>.gt
```

Wisconsin and Cora are included. To retrain, prepare the same four artifacts for Texas, Photo, and DBLP. To evaluate another target, add its graph, strict cache, query file, and ground-truth file. Ground truth is used only by offline evaluation.

`tools/build_senior_cache_exact.py` builds the strict token cache after a compatible project-format graph file is available. It does not download datasets.
