# Dataset preparation

TOFA-CS uses three fixed source graphs and six unseen target graphs. The public
runner never changes this source/target role assignment.

| Role | Repository name | Upstream dataset |
|---|---|---|
| source | `texas` | PyG `WebKB(name="Texas")` |
| source | `photo` | PyG `Amazon(name="Photo")` |
| source | `dblp` | PyG `CitationFull(name="DBLP")` |
| target | `wisconsin` | PyG `WebKB(name="Wisconsin")` |
| target | `cora` | PyG `Planetoid(name="Cora")` |
| target | `citeseer` | PyG `Planetoid(name="CiteSeer")` |
| target | `cs` (CoCS) | PyG `Coauthor(name="CS")` |
| target | `physics` | PyG `Coauthor(name="Physics")` |
| target | `reddit` | PyG `Reddit` |

Official PyTorch Geometric references:

- WebKB: <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.WebKB.html>
- Amazon: <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Amazon.html>
- CitationFull: <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.CitationFull.html>
- Planetoid: <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html>
- Coauthor: <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Coauthor.html>
- Reddit: <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Reddit.html>

## Required project layout

For a graph named `<graph>`, place the project-format graph, strict cache,
queries, and ground truth under `data/` as follows:

```text
data/
├── <graph>_pyg.pt
├── <graph>_d64_khop4_Nl10_pos10_neg10.pt
└── <graph>/
    ├── <graph>.query
    └── <graph>.gt
```

For Photo and Coauthor CS, `<graph>_dgl.pt` is also accepted as the graph-file
name. The graph file must contain node features and `edge_index` (or the
equivalent project mapping). The strict cache stores the paper's 64-dimensional
features and local/global tokenization with `khop=4`, `Nl=10`, `pos=10`, and
`neg=10`.

The repository deliberately does not auto-download or silently regenerate
benchmark data. This avoids upstream version drift. After converting an
upstream graph to the compatible project-format `.pt` file, the strict cache
can be built with:

```bash
python tools/build_senior_cache_exact.py --help
```

Query and ground-truth files must follow the benchmark protocol used by the
paper; they are not generic node-classification train/test splits. Target
ground truth is read only by the offline evaluator and is never supplied to
training, representation calibration, or community retrieval.

## Bundled files

Wisconsin and Cora are bundled. Their strict-cache, query, and ground-truth
hashes match the archived final experiment manifest. Verify all bundled data
with `data/SHA256SUMS.txt` before a release or server run.

The remaining datasets should be placed in the same layout. Because the source
graphs are not bundled, retraining requires the user to prepare Texas, Photo,
and DBLP first. Evaluating a verified pretrained model requires only the chosen
target graph(s).
