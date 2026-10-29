import os
import random
import json
import argparse
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

import torch
import torch.nn.functional as F
try:
    from torch_geometric.data import Data as PygData
except ModuleNotFoundError:  # pragma: no cover - only for lightweight preflight/test envs
    class PygData:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
)
from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, jaccard_score


DEFAULT_DENSE_ALIGNMENT_THRESHOLD_GB = 8.0


def _dense_matrix_nbytes(num_nodes: int, dtype: torch.dtype | None = None) -> int:
    """Return bytes required by one dense [N,N] adjacency-like matrix."""
    if dtype is None:
        element_size = 4
    else:
        try:
            element_size = torch.empty((), dtype=dtype).element_size()
        except Exception:
            element_size = 4
    n = int(num_nodes)
    return n * n * int(element_size)


def estimate_dense_alignment_gb(num_nodes: int, dtype: torch.dtype | None = None) -> float:
    """Public helper used by the cache builder preflight."""
    return _dense_matrix_nbytes(num_nodes, dtype=dtype) / float(1024 ** 3)


def _sparse_sym_normalized_laplacian_feature_score(
    X_tensor: torch.Tensor,
    adj_tensor: torch.Tensor,
) -> torch.Tensor:
    """Compute diag(X^T L X) without materializing dense adjacency or L.

    This is mathematically identical to the dense eigendecomposition path used
    below when the full Fourier basis is used:

        sum_k lambda_k * (u_k^T x_j)^2 = x_j^T L x_j,

    where L = I - D^{-1/2} A_sym D^{-1/2}.  The original code only needs the
    feature-wise scores for sorting; it does not use eigenvectors elsewhere.
    """
    import gc

    X64 = X_tensor.detach().cpu().to(dtype=torch.float64).contiguous()
    n = int(X64.shape[0])
    adj_cpu = adj_tensor.coalesce().detach().cpu()
    if tuple(adj_cpu.shape) != (n, n):
        raise ValueError(f"adjacency shape {tuple(adj_cpu.shape)} does not match features with N={n}")

    idx = adj_cpu.indices().cpu().numpy()
    values = adj_cpu.values().detach().cpu().numpy()
    # Match dense semantics: adj_dense = torch.maximum(adj_dense, adj_dense.T).
    # scipy.sparse.maximum performs elementwise max without building [N,N].
    adj_sp = sp.coo_matrix((values, (idx[0], idx[1])), shape=(n, n), dtype=np.float64).tocsr()
    adj_sp = adj_sp.maximum(adj_sp.T).tocsr()
    adj_sp.eliminate_zeros()

    deg = np.asarray(adj_sp.sum(axis=1)).reshape(-1)
    deg_inv_sqrt = np.power(deg + 1e-8, -0.5, dtype=np.float64)
    deg_inv_sqrt[~np.isfinite(deg_inv_sqrt)] = 0.0

    # Compute A_norm @ X as D^{-1/2} A D^{-1/2} X, but never build A_norm as dense.
    X_np = X64.numpy()
    right_scaled = X_np * deg_inv_sqrt[:, None]
    ax = adj_sp.dot(right_scaled)
    ax *= deg_inv_sqrt[:, None]

    # diag(X^T (I - A_norm) X) = sum_i X_i^2 - sum_i X_i * (A_norm X)_i
    freq_score_np = np.sum(X_np * X_np, axis=0) - np.sum(X_np * ax, axis=0)

    del adj_sp, right_scaled, ax
    gc.collect()
    return torch.from_numpy(np.asarray(freq_score_np, dtype=np.float64))



def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()

    # main parameters   citeseer    cora    photo
    parser.add_argument('--name', type=str, default=None)
    # parser.add_argument('--dataset', type=str, default='cora',help='Choose from {pubmed}')
    # parser.add_argument('--test_dataset', type=str, default='citeseer',help='Choose from {pubmed}')
    
    # parser.add_argument('--device', type=int, default=1, help='Device cuda id')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=666, 
                        help='Random seed.')

    # model parameters
    parser.add_argument('--latent_dim_node', type=int, default=512)
    parser.add_argument('--latent_dim_neighbor', type=int, default=512)
    parser.add_argument('--diffusion_T', type=int, default=200)



    parser.add_argument('--hops', type=int, default=4,
                        help='Hop of neighbors to be calculated')
    parser.add_argument('--pe_dim', type=int, default=15,
                        help='position embedding size')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden layer size')
    parser.add_argument('--ffn_dim', type=int, default=64,
                        help='FFN layer size')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of Transformer heads')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout')
    parser.add_argument('--attention_dropout', type=float, default=0.4,
                        help='Dropout in the attention layer')
    parser.add_argument('--readout', type=str, default="mean")
    parser.add_argument('--alpha', type=float, default=0.1, 
                        help='the value the balance the loss.')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='Batch size')
    parser.add_argument('--group_epoch_gap', type=int, default=20,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Number of epochs to train.')
    parser.add_argument('--tot_updates',  type=int, default=1000,
                        help='used for optimizer learning rate scheduling')
    parser.add_argument('--warmup_updates', type=int, default=400,
                        help='warmup steps')
    parser.add_argument('--peak_lr', type=float, default=0.01, 
                        help='learning rate')
    parser.add_argument('--end_lr', type=float, default=0.0001, 
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--patience', type=int, default=50, 
                        help='Patience for early stopping')
    
    # model saving
    parser.add_argument('--save_path', type=str, default='./model/',
                        help='The path for the model to save')
    parser.add_argument('--model_name', type=str, default='cora',
                        help='The name for the model to save')
    parser.add_argument('--embedding_path', type=str, default='./pretrain_result/',
                        help='The path for the embedding to save')
    


    parser.add_argument('--embedding_tensor_name', type=str, help='embedding tensor name')
    parser.add_argument('--EmbeddingPath', type=str, default='./pretrain_result/', help='embedding path')
    parser.add_argument('--topk', type=int, default=400, help='the number of nodes selected.')

    return parser.parse_args()




class Dataset:
    """
    For each graph, we finally save:
      - adj (sparse COO)
      - feat (aligned node features)           [N, d]
      - label (community id)                  [N]
      - local_token_seqs (1-hop top-Nl + self) [N, 1+Nl, d]
      - global_token_seqs (k-hop aggregated)   [N, 1+k_hop, d]
      - pos_samples (same-comm 1-hop)         [N, pos_count]   (node ids)
      - neg_samples (non-1hop & diff-comm)    [N, neg_count]   (node ids)
    """

    def __init__(
        self,
        dims: int,
        name: str = "cora",
        k_hop: int = 4,              # for global tokens, default 4 -> [N,5,d]
        local_topk: int = 10,        # N_l, local top-k within 1-hop neighbors
        pos_count: int = 10,         # number of positive sampled node ids
        neg_count: int = 10,         # number of negative sampled node ids
        seed: int = 42,
        prefix: str = "./dataset/"
    ):
        self.name = name
        self.dims = dims
        self.k_hop = k_hop
        self.local_topk = local_topk
        self.pos_count = pos_count
        self.neg_count = neg_count
        self.seed = seed

        os.makedirs(prefix, exist_ok=True)

        # 文件名：包含关键超参数，方便复用缓存
        preprocess_filename = (
            f"{prefix}{name}_d{dims}_khop{k_hop}_Nl{local_topk}_pos{pos_count}_neg{neg_count}.pt"
        )

        if os.path.exists(preprocess_filename):
            data = torch.load(preprocess_filename)
            adj = data["adj"].coalesce()
            feat = data["feat"]
            label = data["label"]

            local_token_seqs = data["local_token_seqs"]
            global_token_seqs = data["global_token_seqs"]
            pos_samples = data["pos_samples"]
            neg_samples = data["neg_samples"]

        else:
            # 1) 读取原始数据（你原来的逻辑）
            # adj, features, label = get_dataset(name, dims)
            adj, features, label = get_dataset(name, dims)

            adj = adj.coalesce()
            row, col = adj.indices()

            edge_index = (
                torch.tensor(row, dtype=torch.long),
                torch.tensor(col, dtype=torch.long)
            )

            # 2) 特征对齐（你原来的逻辑）
            feat = my_feat_alignment(features, adj, edge_index, dims)  # [N, d]

            # 3) 构建 1-hop 邻接索引（CSR-like）用于高效取邻居
            indptr, indices = self._build_csr_from_coo(adj, num_nodes=feat.size(0))

            # 4) local tokens: [N, 1+Nl, d]
            local_token_seqs = self._build_local_token_sequences(
                feat=feat,
                indptr=indptr,
                indices=indices,
                topk=self.local_topk
            )

            # 5) global tokens: [N, 1+k_hop, d]（你指定沿用 re_features）
            global_token_seqs = re_features(adj.to(torch.float), feat, self.k_hop)

            # 6) pos/neg sampling ids
            pos_samples, neg_samples = self._build_pos_neg_samples(
                label=label,
                indptr=indptr,
                indices=indices,
                pos_count=self.pos_count,
                neg_count=self.neg_count,
                seed=self.seed
            )

            torch.save(
                {
                    "adj": adj.coalesce(),
                    "feat": feat,
                    "label": label,
                    "local_token_seqs": local_token_seqs,
                    "global_token_seqs": global_token_seqs,
                    "pos_samples": pos_samples,
                    "neg_samples": neg_samples,
                },
                preprocess_filename
            )

        # PyG Data 打包（字段名尽量语义化）
        self.graph = PygData(
            x=feat,
            adj=adj,
            y=label,
            local_token_seqs=local_token_seqs,
            global_token_seqs=global_token_seqs,
            pos_samples=pos_samples,
            neg_samples=neg_samples
        )

    # -------------------------------------------------------------------------
    # Helper: build CSR-like (indptr, indices) from sparse COO adjacency
    # -------------------------------------------------------------------------
    @staticmethod
    def _build_csr_from_coo(adj: torch.Tensor, num_nodes: int):
        """
        Input:  adj is torch sparse COO, coalesced
        Output: indptr [N+1] on CPU, indices [E] on CPU (sorted by row)

        torch.sparse_coo_tensor(...).coalesce() already returns lexicographically
        sorted COO indices in normal PyTorch builds.  The original code always
        performed an additional argsort(row), which is equivalent but very costly
        on Reddit-scale E.  We keep the fallback sort for safety, but skip it when
        the coalesced row vector is already nondecreasing.
        """
        adj = adj.coalesce()
        row, col = adj.indices()

        if row.numel() > 1 and not bool(torch.all(row[1:] >= row[:-1]).item()):
            perm = torch.argsort(row)
            row = row[perm]
            col = col[perm]

        row = row.to(torch.long)
        col = col.to(torch.long)

        counts = torch.bincount(row.cpu(), minlength=num_nodes)  # [N]
        indptr = torch.zeros(num_nodes + 1, dtype=torch.long)
        indptr[1:] = torch.cumsum(counts, dim=0)

        # put them on CPU for fast slicing in python loops
        return indptr.cpu(), col.cpu()

    @staticmethod
    def _neighbors_from_csr(indptr: torch.Tensor, indices: torch.Tensor, node_id: int):
        start = indptr[node_id].item()
        end = indptr[node_id + 1].item()
        return indices[start:end]  # 1D tensor (CPU)

    # -------------------------------------------------------------------------
    # Local tokens: self + top-k similar 1-hop neighbors
    # -------------------------------------------------------------------------
    def _build_local_token_sequences(self, feat, indptr, indices, topk: int):
        """
        local seq for each node u:
          [u] + top-k neighbors in 1-hop ranked by cosine similarity
        output shape: [N, 1+topk, d]
        """
        N, d = feat.size()
        # normalize once
        feat_norm = F.normalize(feat, p=2, dim=1)

        # store ids first, then gather features once
        local_ids = torch.empty((N, 1 + topk), dtype=torch.long)
        local_ids[:, 0] = torch.arange(N, dtype=torch.long)

        for u in range(N):
            nbrs = self._neighbors_from_csr(indptr, indices, u)
            # remove self if exists
            if nbrs.numel() > 0:
                nbrs = nbrs[nbrs != u]

            if nbrs.numel() == 0:
                chosen = torch.full((topk,), u, dtype=torch.long)
            else:
                # cosine similarity with neighbors
                sims = (feat_norm[nbrs] * feat_norm[u]).sum(dim=-1)  # [deg(u)]
                k = min(topk, nbrs.numel())
                topk_idx = torch.topk(sims, k=k, largest=True).indices
                chosen = nbrs[topk_idx].to(torch.long)

                # pad if not enough neighbors
                if k < topk:
                    pad = chosen[torch.randint(0, k, (topk - k,))] if k > 0 else torch.full((topk - k,), u, dtype=torch.long)
                    chosen = torch.cat([chosen, pad], dim=0)

            local_ids[u, 1:] = chosen

        # gather: [N, 1+topk, d]
        local_token_seqs = feat[local_ids]
        return local_token_seqs

    # -------------------------------------------------------------------------
    # Large-graph equivalent fast path for local tokens.
    # -------------------------------------------------------------------------
    def _build_local_token_sequences_fast(
        self,
        feat,
        indptr,
        indices,
        topk: int,
        *,
        edge_score_chunk_size: int = 1_000_000,
        score_device: str = "auto",
        seed: int | None = None,
        row_block_size: int = 4096,
        progress_every: int = 50,
    ):
        """Sparse-safe exact local-token construction for large graphs.

        The original implementation computes, for each node u, cosine scores
        between feat[u] and every 1-hop neighbor, then selects top-k neighbors
        and pads from the selected ids when degree < k.  This method preserves
        the same definition but processes nodes in row blocks.  It never builds
        an N x N score matrix and, unlike the earlier fast path, it also avoids
        keeping global row_cpu[E] and scores[E] arrays for all edges.

        For small graphs, keep using _build_local_token_sequences() so historical
        caches remain governed by the old loop and RNG behavior.
        """
        N, d = feat.size()
        if int(topk) < 0:
            raise ValueError("topk must be non-negative")
        if int(topk) == 0:
            return feat[torch.arange(N, dtype=torch.long).view(N, 1)]

        feat_cpu = feat.detach().cpu().float().contiguous()
        feat_norm = F.normalize(feat_cpu, p=2, dim=1).contiguous()
        indptr_cpu = indptr.detach().cpu().to(torch.long).contiguous()
        indices_cpu = indices.detach().cpu().to(torch.long).contiguous()
        E = int(indices_cpu.numel())

        use_cuda = False
        if str(score_device) in {"auto", "cuda"} and torch.cuda.is_available():
            use_cuda = True
        if str(score_device) == "cpu":
            use_cuda = False
        if str(score_device) == "cuda" and not use_cuda:
            print("[fast-local] CUDA requested but unavailable; using CPU edge scoring", flush=True)

        row_block = max(1, int(row_block_size))
        chunk = max(1, int(edge_score_chunk_size))
        progress_every = max(1, int(progress_every))
        local_ids = torch.empty((N, 1 + int(topk)), dtype=torch.long)
        local_ids[:, 0] = torch.arange(N, dtype=torch.long)

        if use_cuda:
            feat_norm_dev = feat_norm.cuda(non_blocking=False)
        else:
            feat_norm_dev = feat_norm

        print(
            f"[fast-local] row-block scoring N={N}, E={E}, row_block={row_block}, "
            f"edge_chunk={chunk}, device={'cuda' if use_cuda else 'cpu'}",
            flush=True,
        )
        processed_edges = 0
        block_id = 0
        with torch.no_grad():
            for u0 in range(0, N, row_block):
                u1 = min(N, u0 + row_block)
                e0 = int(indptr_cpu[u0].item())
                e1 = int(indptr_cpu[u1].item())
                deg_block = indptr_cpu[u0 + 1:u1 + 1] - indptr_cpu[u0:u1]
                block_edges = int(e1 - e0)

                if block_edges > 0:
                    row_block_cpu = torch.repeat_interleave(
                        torch.arange(u0, u1, dtype=torch.long),
                        deg_block,
                    )
                    col_block_cpu = indices_cpu[e0:e1]
                    score_block = torch.empty(block_edges, dtype=torch.float32)
                    for s in range(0, block_edges, chunk):
                        t = min(block_edges, s + chunk)
                        r = row_block_cpu[s:t]
                        c = col_block_cpu[s:t]
                        if use_cuda:
                            r_dev = r.cuda(non_blocking=False)
                            c_dev = c.cuda(non_blocking=False)
                            sc = (feat_norm_dev[r_dev] * feat_norm_dev[c_dev]).sum(dim=1).detach().cpu()
                        else:
                            sc = (feat_norm_dev[r] * feat_norm_dev[c]).sum(dim=1).detach().cpu()
                        score_block[s:t] = sc.to(dtype=torch.float32)
                    processed_edges += block_edges
                else:
                    score_block = torch.empty(0, dtype=torch.float32)

                for u in range(u0, u1):
                    start = int(indptr_cpu[u].item())
                    end = int(indptr_cpu[u + 1].item())
                    rel_start = start - e0
                    rel_end = end - e0
                    nbrs = indices_cpu[start:end]
                    if nbrs.numel() > 0:
                        keep = nbrs != int(u)
                        nbrs = nbrs[keep]
                        row_scores = score_block[rel_start:rel_end][keep]
                    else:
                        row_scores = score_block[rel_start:rel_end]

                    if nbrs.numel() == 0:
                        chosen = torch.full((int(topk),), int(u), dtype=torch.long)
                    else:
                        k = min(int(topk), int(nbrs.numel()))
                        top_idx = torch.topk(row_scores, k=k, largest=True).indices
                        chosen = nbrs[top_idx].to(torch.long)
                        if k < int(topk):
                            # Preserve the original implementation's padding RNG
                            # behavior: it uses the process-global torch RNG here.
                            pad_idx = torch.randint(0, k, (int(topk) - k,))
                            chosen = torch.cat([chosen, chosen[pad_idx]], dim=0)
                    local_ids[u, 1:] = chosen

                block_id += 1
                if block_id % progress_every == 0 or u1 == N:
                    print(
                        f"[fast-local] processed rows {u1}/{N}, edges {processed_edges}/{E}",
                        flush=True,
                    )

        if use_cuda:
            del feat_norm_dev
            torch.cuda.empty_cache()

        # Gather in moderate chunks to avoid an extra peak allocation from a
        # very large advanced-indexing expression.
        out = torch.empty((N, 1 + int(topk), d), dtype=feat_cpu.dtype)
        gather_chunk = max(1, min(N, 16384))
        for start in range(0, N, gather_chunk):
            end = min(N, start + gather_chunk)
            out[start:end] = feat_cpu[local_ids[start:end]]
            if ((start // gather_chunk) + 1) % max(1, progress_every) == 0 or end == N:
                print(f"[fast-local] gathered rows {end}/{N}", flush=True)
        return out

    # -------------------------------------------------------------------------
    # Positive / Negative sampling ids for contrastive learning
    # -------------------------------------------------------------------------
    def _build_pos_neg_samples(self, label, indptr, indices, pos_count, neg_count, seed: int):
        """
        pos(u): random choose pos_count nodes from 1-hop neighbors with same community
        neg(u): random choose neg_count nodes that are:
               - not directly connected to u (exclude 1-hop neighbors)
               - different community
        only save node ids.
        """
        random.seed(seed)
        torch.manual_seed(seed)

        N = label.size(0)
        label_cpu = label.cpu()

        # precompute pools by label (for efficient diff-community sampling)
        uniq = torch.unique(label_cpu).tolist()
        nodes_by_label = {}
        for c in uniq:
            nodes_by_label[c] = (label_cpu == c).nonzero(as_tuple=False).view(-1)

        diff_pool = {}
        for c in uniq:
            others = [nodes_by_label[oc] for oc in uniq if oc != c]
            diff_pool[c] = torch.cat(others, dim=0) if len(others) > 0 else torch.empty((0,), dtype=torch.long)

        pos_samples = torch.empty((N, pos_count), dtype=torch.long)
        neg_samples = torch.empty((N, neg_count), dtype=torch.long)

        for u in range(N):
            u_lab = label_cpu[u].item()

            nbrs = self._neighbors_from_csr(indptr, indices, u)
            if nbrs.numel() > 0:
                nbrs = nbrs[nbrs != u]

            # neighbor set for fast rejection (CPU)
            nbr_set = set(nbrs.tolist())
            nbr_set.add(u)

            # ---- positives: 1-hop & same community
            if nbrs.numel() == 0:
                pos = [u] * pos_count
            else:
                same = [v for v in nbrs.tolist() if label_cpu[v].item() == u_lab]
                if len(same) == 0:
                    pos = [u] * pos_count
                elif len(same) >= pos_count:
                    pos = random.sample(same, pos_count)
                else:
                    pos = random.choices(same, k=pos_count)

            pos_samples[u] = torch.tensor(pos, dtype=torch.long)

            # ---- negatives: non-1hop & different community
            pool = diff_pool[u_lab]
            if pool.numel() == 0:
                neg = [0] * neg_count
            else:
                neg = []
                # rejection sampling: neighbors are typically small, acceptance is high
                max_tries = neg_count * 50
                tries = 0
                while len(neg) < neg_count and tries < max_tries:
                    v = pool[torch.randint(0, pool.numel(), (1,)).item()].item()
                    if v not in nbr_set:
                        neg.append(v)
                    tries += 1

                # fallback if extremely unlucky: allow replacement from pool ignoring neighbor constraint (rare)
                if len(neg) < neg_count:
                    extra = pool[torch.randint(0, pool.numel(), (neg_count - len(neg),))].tolist()
                    neg.extend(extra)

            neg_samples[u] = torch.tensor(neg[:neg_count], dtype=torch.long)

        return pos_samples, neg_samples

    def _build_pos_neg_samples_fast(self, label, indptr, indices, pos_count, neg_count, seed: int, *, progress_every_nodes: int = 10000):
        """Large-graph equivalent sampler for pos_samples/neg_samples.

        It preserves the cache definition:
        - positives are 1-hop neighbors with the same community label;
        - negatives are sampled from different-label nodes while rejecting the
          query node and direct 1-hop neighbors;
        - replacement is used only in the same fallback cases as the original
          implementation, i.e., not enough positives or rare negative retry
          exhaustion.

        The difference is implementation only: vectorized NumPy draws replace
        scalar torch.randint calls in the inner negative-sampling loop.
        """
        rng = np.random.default_rng(int(seed))
        label_np = label.detach().cpu().numpy()
        indptr_np = indptr.detach().cpu().numpy().astype(np.int64, copy=False)
        indices_np = indices.detach().cpu().numpy().astype(np.int64, copy=False)

        N = int(label_np.shape[0])
        pos_count = int(pos_count)
        neg_count = int(neg_count)
        pos_samples_np = np.empty((N, pos_count), dtype=np.int64)
        neg_samples_np = np.empty((N, neg_count), dtype=np.int64)

        uniq = np.unique(label_np)
        nodes_by_label = {c: np.flatnonzero(label_np == c).astype(np.int64, copy=False) for c in uniq}
        all_nodes = np.arange(N, dtype=np.int64)
        diff_pool = {}
        for c in uniq:
            # Equivalent to concatenating every label pool whose label != c, but
            # faster and lower-overhead when label count is small.
            diff_pool[c] = all_nodes[label_np != c]

        progress_every_nodes = max(1, int(progress_every_nodes))
        print(f"[pos-neg-fast] sampling N={N}, pos_count={pos_count}, neg_count={neg_count}", flush=True)
        for u in range(N):
            u_lab = label_np[u]
            nbrs = indices_np[indptr_np[u]:indptr_np[u + 1]]
            if nbrs.size:
                nbrs = nbrs[nbrs != u]

            # Positives: same-community 1-hop neighbors.
            if nbrs.size == 0:
                pos_samples_np[u, :] = u
            else:
                same = nbrs[label_np[nbrs] == u_lab]
                if same.size == 0:
                    pos_samples_np[u, :] = u
                elif same.size >= pos_count:
                    pos_samples_np[u, :] = rng.choice(same, size=pos_count, replace=False)
                else:
                    pos_samples_np[u, :] = rng.choice(same, size=pos_count, replace=True)

            # Negatives: different-community and not directly connected.
            pool = diff_pool[u_lab]
            if pool.size == 0:
                neg_samples_np[u, :] = 0
                continue

            banned = set(nbrs.tolist())
            banned.add(int(u))
            neg = []
            tries = 0
            max_tries = neg_count * 50
            # Draw in vectorized mini-batches.  Rejection distribution is the
            # same as repeatedly drawing uniformly from diff_pool and accepting
            # non-neighbors, but avoids one torch.randint call per candidate.
            while len(neg) < neg_count and tries < max_tries:
                draw = min(max(neg_count * 4, 32), max_tries - tries)
                cand = pool[rng.integers(0, pool.size, size=draw, endpoint=False)]
                for v in cand.tolist():
                    if v not in banned:
                        neg.append(int(v))
                        if len(neg) >= neg_count:
                            break
                tries += draw

            if len(neg) < neg_count:
                extra = pool[rng.integers(0, pool.size, size=neg_count - len(neg), endpoint=False)].tolist()
                neg.extend(int(v) for v in extra)
            neg_samples_np[u, :] = np.asarray(neg[:neg_count], dtype=np.int64)

            if (u + 1) % progress_every_nodes == 0 or (u + 1) == N:
                print(f"[pos-neg-fast] processed rows {u + 1}/{N}", flush=True)

        return torch.from_numpy(pos_samples_np), torch.from_numpy(neg_samples_np)


# class Dataset:
#     def __init__(self, dims, name='cora', incontext_count=5, k_hop=3,sample_count=10, prefix='./dataset/constructed/my/'):
#         self.name = name
#         self.dims = dims
#         self.graph = None
#         self.incontext_count = incontext_count
#         self.k_hop = k_hop
#         self.sample_count = sample_count

#         # 修改文件名格式，包含维度、正采样数量和跳数
#         preprocess_filename = f'{prefix}{name}_{dims}_{k_hop}_{incontext_count}_{sample_count}.pt'

#         if os.path.exists(preprocess_filename):
#             # 直接读取已保存的文件
#             data = torch.load(preprocess_filename)
#             adj = data["adj"]
#             feat = data["feat"]
#             label = data.get("label", None)  # 获取标签（社区归属）
#             # 尝试读取保存的正采样和token序列
#             self.few_shot_com = data.get("few_shot_com", None)
#             self.incontext_nodes = data.get("incontext_nodes", None)
#             self.similarity_samples = data.get("similarity_samples", None)
#             self.label_samples = data.get("label_samples", None)
#             self.tokenized_sequences = data.get("tokenized_seqs", None)

#         self.graph = Data(
#             x=feat,
#             adj=adj,
#             y=label,
#             few_shot_com=self.few_shot_com,
#             incontext_nodes=self.incontext_nodes,
#             similarity_samples=self.similarity_samples,
#             label_samples=self.label_samples,
#             tokenized_seqs=self.tokenized_sequences
#         )

#     def _generate_incontext_nodes(self, adj, label):
#         """
#         为每个节点生成指定数量的k跳以内且同社区的采样
#         返回维度: [节点数, incontext_count]
#         """
#         num_nodes = label.size(0)
#         positive_samples = torch.zeros((num_nodes, self.incontext_count), dtype=torch.long)
        
#         # 计算邻接矩阵的稀疏表示
#         adj_sp = sp.csr_matrix(adj.to_dense().numpy())
        
#         # 计算k跳邻居矩阵 (累积1到k跳所有邻居)
#         k_hop_adj = adj_sp.copy()  # 1跳邻居
#         current_adj = adj_sp.copy()
        
#         for hop in range(2, self.k_hop + 1):
#             current_adj = current_adj.dot(adj_sp)
#             k_hop_adj = k_hop_adj + current_adj
    
#         # 转为binary矩阵（只关心是否可达）
#         k_hop_adj = (k_hop_adj > 0).astype(int)
    
#         # 为每个节点采样同社区的k跳邻居
#         for node_idx in range(num_nodes):
#             # 获取节点的k跳邻居
#             neighbors = set(k_hop_adj[node_idx].nonzero()[1])
            
#             # 移除节点自身
#             if node_idx in neighbors:
#                 neighbors.remove(node_idx)
            
#             # 筛选同社区节点
#             node_comm = label[node_idx].item()
#             same_comm_neighbors = [n for n in neighbors if label[n].item() == node_comm]
            
#             # 采样逻辑（简化版）
#             if len(same_comm_neighbors) == 0:
#                 # 如果没有符合条件的邻居，采样自己
#                 sampled_nodes = [node_idx] * self.incontext_count
#             else:
#                 # 有重复地随机采样
#                 sampled_nodes = random.choices(same_comm_neighbors, k=self.incontext_count)
            
#             positive_samples[node_idx] = torch.tensor(sampled_nodes, dtype=torch.long)
    
#         return positive_samples
    

#     def _generate_samples(self, adj, feat, label):
#         """
#         同时生成两种采样：
#         1. 基于相似度的采样 (特征相似性和拓扑相似性)
#         2. 基于标签的采样 (同社区/不同社区)
        
#         返回：
#         - similarity_samples: [num_nodes, 2, sample_count] - 基于相似度的正负样本
#         - label_samples: [num_nodes, 2, sample_count] - 基于标签的正负样本
#         """
#         num_nodes = feat.size(0)
        
#         # 初始化结果张量
#         similarity_samples = torch.zeros((num_nodes, 2, self.sample_count), dtype=torch.long)
#         label_samples = torch.zeros((num_nodes, 2, self.sample_count), dtype=torch.long)
        
#         # 将邻接矩阵转换为稠密矩阵，用于相似度计算
#         adj_dense = adj.to_dense()
        
#         # =================== 归一化特征和邻接矩阵 ===================
#         # L2归一化特征
#         feat_norm = feat.clone()
#         feat_norm_l2 = torch.norm(feat_norm, p=2, dim=1, keepdim=True)
#         feat_norm = feat_norm / (feat_norm_l2 + 1e-8)
        
#         # 归一化邻接矩阵 (D^-0.5 * A * D^-0.5)
#         A = adj_dense.clone()
#         A = A + torch.eye(num_nodes)  # 添加自环
#         D_inv_sqrt = torch.diag(torch.pow(A.sum(dim=1), -0.5))
#         D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
#         A_norm = D_inv_sqrt.mm(A).mm(D_inv_sqrt)
        
#         # =================== 计算k跳拓扑特征 ===================
#         # 计算k跳特征传播 (A^k * X)
#         A_k = A_norm.clone()
#         for _ in range(self.k_hop - 1):
#             A_k = A_k.mm(A_norm)
#         X_topo = A_k.mm(feat)
        
#         # 归一化拓扑特征
#         X_topo_norm = X_topo.clone()
#         X_topo_norm_l2 = torch.norm(X_topo_norm, p=2, dim=1, keepdim=True)
#         X_topo_norm = X_topo_norm / (X_topo_norm_l2 + 1e-8)
        
#         # =================== 基于社区标签预处理 ===================
#         # 按标签将节点分组
#         communities = {}
#         for node_idx, comm_label in enumerate(label):
#             comm_label = comm_label.item()
#             if comm_label not in communities:
#                 communities[comm_label] = []
#             communities[comm_label].append(node_idx)
        
#         # =================== 为每个节点生成样本 ===================
#         for node_idx in range(num_nodes):
#             # -------- 基于相似度采样 --------
#             # 计算属性相似度
#             feat_sim = torch.mv(feat_norm, feat_norm[node_idx])
            
#             # 计算拓扑相似度
#             topo_sim = torch.mv(X_topo_norm, X_topo_norm[node_idx])
            
#             # 混合相似度 (属性 + 拓扑)/2
#             hybrid_sim = (feat_sim + topo_sim) * 0.5
            
#             # 将自身设为负无穷，避免选到自己
#             hybrid_sim[node_idx] = float("-inf")
            
#             # 正样本: 取相似度最高的k个
#             pos_indices = torch.topk(hybrid_sim, self.sample_count).indices
#             similarity_samples[node_idx, 0] = pos_indices
            
#             # 负样本: 随机选择k个（排除自身和正样本）
#             mask = torch.ones(num_nodes, dtype=torch.bool)
#             mask[node_idx] = False  # 排除自身
#             mask[pos_indices] = False  # 排除已选的正样本
#             remaining_nodes = torch.nonzero(mask, as_tuple=False).squeeze(-1)

#             # 随机选择指定数量的负样本
#             if remaining_nodes.dim() == 0 and remaining_nodes.numel() > 0:
#                 # 如果只剩一个节点可选，重复这个节点
#                 neg_indices = torch.tensor([remaining_nodes.item()] * self.sample_count)
#             elif len(remaining_nodes) < self.sample_count:
#                 # 如果可选节点不足sample_count个，随机重复采样
#                 neg_indices = remaining_nodes[torch.randint(len(remaining_nodes), (self.sample_count,))]
#             else:
#                 # 随机选择sample_count个不同的节点
#                 perm = torch.randperm(len(remaining_nodes))
#                 neg_indices = remaining_nodes[perm[:self.sample_count]]

#             similarity_samples[node_idx, 1] = neg_indices
            
#             # -------- 基于标签采样 --------
#             # 获取当前节点的社区标签
#             node_comm = label[node_idx].item()
            
#             # 获取同社区节点列表（排除自身）
#             same_comm = [n for n in communities[node_comm] if n != node_idx]
            
#             # 获取不同社区节点列表
#             diff_comm = []
#             for comm, nodes in communities.items():
#                 if comm != node_comm:
#                     diff_comm.extend(nodes)
            
#             # 正样本: 从同社区中随机采样
#             if len(same_comm) >= self.sample_count:
#                 pos_label_indices = torch.tensor(random.sample(same_comm, self.sample_count))
#             else:
#                 # 不足时有重复地随机采样
#                 pos_label_indices = torch.tensor(random.choices(same_comm, k=self.sample_count) if same_comm else [node_idx] * self.sample_count)
            
#             label_samples[node_idx, 0] = pos_label_indices
            
#             # 负样本: 从不同社区中随机采样
#             if len(diff_comm) >= self.sample_count:
#                 neg_label_indices = torch.tensor(random.sample(diff_comm, self.sample_count))
#             else:
#                 # 不足时有重复地随机采样
#                 neg_label_indices = torch.tensor(random.choices(diff_comm, k=self.sample_count) if diff_comm else [0] * self.sample_count)
            
#             label_samples[node_idx, 1] = neg_label_indices
        
#         return similarity_samples, label_samples #[3327, 2, 10], [3327, 2, 10]

        
#     def _generate_tokenized_sequences(self, adj, feat):
#         """
#         为每个节点生成tokenized序列:
#         - 节点自身特征
#         - 1到k-hop子图聚合特征
#         每个特征向量后附加一个独热编码，表示是哪一跳的特征
#         """
#         num_nodes = feat.size(0)
#         feat_dim = feat.size(1)
        
#         # 计算邻接矩阵的稀疏表示 - 修复类型问题
#         adj_dense = adj.to_dense().numpy()
#         adj_with_self_loops = sp.csr_matrix(adj_dense, dtype=np.float32)
#         adj_with_self_loops.setdiag(1)
        
#         # 初始化序列长度 = 1(自身) + k_hop(子图特征)
#         seq_length = 1 + self.k_hop
        
#         # 独热编码长度 = k_hop + 1 (节点自身 + k个hop)
#         onehot_dim = self.k_hop + 1
        
#         # 生成特征维度 = 原始特征维度 + 独热编码维度
#         extended_feat_dim = feat_dim + onehot_dim
        
#         # 初始化序列张量
#         tokenized_seqs = torch.zeros((num_nodes, seq_length, extended_feat_dim))
        
#         # 计算k-hop邻居的聚合特征
#         hop_features = self._compute_k_hop_features(adj_with_self_loops, feat, self.k_hop)
        
#         # 为每个节点生成序列
#         for node_idx in range(num_nodes):
#             # 第一个token: 节点自身特征 + 独热编码 [1,0,0,...]
#             tokenized_seqs[node_idx, 0, :feat_dim] = feat[node_idx]
#             tokenized_seqs[node_idx, 0, feat_dim] = 1.0  # 设置独热编码
            
#             # 后续tokens: k-hop邻居聚合特征 + 独热编码
#             for hop in range(self.k_hop):
#                 # 特征部分
#                 tokenized_seqs[node_idx, hop+1, :feat_dim] = hop_features[hop][node_idx]
#                 # 独热编码部分 [0,1,0,0,...] 或 [0,0,1,0,...] 等
#                 tokenized_seqs[node_idx, hop+1, feat_dim+hop+1] = 1.0
        
#         return tokenized_seqs  # [num_nodes, 1+k_hop, feat_dim+onehot_dim]

#     def _compute_k_hop_features(self, adj, features, k_hop):

#         adj_normalized = normalize_adj(adj)
#         adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
        
#         # 初始化特征列表，存储每一跳的聚合特征
#         hop_features = []
        
#         # 存储当前跳的特征
#         current_features = features
        
#         # 计算每一跳的聚合特征
#         for hop in range(k_hop):
#             # 使用当前特征计算下一跳的聚合特征
#             current_features = torch.sparse.mm(adj_normalized, current_features)
            
#             # # 归一化
#             # norm = torch.norm(current_features, dim=1, keepdim=True)
#             # normalized_feat = current_features / (norm + 1e-10)
            
#             # 添加到结果列表
#             hop_features.append(current_features)
    
#         return hop_features


def test(query_id, new_embs, labels, threshold=0.2):
    """
    query_id: int, 查询节点索引
    new_embs: [N, d] Tensor, 所有节点 embedding
    labels: [N] Tensor/ndarray, 每个节点的社区id
    threshold: float, 相似度阈值

    return: f1分数 (float), 预测集合 (list), 真实集合 (list)
    """
    device = new_embs.device

    # (1) 取 query embedding
    q_emb = new_embs[query_id].unsqueeze(0)  # [1, d]

    # (2) 归一化
    new_embs_norm = F.normalize(new_embs, dim=1)#[2708, 512]
    q_emb_norm = F.normalize(q_emb, dim=1) #[1, 512]

    # (3) 计算余弦相似度
    sims = torch.mm(q_emb_norm, new_embs_norm.T).squeeze(0)  # [N]

    # (4) 根据阈值选预测集合
    # pred_nodes = (sims > threshold).nonzero(as_tuple=False).view(-1).tolist()

    pred_nodes = GlobalSearch([query_id.item()], sims.tolist()) 

    # (5) 真实社区集合
    query_label = labels[query_id]
    true_nodes = (labels == query_label).nonzero(as_tuple=False).view(-1).tolist()

    # (6) 转换成 one-hot 标记，用 sklearn f1_score
    y_true = torch.zeros_like(sims, dtype=torch.long)
    y_true[true_nodes] = 1
    y_pred = torch.zeros_like(sims, dtype=torch.long)
    y_pred[pred_nodes] = 1

    f1 = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy())

    return f1

def subgraph_density_controled(candidate_score, graph_score): #计算ESG  7页左上角
    
    weight_gain = (sum(candidate_score)-sum(graph_score)*(len(candidate_score)**1)/(len(graph_score)**1))/(len(candidate_score)**0.50)
    return weight_gain
def GlobalSearch(query_index, graph_score):  # [3,] , [2708,]  算法5

    candidates = query_index
    selected_candidate = candidates

    graph_score=np.array(graph_score) #(2708,)
    max2min_index = np.argsort(-graph_score) #从大到小排序,返回索引
    
    startpoint = 0
    endpoint = int(0.50*len(max2min_index)) #1354
    if endpoint >= 10000:
        endpoint = 10000
    
    while True: #找到一个最佳分割点
        candidates_half = query_index+[max2min_index[i] for i in range(0, int((startpoint+endpoint)/2))]  #len() =680  加上前一半分数大的节点
        candidate_score_half = [graph_score[i] for i in candidates_half] #记录这些节点的score
        candidates_density_half = subgraph_density_controled(candidate_score_half, graph_score)

        candidates = query_index+[max2min_index[i] for i in range(0, endpoint)] #len() =1357
        candidate_score = [graph_score[i] for i in candidates]
        candidates_density = subgraph_density_controled(candidate_score, graph_score)

        if candidates_density>= candidates_density_half: #类似折半查找
            startpoint = int((startpoint+endpoint)/2)
            endpoint = endpoint
        else:
            startpoint = startpoint
            endpoint = int((startpoint+endpoint)/2)
        
        if startpoint == endpoint or startpoint+1 == endpoint:
            break

    selected_candidate = query_index+[max2min_index[i] for i in range(0, startpoint)] 
    
    return selected_candidate #选择score大的节点,直到分割点


def get_dataset(dataset, pe_dim):
    if dataset in {"pubmed", "photo", "cs", "cora", "physics","citeseer"}:
        if dataset in {"photo", "cs"}:
            file_path = "dataset/"+dataset+"_dgl.pt"
        else:
            file_path = "dataset/"+dataset+"_pyg.pt"
        # file_path = "dataset/"+dataset+".pt"
        data_list = torch.load(file_path)
        
        adj = data_list[0] #邻接矩阵 [2708, 2708]
        features = data_list[1] #特征矩阵 [2708, 1433]
        label = data_list[2] #社区标签 [2708]
        
    
    elif dataset in {"texas", "cornell", "wisconsin", "dblp", "reddit"}:
        file_path = "dataset/"+dataset+"_pyg.pt"

        data_list = torch.load(file_path)
        
        adj = data_list[0]
        features = data_list[1]
        label = data_list[2]
    
    print(type(adj), type(features), type(label))
    
    return adj.cpu().type(torch.LongTensor), features.float(), label.long()



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def feat_alignment(X, edge_index, dims): #[17716, 1639], 2*[105734], 64
    """Feature alignment: PCA + smoothness-based sorting."""
    edge_src, edge_dst = edge_index
    num_edges = edge_src.shape[0]

    # 如果维度过低，先做随机投影到 256
    if X.shape[1] < dims:
        transformer = GaussianRandomProjection(n_components=256, random_state=0)
        X = transformer.fit_transform(X.cpu().numpy())
    else:
        X = X.cpu().numpy()

    # PCA 投影到 dims
    pca = PCA(n_components=dims, random_state=0)
    X_transformed = pca.fit_transform(X)
    X_transformed = torch.FloatTensor(X_transformed) #[17716, 64]

    # Min-Max 归一化
    X_min, _ = torch.min(X_transformed, dim=0)
    X_max, _ = torch.max(X_transformed, dim=0)
    X_s = (X_transformed - X_min) / (X_max - X_min + 1e-10)

    # 计算 smoothness 值
    smooth_coefficients = torch.zeros(X_transformed.shape[1])
    for k in range(X_transformed.shape[1]):
        differences = X_s[edge_src, k] - X_s[edge_dst, k]
        smooth_coefficients[k] = torch.sum(differences ** 2) / num_edges

    # 根据 smoothness 排序（从低到高）
    _, sorted_indices = torch.sort(smooth_coefficients)
    X_reordered = X_transformed[:, sorted_indices]

    return X_reordered

def my_feat_alignment(
    X,
    adj,
    edge_index,
    dims,
    alpha=0.5,
    p_min=0.05,
    p_max=0.2,
    beta=1,
    *,
    sparse_safe=None,
    dense_threshold_gb=None,
):  # [17716, 1639], 2*[105734], 64
    """Feature alignment: PCA + spectral-smoothness sorting.

    Small graphs keep the original dense full-eigendecomposition branch exactly.
    Large graphs automatically use a sparse-safe branch that computes the same
    full-basis frequency score via the identity

        sum_k lambda_k * (u_k^T x_j)^2 == x_j^T L x_j.

    Therefore the cache semantics are unchanged: we still sort PCA-projected
    features by the normalized-Laplacian spectral energy.  The sparse branch
    only avoids materializing dense adjacency/Laplacian/eigenvectors.
    """

    # 如果维度过低，先做随机投影到 256
    if X.shape[1] < dims:
        transformer = GaussianRandomProjection(n_components=256, random_state=0)
        X = transformer.fit_transform(X.cpu().numpy())
    else:
        X = X.cpu().numpy()

    # PCA 投影到 dims
    pca = PCA(n_components=dims, random_state=0)
    X = pca.fit_transform(X)
    X = torch.FloatTensor(X)  # [N, dims]

    X_tensor = X
    adj_tensor = adj.coalesce() if getattr(adj, "is_sparse", False) else adj
    n = int(X_tensor.size(0))

    if dense_threshold_gb is None:
        dense_threshold_gb = DEFAULT_DENSE_ALIGNMENT_THRESHOLD_GB
    dense_gb = estimate_dense_alignment_gb(n, dtype=getattr(adj_tensor, "dtype", torch.float32))
    if sparse_safe is None:
        sparse_safe = dense_gb > float(dense_threshold_gb)

    if sparse_safe:
        print(
            f"[sparse-safe-align] using sparse normalized-Laplacian energy: "
            f"N={n}, estimated_dense_adj={dense_gb:.2f} GiB, threshold={float(dense_threshold_gb):.2f} GiB",
            flush=True,
        )
        X_tensor64 = X_tensor.to(dtype=torch.float64).contiguous()
        freq_score = _sparse_sym_normalized_laplacian_feature_score(X_tensor64, adj_tensor)
        sorted_features = torch.argsort(freq_score)
        return X_tensor64[:, sorted_features]

    # ============================================================
    # Original dense branch.  Keep this path for small graphs so historical
    # cache values remain governed by the old torch.linalg.eigh implementation.
    # ============================================================
    print(
        f"[dense-align] using original torch dense eigendecomposition: "
        f"N={n}, estimated_dense_adj={dense_gb:.2f} GiB, threshold={float(dense_threshold_gb):.2f} GiB",
        flush=True,
    )

    # 1) 构造稠密邻接矩阵（只允许小图走到这里）
    adj_dense = adj_tensor.to_dense()
    adj_dense = torch.maximum(adj_dense, adj_dense.T)
    device = adj_dense.device
    X_tensor = X_tensor.to(device=device)

    # 2) 构造对称归一化 Laplacian: L = I - D^{-1/2} A D^{-1/2}
    deg = adj_dense.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
    D_inv_sqrt = deg_inv_sqrt.unsqueeze(1)
    L = torch.eye(n, device=device) - D_inv_sqrt * adj_dense * D_inv_sqrt.T

    # 3) 原始完整 dense eigendecomposition
    eigvals, U = torch.linalg.eigh(L)

    # 4) Graph Fourier Transform: coeffs = U^T X
    coeffs = U.T @ X_tensor
    energy = coeffs * coeffs
    freq_score = (eigvals.unsqueeze(1) * energy).sum(dim=0)
    sorted_features = torch.argsort(freq_score)
    X_sorted = X_tensor[:, sorted_features]
    return X_sorted



def my_feat_alignment_cpu(X, adj, edge_index, dims,alpha=0.5, p_min=0.05, p_max=0.2, beta=1): #[17716, 1639], 2*[105734], 64
    """Feature alignment: PCA + smoothness-based sorting."""


    # 如果维度过低，先做随机投影到 256
    if X.shape[1] < dims:
        transformer = GaussianRandomProjection(n_components=256, random_state=0)
        X = transformer.fit_transform(X.cpu().numpy())
    else:
        X = X.cpu().numpy()

    # PCA 投影到 dims
    pca = PCA(n_components=dims, random_state=0)
    X = pca.fit_transform(X)
    X = torch.FloatTensor(X) #[17716, 64]


    X_np = X.cpu().numpy()               # (2708, 64)

    indices = adj.indices().cpu().numpy()
    values = adj.values().cpu().numpy()
    n = adj.size(0)

    # ===========================================================
    # 1) 构造稀疏邻接矩阵（SciPy）
    # ===========================================================
    adj_sp = sp.coo_matrix((values, (indices[0], indices[1])),
                        shape=(n, n))
    # 变成对称
    adj_sp = adj_sp + adj_sp.T.multiply(adj_sp.T > adj_sp) - adj_sp.multiply(adj_sp.T > adj_sp)

    # ===========================================================
    # 2) 构造对称归一化 Laplacian  L = I - D^{-1/2} A D^{-1/2}
    # ===========================================================
    deg = np.array(adj_sp.sum(1)).flatten()
    deg_inv_sqrt = 1.0 / np.sqrt(deg + 1e-8)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)

    L = sp.eye(n) - D_inv_sqrt @ adj_sp @ D_inv_sqrt

    # ===========================================================
    # 3) 取前 K 个最小特征值/特征向量（图傅里叶基）
    # ===========================================================
    K = 200  # 你可以改，例如 200、100、300
    eigvals, U = eigsh(L, k=K, which='SM')   # U shape: (2708, K)

    # 排序
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    U = U[:, idx]

    # ===========================================================
    # 4) Graph Fourier Transform：coeffs = U^T X
    # ===========================================================
    # X: (n, 64)
    coeffs = U.T @ X_np      # shape = (K, 64)

    # ===========================================================
    # 5) 每个特征在每个频率上的"能量" = coeffs^2
    # ===========================================================
    energy = coeffs**2       # shape (K, 64)

    # ===========================================================
    # 6) 对每个特征，计算它的加权频率中心（越小越低频）
    #    freq_score[d] = sum_k (eigval[k] * energy[k, d])
    # ===========================================================
    freq_score = (eigvals.reshape(-1, 1) * energy).sum(axis=0)

    # 排序特征（从低频 → 高频）
    sorted_features = np.argsort(freq_score)
    X_sorted = X_np[:, sorted_features]




    return torch.tensor(X_sorted, dtype=torch.float32)



def re_features(adj, features, K):#[2708, 2708] , [2708, 1436] , 5
    """Senior-compatible hop feature trajectory.

    Original semantics are exactly:
        out[:, 0, :] = features
        out[:, h, :] = adj @ out[:, h-1, :]

    The old implementation filled this tensor row-by-row in Python.  This
    vectorized version performs the same sparse/dense matrix multiplications
    and whole-slice assignments, avoiding O(N*K) Python copies.  It also never
    densifies a sparse adjacency.
    """
    if int(K) < 0:
        raise ValueError("K must be non-negative")
    x = features.float().contiguous()
    adj_work = adj.coalesce().float() if isinstance(adj, torch.Tensor) and adj.is_sparse else adj.float()
    if isinstance(adj_work, torch.Tensor) and adj_work.device != x.device:
        x = x.to(adj_work.device)
    N, d = int(x.shape[0]), int(x.shape[1])
    nodes_features = torch.empty((N, int(K) + 1, d), dtype=x.dtype, device=x.device)
    nodes_features[:, 0, :] = x

    cur = x
    for i in range(int(K)):  # k-hop propagated features
        if isinstance(adj_work, torch.Tensor) and adj_work.is_sparse:
            cur = torch.sparse.mm(adj_work, cur)
        else:
            cur = torch.matmul(adj_work, cur)
        nodes_features[:, i + 1, :] = cur

    return nodes_features #[2708, 6, 1436]


def re_features_auto_device(adj, features, K, *, device: str = "auto"):
    """Run repeated sparse propagation on CUDA when requested, with CPU fallback.

    This preserves the exact senior definition:
        out[:, 0, :] = features
        out[:, h, :] = adj @ out[:, h-1, :]
    and returns a CPU tensor for cache compatibility.
    """
    def _run(adj_work, feat_work, label: str):
        x = feat_work.float().contiguous()
        adj_local = adj_work.coalesce().float() if isinstance(adj_work, torch.Tensor) and adj_work.is_sparse else adj_work.float()
        N, d = int(x.shape[0]), int(x.shape[1])
        out = torch.empty((N, int(K) + 1, d), dtype=x.dtype, device=x.device)
        out[:, 0, :] = x
        cur = x
        print(f"[global-tokens] {label} start N={N}, D={d}, hops={int(K)}", flush=True)
        for hop in range(int(K)):
            if isinstance(adj_local, torch.Tensor) and adj_local.is_sparse:
                cur = torch.sparse.mm(adj_local, cur)
            else:
                cur = torch.matmul(adj_local, cur)
            out[:, hop + 1, :] = cur
            print(f"[global-tokens] {label} finished hop {hop + 1}/{int(K)}", flush=True)
        return out

    want_cuda = str(device) in {"auto", "cuda"} and torch.cuda.is_available()
    if str(device) == "cpu":
        want_cuda = False
    if str(device) == "cuda" and not want_cuda:
        print("[global-tokens] CUDA requested but unavailable; using CPU", flush=True)
    if want_cuda:
        try:
            adj_dev = adj.coalesce().float().cuda()
            feat_dev = features.float().contiguous().cuda()
            out = _run(adj_dev, feat_dev, "cuda").detach().cpu().contiguous()
            del adj_dev, feat_dev
            torch.cuda.empty_cache()
            return out
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print("[global-tokens] CUDA sparse propagation OOM; falling back to CPU", flush=True)
            torch.cuda.empty_cache()
    return _run(adj.coalesce().float().cpu(), features.float().cpu(), "cpu").cpu().contiguous()





def preprocess_features(features):
    """Row-normalize feature matrix"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # 确保是稀疏矩阵并且数据类型正确
    if not isinstance(adj, sp.coo_matrix):
        adj = sp.coo_matrix(adj, dtype=np.float32)
    
    rowsum = np.array(adj.sum(1), dtype=np.float32).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()




def transform_coo_to_csr(adj): #[2708, 2708]
    row=adj._indices()[0] #[10556]
    col=adj._indices()[1] #[10556]
    data=adj._values() #[10556]
    shape=adj.size()  #[2708, 2708]
    adj=sp.csr_matrix((data, (row, col)), shape=shape)
    return adj #<2708x2708 sparse matrix of type '<class 'numpy.int64'>'
	            # with 10556 stored elements in Compressed Sparse Row format>

def transform_csr_to_coo(adj, size=None):
    adj = adj.tocoo()
    adj = torch.sparse.LongTensor(torch.LongTensor([adj.row.tolist(), adj.col.tolist()]),
                              torch.LongTensor(adj.data.astype(np.int32)),
                              torch.Size([size, size]))
    return adj

def transform_sp_csr_to_coo(adj, batch_size, node_num): #2708x2708 , 2708 , 2708
    # chunks
    node_index = [i for i in range(node_num)]# [0-2708]
    divide_index = [node_index[i:i+batch_size] for i in range(0, len(node_index), batch_size)] #按照batchsize划分节点索引 , 比如每组1000个节点

    # adj of each chunks, in the format of sp_csr
    print("start mini batch: adj of each chunks")
    adj_sp_csr = [adj[divide_index[i]][:, divide_index[i]] for i in range(len(divide_index))]# 按划分好的节点索引划分邻接矩阵
    print("start mini batch: minus adj of each chunks")
    minus_adj_sp_csr = [sp.csr_matrix(torch.ones(item.shape))-item for item in adj_sp_csr] #1减去邻接矩阵

    # adj_tensor_coo = [transform_csr_to_coo(item).to_dense() for item in adj_sp_csr]
    # minus_adj_tensor_coo = [transform_csr_to_coo(item).to_dense() for item in minus_adj_sp_csr]
    print("start mini batch: back to torch coo adj")
    adj_tensor_coo = [transform_csr_to_coo(adj_sp_csr[i], len(divide_index[i])).to_dense() for i in range(len(divide_index))]
    print("start mini batch: back to torch coo minus adj")
    minus_adj_tensor_coo = [transform_csr_to_coo(minus_adj_sp_csr[i], len(divide_index[i])).to_dense() for i in range(len(divide_index))]

    return adj_tensor_coo, minus_adj_tensor_coo  #都转化为密集矩阵了(不是靠索引定位的)




def epoch_evaluate(embedding_tensor, test_dataset): # (2708, 1024)
    
    query, labels = load_query_n_gt("./dataset/", test_dataset, embedding_tensor.shape[0]) #[100, 2708] , [100, 2708]

    # start = time.time()
    query_feature = torch.mm(query, embedding_tensor) # [100, 1024]  (query_num, embedding_dim)
    
    query_num = torch.sum(query, dim=1) #[100]  记录每次查询节点集里面有几个节点
    query_feature = torch.div(query_feature, query_num.view(-1, 1))  # [100, 1024]   类似归一化, 因为前面存在加好几个节点的emb
    
    # cosine similarity
    # print(query_feature.shape, embedding_tensor.shape)
    
    #算法3 
    query_score = cosin_similarity(query_feature, embedding_tensor) # [100, 2708]  <- [100, 1024], [2708, 1024]
    # query_score = torch.nn.functional.normalize(query_score, dim=1, p=1) # [100, 2708])

    # print("query_score.shape: ", query_score.shape)

    y_pred = torch.zeros_like(query_score) #[100, 2708]
    for i in tqdm(range(query_score.shape[0])): #range(0, 100)
        query_index = (torch.nonzero(query[i]).squeeze()).reshape(-1) #torch.Size([3])

        selected_candidates = GlobalSearch(query_index.tolist(), query_score[i].tolist())  #算法5
        # selected_candidates = threshold_search(query_score[i])
        for j in range(len(selected_candidates)):
            y_pred[i][selected_candidates[j]] = 1 #将预测的社区以独热编码的形式标记出来
        
    # end = time.time()
    # print("The global search using time: {:.4f}".format(end-start)) 
    # print("The global search using time (one query): {:.4f}".format((end-start)/query_feature.shape[0])) 
    f1_score = f1_score_calculation(y_pred.int(), labels.int())

    # print("F1 score by maximum weight gain: {:.4f}".format(f1_score))

    nmi, ari, jac = evaluation(y_pred.int(), labels.int())
    
    # print("NMI score by maximum weight gain: {:.4f}".format(nmi))
    # print("ARI score by maximum weight gain: {:.4f}".format(ari))
    # print("JAC score by maximum weight gain: {:.4f}".format(jac))

    print("F1:{:.4f} ".format(f1_score)+" NMI:{:.4f} ".format(nmi)+" ARI:{:.4f} ".format(ari)+" JAC:{:.4f} ".format(jac))
    return  f1_score


def f1_score_calculation(y_pred, y_true): #[100, 2708], [100, 2708]
    y_pred = y_pred.reshape(1, -1)
    y_true = y_true.reshape(1, -1)
    pre = torch.sum(torch.multiply(y_pred, y_true))/(torch.sum(y_pred)+1E-9)
    rec = torch.sum(torch.multiply(y_pred, y_true))/(torch.sum(y_true)+1E-9)
    F1 = 2 * pre * rec / (pre + rec+1E-9)
    # print("recall: ", rec, "pre: ", pre)
    return F1

def evaluation(comm_find, comm):
    
    comm_find = comm_find.reshape(-1)
    comm = comm.reshape(-1)
    
    return normalized_mutual_info_score(comm, comm_find), adjusted_rand_score(comm, comm_find), jaccard_score(comm, comm_find)


def load_query_n_gt(path, dataset, vec_length):
    # load query and ground truth
    query = []
    file_query = open(path + dataset + '/' + dataset + ".query", 'r')
    for line in file_query:
        vec = [0 for i in range(vec_length)]  # [2708]
        line = line.strip()
        line = line.split(" ")
        for i in line:  #本次查询节点(该line的)独热编码
            vec[int(i)] = 1  # [2708]
        query.append(vec)

    gt = []
    file_gt = open(path + dataset + '/' + dataset + ".gt", 'r')
    for line in file_gt:
        vec = [0 for i in range(vec_length)]
        line = line.strip()
        line = line.split(" ")
        
        for i in line:
            vec[int(i)] = 1
        gt.append(vec)
    
    return torch.Tensor(query), torch.Tensor(gt)



def cosin_similarity(query_tensor, emb_tensor):# [100, 1024], [2708, 1024]
    # similarity = torch.stack([torch.cosine_similarity(query_tensor[i], emb_tensor, dim=1) for i in range(len(query_tensor))], 0)
    similarity = torch.stack([torch.cosine_similarity(query_tensor[i].reshape(1, -1), emb_tensor, dim=1) for i in range(len(query_tensor))], 0)
    # print(similarity.shape)
    return similarity



class LossEarlyStopper:
    """
    Stop if train loss hasn't reached a NEW minimum for `patience` consecutive epochs.
    Save the LAST epoch model when stopping (not the best).
    """
    def __init__(self, patience=5, ckpt_path="last_epoch.pt", min_delta=0.0):
        self.patience = patience
        self.ckpt_path = ckpt_path
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.bad_epochs = 0

    def step(self, epoch_loss: float, model: torch.nn.Module, epoch: int):
        """
        Returns: should_stop (bool)
        Side-effect: if should_stop, save current(last) epoch model
        """
        improved = epoch_loss < (self.best_loss - self.min_delta)
        if improved:
            self.best_loss = epoch_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        if self.bad_epochs >= self.patience:
            # save LAST epoch model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_loss": self.best_loss,
                    "last_epoch_loss": epoch_loss,
                },
                self.ckpt_path
            )
            return True
        return False