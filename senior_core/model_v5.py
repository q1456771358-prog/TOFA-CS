"""V5 adapter for the senior OFA-CS backbone.

This file intentionally starts from the uploaded senior model semantics:
``global hop trajectory -> TransformerBlock -> node/neighbor representation ->
diffusion refinement -> query mean cosine -> GlobalSearch``.

V5 changes in this adapter:
- no V4 multifrequency/homo-hetero selector is imported;
- diffusion uses ``ofacs.guided_diffusion`` rather than V4 identity correction;
- deterministic ``encode_refined`` is available for fixed inference;
- legacy link/prototype losses remain optional auxiliary losses only;
- query-boundary loss is implemented in ``ofacs.query_boundary_loss`` and should be
  called by the training loop, not hidden inside the encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .layer import TransformerBlock
from ofacs.errors import V5ProtocolError
from ofacs.guided_diffusion import GuidedDiffusionConfig, GuidedDiffusionRefiner


@dataclass(frozen=True)
class V5ModelForwardOutput:
    node_tensor: torch.Tensor
    neighbor_tensor: torch.Tensor
    raw_embedding: torch.Tensor
    refined_embedding: torch.Tensor
    diffusion_loss: torch.Tensor | None = None
    diffusion_report: dict[str, object] | None = None


def _cfg(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


class PretrainModelV5(nn.Module):
    """Senior backbone with V5 guided diffusion hooks."""

    def __init__(self, input_dim: int, config: Any):
        super().__init__()
        self.input_dim = int(input_dim)
        self.config = config
        self.hidden_dim = int(_cfg(config, "hidden_dim", 512))
        self.hops = int(_cfg(config, "hops", 4))
        self.readout = str(_cfg(config, "readout", "mean"))
        self.condition_dim = int(_cfg(config, "condition_dim", 16))
        self.diffusion_T = int(_cfg(config, "diffusion_T", 200))
        self.guidance_eta = float(_cfg(config, "guidance_eta", 0.50))
        self.legacy_link_loss_max_nodes = int(_cfg(config, "legacy_link_loss_max_nodes", 4096))

        self.encoder = TransformerBlock(
            hops=self.hops,
            input_dim=self.input_dim,
            n_layers=int(_cfg(config, "n_layers", 1)),
            num_heads=int(_cfg(config, "n_heads", 8)),
            hidden_dim=self.hidden_dim,
            dropout_rate=float(_cfg(config, "dropout", 0.4)),
            attention_dropout_rate=float(_cfg(config, "attention_dropout", 0.4)),
        )

        diff_cfg = GuidedDiffusionConfig(
            latent_dim=self.hidden_dim,
            base_cond_dim=self.hidden_dim,
            structure_cond_dim=self.condition_dim,
            diffusion_T=self.diffusion_T,
            guidance_eta=self.guidance_eta,
            hidden_dim=int(_cfg(config, "diffusion_hidden_dim", 256)),
            time_emb_dim=int(_cfg(config, "diffusion_time_emb_dim", 128)),
        )
        self.diffusion_node = GuidedDiffusionRefiner(diff_cfg)
        self.diffusion_neighbor = GuidedDiffusionRefiner(diff_cfg)

    def _pool_neighbor(self, neighbor_tensor: torch.Tensor) -> torch.Tensor:
        """Pool hop-level neighbor trajectory to ``[B,H]`` without PyG dependency."""

        if neighbor_tensor.ndim != 3:
            raise V5ProtocolError(f"neighbor_tensor must be [B,hops,H], got {tuple(neighbor_tensor.shape)}")
        if self.readout == "sum":
            return neighbor_tensor.sum(dim=1)
        if self.readout == "mean":
            # Senior code calls PyG global_mean_pool(x, batch=torch.tensor([0]))
            # on a 3D [N,hops,H] tensor. With that exact call, PyG aggregates
            # over the hop dimension but does not divide by hop count, so
            # "mean" is effectively sum. Keep this bug-compatible semantics.
            return neighbor_tensor.sum(dim=1)
        if self.readout == "max":
            return neighbor_tensor.max(dim=1).values
        raise V5ProtocolError(f"invalid readout: {self.readout}")

    def encode_raw(self, global_sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode global hop trajectory and return node/neighbor/raw concat embeddings."""

        node_tensor, neighbor_tensor = self.encoder(global_sequence)
        if node_tensor.ndim != 3 or node_tensor.shape[1] != 1:
            raise V5ProtocolError(f"encoder node output must be [B,1,H], got {tuple(node_tensor.shape)}")
        node_tensor = node_tensor.squeeze(1)
        neighbor_tensor = self._pool_neighbor(neighbor_tensor)
        raw_embedding = torch.cat([node_tensor, neighbor_tensor], dim=-1)
        return node_tensor, neighbor_tensor, raw_embedding

    def _expand_condition(self, condition: torch.Tensor | None, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if condition is None:
            return torch.zeros(batch_size, self.condition_dim, device=device, dtype=dtype)
        cond = condition.to(device=device, dtype=dtype)
        if cond.ndim == 1:
            cond = cond.unsqueeze(0).expand(batch_size, -1)
        if cond.ndim != 2 or cond.shape[0] != batch_size or cond.shape[1] != self.condition_dim:
            raise V5ProtocolError(f"condition must be [B,{self.condition_dim}] or [{self.condition_dim}], got {tuple(cond.shape)}")
        return cond

    def _one_step_refine_from_states(
        self,
        node_tensor: torch.Tensor,
        neighbor_tensor: torch.Tensor,
        *,
        structure_condition: torch.Tensor | None = None,
        t_eval: int | None = None,
        noise_mode: str = "random",
    ) -> torch.Tensor:
        """One-step x0 estimate matching the training embedding path."""

        cond = self._expand_condition(structure_condition, node_tensor.shape[0], device=node_tensor.device, dtype=node_tensor.dtype)
        b = node_tensor.shape[0]
        if t_eval is None:
            t = torch.randint(0, self.diffusion_T, (b,), device=node_tensor.device, dtype=torch.long)
            t_n = torch.randint(0, self.diffusion_T, (b,), device=neighbor_tensor.device, dtype=torch.long)
        else:
            fixed_t = max(0, min(int(t_eval), self.diffusion_T - 1))
            t = torch.full((b,), fixed_t, device=node_tensor.device, dtype=torch.long)
            t_n = torch.full((b,), fixed_t, device=neighbor_tensor.device, dtype=torch.long)
        if noise_mode == "zero":
            noise_node = torch.zeros_like(node_tensor)
            noise_neighbor = torch.zeros_like(neighbor_tensor)
        elif noise_mode == "random":
            noise_node = None
            noise_neighbor = None
        else:
            raise V5ProtocolError(f"unknown noise_mode: {noise_mode}")
        zt_node, _ = self.diffusion_node.q_sample(node_tensor, t, noise=noise_node)
        eps_node = self.diffusion_node.eps_model(zt_node, t, node_tensor.detach(), cond)
        sqrt_acp = self.diffusion_node.sqrt_alphas_cumprod[t].unsqueeze(-1).to(dtype=node_tensor.dtype, device=node_tensor.device)
        sqrt_1m = self.diffusion_node.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1).to(dtype=node_tensor.dtype, device=node_tensor.device)
        node_hat = (zt_node - sqrt_1m * eps_node) / torch.clamp(sqrt_acp, min=1e-12)

        zt_neighbor, _ = self.diffusion_neighbor.q_sample(neighbor_tensor, t_n, noise=noise_neighbor)
        eps_neighbor = self.diffusion_neighbor.eps_model(zt_neighbor, t_n, neighbor_tensor.detach(), cond)
        sqrt_acp_n = self.diffusion_neighbor.sqrt_alphas_cumprod[t_n].unsqueeze(-1).to(dtype=neighbor_tensor.dtype, device=neighbor_tensor.device)
        sqrt_1m_n = self.diffusion_neighbor.sqrt_one_minus_alphas_cumprod[t_n].unsqueeze(-1).to(dtype=neighbor_tensor.dtype, device=neighbor_tensor.device)
        neighbor_hat = (zt_neighbor - sqrt_1m_n * eps_neighbor) / torch.clamp(sqrt_acp_n, min=1e-12)
        return torch.cat([node_hat, neighbor_hat], dim=-1)

    def diffusion_training_loss(
        self,
        node_tensor: torch.Tensor,
        neighbor_tensor: torch.Tensor,
        *,
        structure_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        """Source diffusion noise-prediction loss for node and neighbor states."""

        cond = self._expand_condition(structure_condition, node_tensor.shape[0], device=node_tensor.device, dtype=node_tensor.dtype)
        loss_node, rep_node = self.diffusion_node.training_loss(node_tensor, base_cond=node_tensor.detach(), structure_cond=cond)
        loss_neighbor, rep_neighbor = self.diffusion_neighbor.training_loss(neighbor_tensor, base_cond=neighbor_tensor.detach(), structure_cond=cond)
        report = {
            "node": rep_node.to_dict(),
            "neighbor": rep_neighbor.to_dict(),
            "total_eps_loss": float((loss_node + loss_neighbor).detach().cpu().item()),
            "labels_used": False,
            "ground_truth_used": False,
            "unseen_training_used": False,
        }
        return loss_node + loss_neighbor, report

    @torch.no_grad()
    def encode_refined(
        self,
        global_sequence: torch.Tensor,
        *,
        structure_condition: torch.Tensor | None = None,
        t_eval: int | None = None,
        stochastic: bool = False,
        disable_diffusion: bool = False,
        ratio_guidance_node: torch.nn.Module | None = None,
        ratio_guidance_neighbor: torch.nn.Module | None = None,
        ratio_guidance_weight: float = 0.0,
        ratio_guidance_clip: float = 5.0,
        adaptive_ratio_guidance: bool = False,
        ratio_guidance_confidence_scale: float = 1.0,
        ratio_guidance_min_mult: float = 0.25,
        ratio_guidance_max_mult: float = 1.25,
        guidance_diagnostics: object | None = None,
        guidance_runtime_timer: object | None = None,
    ) -> torch.Tensor:
        """Encode refined embeddings for fixed inference.

        ``disable_diffusion=True`` is a controlled ablation switch.  Formal V5-Diff
        inference should keep diffusion enabled and deterministic.  Passing
        ``ratio_guidance_node/neighbor`` activates the DiffGDA-style
        density-ratio guided reverse denoising path.
        """

        node_tensor, neighbor_tensor, raw = self.encode_raw(global_sequence)
        if disable_diffusion:
            return raw
        cond = self._expand_condition(structure_condition, node_tensor.shape[0], device=node_tensor.device, dtype=node_tensor.dtype)
        node_hat = self.diffusion_node.reverse_denoise(
            node_tensor,
            base_cond=node_tensor,
            structure_cond=cond,
            t_eval=t_eval,
            stochastic=stochastic,
            ratio_guidance=ratio_guidance_node,
            ratio_guidance_weight=ratio_guidance_weight,
            ratio_guidance_clip=ratio_guidance_clip,
            adaptive_ratio_guidance=adaptive_ratio_guidance,
            ratio_guidance_confidence_scale=ratio_guidance_confidence_scale,
            ratio_guidance_min_mult=ratio_guidance_min_mult,
            ratio_guidance_max_mult=ratio_guidance_max_mult,
            guidance_diagnostics=guidance_diagnostics,
            guidance_runtime_timer=guidance_runtime_timer,
            guidance_branch="node",
        )
        neighbor_hat = self.diffusion_neighbor.reverse_denoise(
            neighbor_tensor,
            base_cond=neighbor_tensor,
            structure_cond=cond,
            t_eval=t_eval,
            stochastic=stochastic,
            ratio_guidance=ratio_guidance_neighbor,
            ratio_guidance_weight=ratio_guidance_weight,
            ratio_guidance_clip=ratio_guidance_clip,
            adaptive_ratio_guidance=adaptive_ratio_guidance,
            ratio_guidance_confidence_scale=ratio_guidance_confidence_scale,
            ratio_guidance_min_mult=ratio_guidance_min_mult,
            ratio_guidance_max_mult=ratio_guidance_max_mult,
            guidance_diagnostics=guidance_diagnostics,
            guidance_runtime_timer=guidance_runtime_timer,
            guidance_branch="neighbor",
        )
        return torch.cat([node_hat, neighbor_hat], dim=-1)

    @torch.no_grad()
    def encode_one_step_estimate(
        self,
        global_sequence: torch.Tensor,
        *,
        structure_condition: torch.Tensor | None = None,
        t_eval: int | None = None,
        noise_mode: str = "zero",
    ) -> torch.Tensor:
        """Deterministic diagnostic view of the training one-step path."""

        node_tensor, neighbor_tensor, _ = self.encode_raw(global_sequence)
        return self._one_step_refine_from_states(
            node_tensor,
            neighbor_tensor,
            structure_condition=structure_condition,
            t_eval=t_eval,
            noise_mode=noise_mode,
        )

    def forward(
        self,
        local_sequence: torch.Tensor | None,
        global_sequence: torch.Tensor,
        *,
        structure_condition: torch.Tensor | None = None,
        compute_diffusion_loss: bool = True,
        disable_diffusion_for_embedding: bool = False,
    ) -> V5ModelForwardOutput:
        # local_sequence is retained only for API compatibility with the senior code.
        node_tensor, neighbor_tensor, raw_embedding = self.encode_raw(global_sequence)
        diffusion_loss = None
        diffusion_report = None
        if compute_diffusion_loss:
            diffusion_loss, diffusion_report = self.diffusion_training_loss(
                node_tensor,
                neighbor_tensor,
                structure_condition=structure_condition,
            )
        if disable_diffusion_for_embedding:
            refined = raw_embedding
        else:
            # Differentiable one-step x0 estimates for query-boundary loss.
            refined = self._one_step_refine_from_states(
                node_tensor,
                neighbor_tensor,
                structure_condition=structure_condition,
                noise_mode="random",
            )
        return V5ModelForwardOutput(
            node_tensor=node_tensor,
            neighbor_tensor=neighbor_tensor,
            raw_embedding=raw_embedding,
            refined_embedding=refined,
            diffusion_loss=diffusion_loss,
            diffusion_report=diffusion_report,
        )

    def legacy_transformer_loss(
        self,
        node_tensor: torch.Tensor,
        neighbor_tensor: torch.Tensor,
        pos_samples: torch.Tensor,
        neg_samples: torch.Tensor,
        adj_: torch.Tensor | None = None,
        minus_adj: torch.Tensor | None = None,
        *,
        enable_link_loss: bool = False,
        gamma: float = 0.2,
    ) -> torch.Tensor:
        """Optional senior auxiliary loss, not the V5 main objective.

        Link loss is disabled by default because it uses dense pairwise similarity.
        When enabled, it refuses graphs larger than ``legacy_link_loss_max_nodes``.
        """

        loss_cl = self.contrastive_loss_with_prototypes(neighbor_tensor, pos_samples, neg_samples, gamma=gamma)
        if not enable_link_loss:
            return loss_cl
        n = int(node_tensor.shape[0])
        if n > self.legacy_link_loss_max_nodes:
            raise V5ProtocolError("legacy link loss would construct full [N,N]; disabled for this graph size")
        if adj_ is None or minus_adj is None:
            raise V5ProtocolError("adj_ and minus_adj are required when enable_link_loss=True")
        pairwise_similarity = node_tensor @ node_tensor.t()
        link_loss = (minus_adj.to(pairwise_similarity.device) * pairwise_similarity - adj_.to(pairwise_similarity.device) * pairwise_similarity).sum() / float(n * n)
        return loss_cl + link_loss

    def contrastive_loss_with_prototypes(
        self,
        neighbor_tensor: torch.Tensor,
        pos_samples: torch.Tensor,
        neg_samples: torch.Tensor,
        gamma: float = 0.2,
        batch_idx: torch.Tensor | None = None,
        normalize: bool = True,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Senior prototype loss retained as weak auxiliary regularization."""

        h = neighbor_tensor
        device = h.device
        if h.ndim != 2:
            raise V5ProtocolError("neighbor_tensor must be [N,H]")
        v_idx = torch.arange(h.size(0), device=device) if batch_idx is None else batch_idx.to(device)
        pos_ids = pos_samples[v_idx].to(device)
        neg_ids = neg_samples[v_idx].to(device)
        h_v = h[v_idx]
        h_pos = h[pos_ids].mean(dim=1)
        h_neg = h[neg_ids].mean(dim=1)
        if normalize:
            h_v = F.normalize(h_v, p=2, dim=-1, eps=eps)
            h_pos = F.normalize(h_pos, p=2, dim=-1, eps=eps)
            h_neg = F.normalize(h_neg, p=2, dim=-1, eps=eps)
        pos_logit = (h_v * h_pos).sum(dim=-1) / float(gamma)
        neg_logit = (h_v * h_neg).sum(dim=-1) / float(gamma)
        logits = torch.stack([pos_logit, neg_logit], dim=1)
        return (-pos_logit + torch.logsumexp(logits, dim=1)).mean()


# Backward-compatible class name for simple imports.  New code should prefer
# PretrainModelV5 to make the stage explicit.
PretrainModel = PretrainModelV5
