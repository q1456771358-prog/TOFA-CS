import torch
import math
import torch.nn as nn
from layer import TransformerBlock
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GCNConv
import torch.nn.functional as F


class PretrainModel(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.config = config

        self.Linear1 = nn.Linear(input_dim, self.config.hidden_dim)
        self.encoder = TransformerBlock(hops=config.hops, 
                        input_dim=input_dim, 
                        n_layers=config.n_layers,
                        num_heads=config.n_heads,
                        hidden_dim=config.hidden_dim,
                        dropout_rate=config.dropout,
                        attention_dropout_rate=config.attention_dropout)
        if config.readout == "sum":
            self.readout = global_add_pool
        elif config.readout == "mean":
            self.readout = global_mean_pool
        elif config.readout == "max":
            self.readout = global_max_pool
        else:
            raise ValueError("Invalid pooling type.")
        
        self.marginloss = nn.MarginRankingLoss(0.5)
                # --- diffusion part for node and neighbor ---
        self.latent_dim_node = config.latent_dim_node
        self.latent_dim_neighbor = config.latent_dim_neighbor
        self.T = config.diffusion_T

        # eps models
        self.eps_model_node = EpsPredictor(self.latent_dim_node, config.hidden_dim)
        self.eps_model_neighbor = EpsPredictor(self.latent_dim_neighbor, config.hidden_dim)

        # diffusion buffers
        betas = torch.linspace(1e-4, 2e-2, self.T)
        alphas = 1. - betas
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - self.alphas_cumprod))





    def q_sample(self, z0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(z0)
        sqrt_acp = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_1m = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_acp * z0 + sqrt_1m * noise, noise
    
    def test_forward(self, local_tokens, global_tokens, t_eval: int = 50, stochastic: bool = False):


        # --- encode and pool ---
        node_tensor, neighbor_tensor = self.encoder(global_tokens)  # (B,1,H), (B,hops,H)
        neighbor_tensor = self.readout(neighbor_tensor, torch.tensor([0]).to(self.config.device))  # (B,1,H)
        node_tensor = node_tensor.squeeze(1).to(self.config.device)        # (B,H)
        neighbor_tensor = neighbor_tensor.squeeze(1).to(self.config.device)  # (B,H)

        B = node_tensor.size(0)


        z0_node = node_tensor                           # (B, H)
        z0_neighbor = neighbor_tensor
        con_tensor = torch.cat([node_tensor, neighbor_tensor], dim=-1)

        # choose noise level
        if t_eval is None:
            t_eval = self.T - 1
        t_eval = int(t_eval)
        t_node = torch.full((B,), t_eval, device=self.config.device, dtype=torch.long)
        t_neighbor = torch.full((B,), t_eval, device=self.config.device, dtype=torch.long)

        # --- forward q_sample: closed-form add noise ---
        sqrt_acp_node = self.sqrt_alphas_cumprod[t_node].unsqueeze(-1)   # (B,1)
        sqrt_1m_node  = self.sqrt_one_minus_alphas_cumprod[t_node].unsqueeze(-1)
        eps_node = torch.randn_like(z0_node)
        zt_node = sqrt_acp_node * z0_node + sqrt_1m_node * eps_node     # (B, Dn)

        sqrt_acp_neighbor = self.sqrt_alphas_cumprod[t_neighbor].unsqueeze(-1)
        sqrt_1m_neighbor  = self.sqrt_one_minus_alphas_cumprod[t_neighbor].unsqueeze(-1)
        eps_neighbor = torch.randn_like(z0_neighbor)
        zt_neighbor = sqrt_acp_neighbor * z0_neighbor + sqrt_1m_neighbor * eps_neighbor  # (B, Dm)

        # --- reverse deterministic loop from t_eval down to 0 ---
        z_node = zt_node
        z_neighbor = zt_neighbor

        for step in reversed(range(0, t_eval + 1)):
            t_step = torch.full((B,), step, device=self.config.device, dtype=torch.long)

            # NODE: predict epsilon
            eps_pred_node = self.eps_model_node(z_node, t_step, node_tensor)  # (B, Dn)
            beta_t = self.betas[step].to(self.config.device)
            alpha_t = self.alphas[step].to(self.config.device)
            alpha_cum_t = self.alphas_cumprod[step].to(self.config.device)

            mu_node = (1.0 / torch.sqrt(alpha_t)) * (z_node - (beta_t / torch.sqrt(1.0 - alpha_cum_t)) * eps_pred_node)

            if step > 0:
                # posterior variance tilde_beta_t
                alpha_cum_prev = self.alphas_cumprod[step - 1].to(self.config.device) if step - 1 >= 0 else torch.tensor(1.0, device=self.config.device)
                tilde_beta = ((1.0 - alpha_cum_prev) / (1.0 - alpha_cum_t)) * beta_t
                if stochastic:
                    z_node = mu_node + torch.sqrt(tilde_beta) * torch.randn_like(z_node)
                else:
                    z_node = mu_node
            else:
                z_node = mu_node  # final

            # NEIGHBOR: predict epsilon
            eps_pred_neighbor = self.eps_model_neighbor(z_neighbor, t_step, neighbor_tensor)
            beta_t_n = self.betas[step].to(self.config.device)
            alpha_t_n = self.alphas[step].to(self.config.device)
            alpha_cum_t_n = self.alphas_cumprod[step].to(self.config.device)

            mu_neighbor = (1.0 / torch.sqrt(alpha_t_n)) * (z_neighbor - (beta_t_n / torch.sqrt(1.0 - alpha_cum_t_n)) * eps_pred_neighbor)

            if step > 0:
                alpha_cum_prev_n = self.alphas_cumprod[step - 1].to(self.config.device) if step - 1 >= 0 else torch.tensor(1.0, device=self.config.device)
                tilde_beta_n = ((1.0 - alpha_cum_prev_n) / (1.0 - alpha_cum_t_n)) * beta_t_n
                if stochastic:
                    z_neighbor = mu_neighbor + torch.sqrt(tilde_beta_n) * torch.randn_like(z_neighbor)
                else:
                    z_neighbor = mu_neighbor
            else:
                z_neighbor = mu_neighbor

        # final denoised latent estimates
        z0_node_hat = z_node
        z0_neighbor_hat = z_neighbor

        # optionally, if you trained a decoder to map latent back to hidden dim, call it here.
        # final_concat in latent space
        final_concat = torch.cat([z0_node_hat, z0_neighbor_hat], dim=-1)

        return final_concat




    def forward(self, local_sequence,global_sequence):
        # --- transformer encoding ---
        node_tensor, neighbor_tensor = self.encoder(global_sequence) # (B,1,512),(B,hops,512)
        neighbor_tensor = self.readout(neighbor_tensor, torch.tensor([0]).to(self.config.device)) # (B,1,512)
        node_tensor = node_tensor.squeeze()      # (B,512)
        neighbor_tensor = neighbor_tensor.squeeze()  # (B,512)

        z0_node = node_tensor
        z0_neighbor = neighbor_tensor

        # z0_node = self.latent_proj_node(node_tensor) # [3327, 128]
        # z0_neighbor = self.latent_proj_neighbor(neighbor_tensor) # (B, latent_dim_neighbor)

        # --- diffusion for node ---
        t_node = torch.randint(0, self.T, (z0_node.size(0),), device=z0_node.device).long()
        zt_node, noise_node = self.q_sample(z0_node, t_node)
        noise_pred_node = self.eps_model_node(zt_node, t_node, node_tensor)

        # --- diffusion for neighbor ---
        t_neighbor = torch.randint(0, self.T, (z0_neighbor.size(0),), device=z0_neighbor.device).long()
        zt_neighbor, noise_neighbor = self.q_sample(z0_neighbor, t_neighbor)
        noise_pred_neighbor = self.eps_model_neighbor(zt_neighbor, t_neighbor, neighbor_tensor)

        # --- concatenate after diffusion ---
        # con_tensor = torch.cat([z0_node, z0_neighbor], dim=-1)  # (B, latent_dim_node+latent_dim_neighbor)
        # return z0_node, z0_neighbor, con_tensor, noise_node, noise_pred_node, noise_neighbor, noise_pred_neighbor

        con_tensor = torch.cat([node_tensor, neighbor_tensor], dim=-1)

        X_prime=0
        # X_prime = self.RelationalSupportAdapter(con_tensor, few_shot_com)


        return node_tensor, neighbor_tensor, con_tensor, X_prime, noise_node, noise_pred_node, noise_neighbor, noise_pred_neighbor
    



    def Transformer_loss(self, node_tensor, neighbor_tensor, pos_samples, neg_samples, adj_, minus_adj): # [2708, 512], [2708, 512], [2708, 2708], [2708, 2708]
        

        TotalLoss = 0.0
        pairwise_similary = torch.mm(node_tensor, node_tensor.t())
        link_loss = minus_adj.multiply(pairwise_similary)-adj_.multiply(pairwise_similary) #公式6
        # link_loss = torch.abs(torch.sum(link_loss))/(adj_.shape[0])
        link_loss = torch.sum(link_loss)/(adj_.shape[0]*adj_.shape[0])
        loss_cl = self.contrastive_loss_with_prototypes(
            neighbor_tensor=neighbor_tensor,   # [2708,512]
            pos_samples=pos_samples,           # [2708,10]
            neg_samples=neg_samples,           # [2708,10]
            gamma=0.2
        )

        # TotalLoss += 0.001*link_loss
        TotalLoss = loss_cl + link_loss
        # TotalLoss = link_loss

        return TotalLoss
    
    def contrastive_loss_with_prototypes(
        self,
        neighbor_tensor: torch.Tensor,   # [N, d]
        pos_samples: torch.Tensor,       # [N, K]
        neg_samples: torch.Tensor,       # [N, K]
        gamma: float = 0.2,
        batch_idx: torch.Tensor = None,  # optional: [B] node ids
        normalize: bool = True,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        Implements:
        h_v^+ = mean_{u in P_v} h_u
        h_v^- = mean_{u in N_v} h_u
        L = (1/|V|) sum_v  -log  exp(h_v^T h_v^+ / gamma) / (exp(...) + exp(h_v^T h_v^- / gamma))

        Args:
        neighbor_tensor: node representations H, shape [N, d]
        pos_samples: positive node ids per node, shape [N, K]
        neg_samples: negative node ids per node, shape [N, K]
        gamma: temperature
        batch_idx: if provided, compute loss only for these nodes
        normalize: whether to L2-normalize representations before dot products

        Returns:
        scalar loss
        """
        H = neighbor_tensor
        device = H.device

        # choose nodes to compute loss on
        if batch_idx is None:
            v_idx = torch.arange(H.size(0), device=device)
        else:
            v_idx = batch_idx.to(device)

        # gather h_v: [B, d]
        h_v = H[v_idx]

        # gather positive/negative ids for selected v: [B, K]
        pos_ids = pos_samples[v_idx].to(device)
        neg_ids = neg_samples[v_idx].to(device)

        # gather positive/negative node embeddings: [B, K, d]
        h_pos = H[pos_ids]   # advanced indexing
        h_neg = H[neg_ids]

        # prototypes: [B, d]
        h_v_pos = h_pos.mean(dim=1)
        h_v_neg = h_neg.mean(dim=1)

        if normalize:
            h_v = F.normalize(h_v, p=2, dim=-1, eps=eps)
            h_v_pos = F.normalize(h_v_pos, p=2, dim=-1, eps=eps)
            h_v_neg = F.normalize(h_v_neg, p=2, dim=-1, eps=eps)

        # logits: [B]
        pos_logit = (h_v * h_v_pos).sum(dim=-1) / gamma
        neg_logit = (h_v * h_v_neg).sum(dim=-1) / gamma

        # stable form of: -log exp(pos)/(exp(pos)+exp(neg))  ==  -pos + logsumexp([pos,neg])
        logits = torch.stack([pos_logit, neg_logit], dim=1)  # [B, 2]
        loss = (-pos_logit + torch.logsumexp(logits, dim=1)).mean()

        return loss




# --- sinusoidal time embedding ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = torch.exp(-math.log(10000) * torch.arange(0, half, device=device).float() / half)
        emb = t.unsqueeze(-1).float() * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return emb

# --- epsilon predictor ---
class EpsPredictor(nn.Module):
    def __init__(self, latent_dim, cond_dim, time_emb_dim=128, hidden=256):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden)
        self.cond_proj = nn.Linear(cond_dim, hidden)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim)
        )
    def forward(self, z_t, t, cond):
        te = self.time_emb(t)
        th = self.time_proj(te)
        ch = self.cond_proj(cond)
        ctx = torch.tanh(th + ch)
        inp = torch.cat([z_t, ctx], dim=-1)
        return self.net(inp)
