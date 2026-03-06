"""
GNN レコメンドモデル

エンコーダー 2種:
  1. HeteroGATEncoder   ─ GATv2 (Attention ベース)
  2. HeteroAPPNPEncoder ─ APPNP  (Personalized PageRank ベース)

損失関数:
  BPR Loss (Bayesian Personalized Ranking)

┌──────────────────────────────────────────────────────┐
│  BPR と PPR の関係                                     │
│                                                       │
│  BPR  = 損失関数（「何を最適化するか」）                   │
│         pos のスコア > neg のスコア を最大化              │
│                                                       │
│  GAT  = 伝播方式 (Attention で隣接ノードの重みを学習)     │
│  APPNP = 伝播方式 (PPR で固定重みの伝播)                 │
│                                                       │
│  → BPR + GAT  : Attention ベース                       │
│  → BPR + APPNP: PPR ベース                             │
│    両方とも BPR で学習するが、情報の伝播方法が異なる        │
└──────────────────────────────────────────────────────┘

APPNP の仕組み (Predict then Propagate):
  1. Predict  : MLP で各ノードの初期予測 H^(0) を計算
  2. Propagate: PPR の反復式で K 回伝播
       H^(k+1) = (1 - α) · Â · H^(k) + α · H^(0)
     α = テレポート確率
       → 確率 α で元の予測に戻る（過平滑化を防ぐ）
       → 確率 (1-α) で隣接ノードの情報を取り入れる
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear


# =================================================================
# Encoder 1: HeteroGAT (Attention ベース)
# =================================================================

class HeteroGATEncoder(nn.Module):
    """ヘテロジニアス GATv2 エンコーダー"""

    def __init__(
        self,
        metadata: tuple,
        in_channels_dict: dict[str, int],
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.projections = nn.ModuleDict({
            nt: Linear(dim, hidden_channels)
            for nt, dim in in_channels_dict.items()
        })

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for layer_i in range(num_layers):
            is_last = (layer_i == num_layers - 1)
            per_head = (out_channels if is_last else hidden_channels) // heads

            conv_dict = {}
            for edge_type in metadata[1]:
                conv_dict[edge_type] = GATv2Conv(
                    in_channels=hidden_channels,
                    out_channels=per_head,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=False,
                )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

            norm_dim = per_head * heads
            self.norms.append(nn.ModuleDict({
                nt: nn.LayerNorm(norm_dim) for nt in metadata[0]
            }))

    def forward(self, x_dict, edge_index_dict):
        h = {nt: self.projections[nt](x) for nt, x in x_dict.items()}

        for i, (conv, norms) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index_dict)
            for nt in h_new:
                z = norms[nt](h_new[nt])
                if nt in h and h[nt].shape == z.shape:
                    z = z + h[nt]
                if i < self.num_layers - 1:
                    z = F.elu(z)
                    z = F.dropout(z, p=self.dropout, training=self.training)
                h_new[nt] = z
            h = h_new
        return h


# =================================================================
# Encoder 2: HeteroAPPNP (Personalized PageRank ベース)
# =================================================================

class HeteroAPPNPEncoder(nn.Module):
    """
    ヘテロジニアス APPNP エンコーダー

    「Predict then Propagate」:
      Step 1 (Predict):  MLP で初期 embedding H^(0) を生成
      Step 2 (Propagate): PPR 反復で K 回伝播
        H^(k+1) = (1 - α) · Â · H^(k) + α · H^(0)

    GAT との比較:
      GAT  : 各層に学習可能な Attention パラメータ → 表現力高い
      APPNP: 伝播は固定の PPR 重み → パラメータ少ない、遠くまで伝播可能

    Parameters
    ----------
    teleport_prob  : α — 元の予測に戻る確率 (0.1〜0.2 が一般的)
    num_iterations : K — PPR の反復回数 (5〜20)
    """

    def __init__(
        self,
        metadata: tuple,
        in_channels_dict: dict[str, int],
        hidden_channels: int = 128,
        out_channels: int = 64,
        teleport_prob: float = 0.15,
        num_iterations: int = 10,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.alpha = teleport_prob
        self.K = num_iterations
        self.dropout = dropout
        self.node_types = metadata[0]
        self.edge_types = metadata[1]

        # Predict: 各ノードタイプ用 MLP
        self.predict_mlps = nn.ModuleDict()
        for nt, in_dim in in_channels_dict.items():
            self.predict_mlps[nt] = nn.Sequential(
                nn.Linear(in_dim, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, out_channels),
            )

    def forward(self, x_dict, edge_index_dict):
        # --- Step 1: Predict ---
        h0 = {}
        for nt, x in x_dict.items():
            if nt in self.predict_mlps:
                h0[nt] = self.predict_mlps[nt](x)
            else:
                h0[nt] = x

        # --- Step 2: Propagate (PPR iterations) ---
        h = {nt: h0[nt].clone() for nt in h0}

        for _k in range(self.K):
            h_new = {nt: torch.zeros_like(h[nt]) for nt in h}
            neighbor_count = {
                nt: torch.zeros(h[nt].shape[0], 1, device=h[nt].device)
                for nt in h
            }

            for edge_type in self.edge_types:
                src_type, _, dst_type = edge_type
                ei = edge_index_dict.get(edge_type)
                if ei is None or ei.shape[1] == 0:
                    continue
                if src_type not in h or dst_type not in h_new:
                    continue

                src_idx = ei[0]
                dst_idx = ei[1]
                src_features = h[src_type][src_idx]

                h_new[dst_type].index_add_(0, dst_idx, src_features)
                ones = torch.ones(src_idx.shape[0], 1, device=ei.device)
                neighbor_count[dst_type].index_add_(0, dst_idx, ones)

            for nt in h_new:
                count = neighbor_count[nt].clamp(min=1)
                h_agg = h_new[nt] / count
                # PPR 更新: H^(k+1) = (1-α) * Â*H^(k) + α * H^(0)
                h[nt] = (1 - self.alpha) * h_agg + self.alpha * h0[nt]

        return h


# =================================================================
# Link Predictor
# =================================================================

class LinkPredictor(nn.Module):
    """user ⊕ item → MLP → score"""

    def __init__(self, embed_dim: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z_u: torch.Tensor, z_i: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_u, z_i], dim=-1)).squeeze(-1)


# =================================================================
# Full Model
# =================================================================

class GNNRecommender(nn.Module):
    """
    GNN レコメンドモデル (Encoder + BPR)

    encoder_type="gat"   → HeteroGATEncoder   (Attention)
    encoder_type="appnp" → HeteroAPPNPEncoder  (Personalized PageRank)
    """

    def __init__(
        self,
        metadata: tuple,
        in_channels_dict: dict[str, int],
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        encoder_type: str = "gat",
        teleport_prob: float = 0.15,
        num_iterations: int = 10,
    ):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == "appnp":
            self.encoder = HeteroAPPNPEncoder(
                metadata, in_channels_dict,
                hidden_channels, out_channels,
                teleport_prob=teleport_prob,
                num_iterations=num_iterations,
                dropout=dropout,
            )
        else:
            self.encoder = HeteroGATEncoder(
                metadata, in_channels_dict,
                hidden_channels, out_channels,
                num_layers, heads, dropout,
            )

        self.predictor = LinkPredictor(out_channels, hidden_channels, dropout)

    def encode(self, x_dict, edge_index_dict):
        return self.encoder(x_dict, edge_index_dict)

    def predict_score(self, z_u, z_i):
        return self.predictor(z_u, z_i)

    def forward(self, x_dict, edge_index_dict, user_idx, pos_idx, neg_idx):
        """BPR Loss: -log σ(score_pos - score_neg)"""
        z = self.encode(x_dict, edge_index_dict)
        z_u = z["user"][user_idx]
        z_pos = z["item"][pos_idx]
        z_neg = z["item"][neg_idx]

        pos_score = self.predict_score(z_u, z_pos)
        neg_score = self.predict_score(z_u, z_neg)

        return -F.logsigmoid(pos_score - neg_score).mean()

    @torch.no_grad()
    def get_embeddings(self, x_dict, edge_index_dict):
        z = self.encode(x_dict, edge_index_dict)
        return z["user"], z["item"]
