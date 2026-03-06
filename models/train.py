"""
学習パイプライン (Multi-Behavior 版)

改善点:
  - 購入・閲覧・お気に入りの 3 種類の行動を全て学習ターゲットに使用
  - 行動タイプごとに重みを設定 (購入 > お気に入り > 閲覧)
  - 評価は購入エッジの val/test で行う（最終目標は購入予測）

  ┌─────────────────────────────────────────────┐
  │  従来: 購入エッジだけで学習                      │
  │    train = 8,000 edges                       │
  │                                               │
  │  改善: 3種の行動を統合して学習                    │
  │    train = 購入 20,000 + 閲覧 54,000            │
  │          + お気に入り 10,800 = 84,800 edges      │
  │    → 学習データ約 10 倍！                        │
  │                                               │
  │  重み付け:                                      │
  │    purchased  : 3.0 (最も重要)                  │
  │    favorited  : 2.0 (興味の明示的表明)            │
  │    viewed     : 1.0 (暗黙的シグナル)             │
  └─────────────────────────────────────────────┘
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import HeteroData

from config import ModelConfig
from models.gnn_model import GNNRecommender


# 行動タイプごとの BPR Loss 重み
BEHAVIOR_WEIGHTS = {
    "purchased": 3.0,   # 最重要：実際に購入
    "favorited": 2.0,   # 明示的な興味
    "viewed":    1.0,   # 暗黙的な興味
}


class Trainer:
    """Multi-Behavior GNN モデルの学習を管理"""

    def __init__(
        self,
        model: GNNRecommender,
        data: HeteroData,
        config: ModelConfig,
    ):
        self.model = model
        self.data = data
        self.cfg = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=1e-6,
        )

        self.n_items = data["item"].num_nodes
        self._split_edges()
        self._build_training_data()

    # --------------------------------------------------
    # Edge splitting
    # --------------------------------------------------

    def _split_edges(self) -> None:
        """購入エッジを train/val/test に分割（評価は購入のみ）"""
        ei = self.data["user", "purchased", "item"].edge_index
        n = ei.shape[1]
        perm = torch.randperm(n)

        n_val = int(n * self.cfg.val_ratio)
        n_test = int(n * self.cfg.test_ratio)

        self.purchase_train = ei[:, perm[: n - n_val - n_test]]
        self.val_edges = ei[:, perm[n - n_val - n_test : n - n_test]]
        self.test_edges = ei[:, perm[n - n_test :]]

        print(f"  📊 購入エッジ分割: train={self.purchase_train.shape[1]:,}  "
              f"val={self.val_edges.shape[1]:,}  test={self.test_edges.shape[1]:,}")

    def _build_training_data(self) -> None:
        """
        3種の行動エッジを統合して学習データを構築
        各ペアに行動タイプの重みを付与
        """
        all_users = []
        all_items = []
        all_weights = []

        # --- 購入 (train 分のみ) ---
        n_p = self.purchase_train.shape[1]
        all_users.append(self.purchase_train[0])
        all_items.append(self.purchase_train[1])
        all_weights.append(torch.full((n_p,), BEHAVIOR_WEIGHTS["purchased"]))

        # --- 閲覧 ---
        if ("user", "viewed", "item") in self.data.edge_types:
            view_ei = self.data["user", "viewed", "item"].edge_index
            n_v = view_ei.shape[1]
            all_users.append(view_ei[0])
            all_items.append(view_ei[1])
            all_weights.append(torch.full((n_v,), BEHAVIOR_WEIGHTS["viewed"]))
            print(f"  📖 閲覧エッジを学習ターゲットに追加: {n_v:,}")

        # --- お気に入り ---
        if ("user", "favorited", "item") in self.data.edge_types:
            fav_ei = self.data["user", "favorited", "item"].edge_index
            n_f = fav_ei.shape[1]
            all_users.append(fav_ei[0])
            all_items.append(fav_ei[1])
            all_weights.append(torch.full((n_f,), BEHAVIOR_WEIGHTS["favorited"]))
            print(f"  ❤️  お気に入りエッジを学習ターゲットに追加: {n_f:,}")

        self.train_users = torch.cat(all_users)
        self.train_items = torch.cat(all_items)
        self.train_weights = torch.cat(all_weights)

        # positive set（ネガティブサンプリングの除外用）
        self.pos_set: set[tuple[int, int]] = set()
        for i in range(len(self.train_users)):
            self.pos_set.add((self.train_users[i].item(), self.train_items[i].item()))

        total = len(self.train_users)
        density = total / (self.data["user"].num_nodes * self.n_items) * 100
        print(f"  🔗 統合学習エッジ数: {total:,}  "
              f"(密度: {density:.1f}%)")

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _negative_sample(self, users: torch.Tensor) -> torch.Tensor:
        """Hard negative sampling: ランダム + positive 回避"""
        negs = torch.randint(0, self.n_items, users.shape)
        for i, u in enumerate(users.tolist()):
            attempts = 0
            while (u, negs[i].item()) in self.pos_set and attempts < 20:
                negs[i] = torch.randint(0, self.n_items, (1,))
                attempts += 1
        return negs

    def _to_device(self) -> tuple[dict, dict]:
        x = {nt: self.data[nt].x.to(self.device) for nt in self.data.node_types}
        ei = {}
        for et in self.data.edge_types:
            idx = self.data[et].edge_index
            # 購入エッジはメッセージパッシングにも train 分のみ使用
            if et == ("user", "purchased", "item"):
                idx = self.purchase_train
            ei[et] = idx.to(self.device)
        return x, ei

    # --------------------------------------------------
    # Train epoch (weighted multi-behavior BPR)
    # --------------------------------------------------

    def _train_epoch(self) -> float:
        self.model.train()
        x, ei = self._to_device()

        # シャッフルして DataLoader に投入
        perm = torch.randperm(len(self.train_users))
        users = self.train_users[perm]
        pos_items = self.train_items[perm]
        weights = self.train_weights[perm]
        neg_items = self._negative_sample(users)

        loader = DataLoader(
            TensorDataset(users, pos_items, neg_items, weights),
            batch_size=self.cfg.batch_size, shuffle=False,  # 既にシャッフル済み
        )

        total_loss, n_batch = 0.0, 0
        for bu, bp, bn, bw in loader:
            bu = bu.to(self.device)
            bp = bp.to(self.device)
            bn = bn.to(self.device)
            bw = bw.to(self.device)

            self.optimizer.zero_grad()

            # エンコード
            z = self.model.encode(x, ei)
            z_u = z["user"][bu]
            z_pos = z["item"][bp]
            z_neg = z["item"][bn]

            pos_score = self.model.predict_score(z_u, z_pos)
            neg_score = self.model.predict_score(z_u, z_neg)

            # 重み付き BPR Loss
            bpr = -F.logsigmoid(pos_score - neg_score)
            loss = (bpr * bw).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batch += 1

        self.scheduler.step()
        return total_loss / max(n_batch, 1)

    # --------------------------------------------------
    # Evaluate (購入エッジのみで評価)
    # --------------------------------------------------

    @torch.no_grad()
    def evaluate(self, edges: torch.Tensor, k: int = 10) -> dict[str, float]:
        self.model.eval()
        x, ei = self._to_device()
        z_user, z_item = self.model.get_embeddings(x, ei)

        ground_truth: dict[int, set[int]] = {}
        for i in range(edges.shape[1]):
            u, it = edges[0, i].item(), edges[1, i].item()
            ground_truth.setdefault(u, set()).add(it)

        recalls, ndcgs, hits = [], [], []
        for u, true_items in ground_truth.items():
            z_u = z_user[u].unsqueeze(0).expand(z_item.shape[0], -1)
            scores = self.model.predict_score(z_u, z_item)

            # train positive を除外
            for it in range(self.n_items):
                if (u, it) in self.pos_set:
                    scores[it] = float("-inf")

            _, topk = torch.topk(scores, k)
            topk_set = set(topk.cpu().tolist())

            hit = len(true_items & topk_set)
            recalls.append(hit / min(len(true_items), k))
            hits.append(1.0 if hit > 0 else 0.0)

            dcg = sum(
                1.0 / np.log2(r + 2)
                for r, idx in enumerate(topk.cpu().tolist())
                if idx in true_items
            )
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), k)))
            ndcgs.append(dcg / max(idcg, 1e-10))

        return {
            f"recall@{k}": float(np.mean(recalls)),
            f"ndcg@{k}": float(np.mean(ndcgs)),
            f"hit@{k}": float(np.mean(hits)),
        }

    # --------------------------------------------------
    # Full training loop
    # --------------------------------------------------

    def train(self, epochs: int | None = None, save_dir: str = "checkpoints") -> dict:
        epochs = epochs or self.cfg.epochs
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        best_ndcg = 0.0
        best_metrics: dict = {}
        patience_cnt = 0

        print(f"\n🚀 学習開始  epochs={epochs}  device={self.device}")
        print(f"   行動重み: purchased={BEHAVIOR_WEIGHTS['purchased']:.1f}  "
              f"favorited={BEHAVIOR_WEIGHTS['favorited']:.1f}  "
              f"viewed={BEHAVIOR_WEIGHTS['viewed']:.1f}")
        print("─" * 72)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            loss = self._train_epoch()
            dt = time.time() - t0

            if epoch % 5 == 0 or epoch == 1:
                val = self.evaluate(self.val_edges, k=10)
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch:4d} │ loss={loss:.4f} │ "
                    f"R@10={val['recall@10']:.4f}  N@10={val['ndcg@10']:.4f}  "
                    f"H@10={val['hit@10']:.4f} │ lr={lr:.2e} │ {dt:.1f}s"
                )
                if val["ndcg@10"] > best_ndcg:
                    best_ndcg = val["ndcg@10"]
                    best_metrics = val
                    patience_cnt = 0
                    torch.save(self.model.state_dict(), save_path / "best_model.pt")
                    print(f"         ✨ best model saved (ndcg={best_ndcg:.4f})")
                else:
                    patience_cnt += 5
                    if patience_cnt >= self.cfg.early_stop_patience:
                        print(f"\n  ⏹️  Early stopping at epoch {epoch}")
                        break

        # Test
        print("─" * 72)
        ckpt = save_path / "best_model.pt"
        if ckpt.exists():
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=True))
        test = self.evaluate(self.test_edges, k=10)
        print(f"  📈 テスト結果:  Recall@10={test['recall@10']:.4f}  "
              f"NDCG@10={test['ndcg@10']:.4f}  Hit@10={test['hit@10']:.4f}")
        return test
