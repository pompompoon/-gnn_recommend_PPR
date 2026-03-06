#!/usr/bin/env python3
"""
GNN レコメンド精度評価モジュール

購入エッジを train/test に分割し、test セットに対して多角的な指標を算出。

指標一覧:
  ─ ランキング指標 ─
  Precision@K    : top-K のうち正解がいくつ含まれるか
  Recall@K       : 全正解のうち top-K にいくつ含まれるか
  NDCG@K         : 上位に正解があるほど高い順位重みつきスコア
  Hit Rate@K     : 1件でも当たったユーザーの割合
  MRR            : 最初の正解が何位に現れるかの逆数の平均
  MAP@K          : 各正解の位置ごとの Precision の平均

  ─ カバレッジ・多様性 ─
  Item Coverage  : 全アイテム中レコメンドに登場したアイテムの割合
  Category Coverage : 全カテゴリ中レコメンドに登場したカテゴリの割合
  Avg Intra-List Diversity (ILD) : レコメンドリスト内のカテゴリ多様性
  Novelty        : 人気アイテムからの逸脱度（自己情報量ベース）

Usage:
  python evaluate.py --db-password パスワード
  python evaluate.py --db-password パスワード --top-k 5,10,20
"""
from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData

from config import Config
from db.connection import DatabaseManager
from models.graph_builder import GraphBuilder
from models.gnn_model import GNNRecommender


# =================================================================
# Evaluator クラス
# =================================================================

class Evaluator:
    """GNN レコメンドモデルの包括的評価"""

    def __init__(
        self,
        model: GNNRecommender,
        builder: GraphBuilder,
        data: HeteroData,
        config: Config,
    ):
        self.model = model
        self.builder = builder
        self.data = data
        self.db = DatabaseManager(config.db.connect_kwargs)
        self.device = torch.device(config.model.device)
        self.config = config

        self.n_users = data["user"].num_nodes
        self.n_items = data["item"].num_nodes

        self._split_edges()
        self._precompute()
        self._build_popularity()
        self._build_item_category_map()

    # --------------------------------------------------
    # 準備
    # --------------------------------------------------

    def _split_edges(self) -> None:
        """購入エッジを train / test に分割 + 全行動を positive set に含める"""
        ei = self.data["user", "purchased", "item"].edge_index
        n = ei.shape[1]
        perm = torch.randperm(n)
        n_test = max(1, int(n * self.config.model.test_ratio))

        self.train_edges = ei[:, perm[:-n_test]]
        self.test_edges = ei[:, perm[-n_test:]]

        # train の positive set（購入 + 閲覧 + お気に入り全て含める）
        self.train_pos: dict[int, set[int]] = {}

        # 購入 (train分)
        for i in range(self.train_edges.shape[1]):
            u = self.train_edges[0, i].item()
            it = self.train_edges[1, i].item()
            self.train_pos.setdefault(u, set()).add(it)

        # 閲覧
        if ("user", "viewed", "item") in self.data.edge_types:
            ve = self.data["user", "viewed", "item"].edge_index
            for i in range(ve.shape[1]):
                u, it = ve[0, i].item(), ve[1, i].item()
                self.train_pos.setdefault(u, set()).add(it)

        # お気に入り
        if ("user", "favorited", "item") in self.data.edge_types:
            fe = self.data["user", "favorited", "item"].edge_index
            for i in range(fe.shape[1]):
                u, it = fe[0, i].item(), fe[1, i].item()
                self.train_pos.setdefault(u, set()).add(it)

        # test の ground truth
        self.test_gt: dict[int, set[int]] = {}
        for i in range(self.test_edges.shape[1]):
            u = self.test_edges[0, i].item()
            it = self.test_edges[1, i].item()
            self.test_gt.setdefault(u, set()).add(it)

        total_pos = sum(len(v) for v in self.train_pos.values())
        print(f"  📊 購入エッジ分割: train={self.train_edges.shape[1]:,}  test={self.test_edges.shape[1]:,}")
        print(f"     全行動 positive set: {total_pos:,} ペア")
        print(f"     テスト対象ユーザー数: {len(self.test_gt):,}")

    @torch.no_grad()
    def _precompute(self) -> None:
        """全ノードの embedding を事前計算"""
        self.model.eval()
        self.model.to(self.device)
        x = {nt: self.data[nt].x.to(self.device) for nt in self.data.node_types}
        ei = {}
        for et in self.data.edge_types:
            idx = self.data[et].edge_index
            if et == ("user", "purchased", "item"):
                idx = self.train_edges  # テストリーク防止
            ei[et] = idx.to(self.device)
        self.z_user, self.z_item = self.model.get_embeddings(x, ei)
        print(f"  ✅ Embedding 計算完了  user={self.z_user.shape}  item={self.z_item.shape}")

    def _build_popularity(self) -> None:
        """アイテム人気度（train 内の出現回数）→ Novelty 計算用"""
        counts = Counter()
        for i in range(self.train_edges.shape[1]):
            it = self.train_edges[1, i].item()
            counts[it] += 1
        total = self.train_edges.shape[1]
        # P(item) = 出現回数 / 全購入数, 未出現は最小確率
        min_p = 1.0 / (total + self.n_items)
        self.item_pop = {it: counts.get(it, 0) / total if counts.get(it, 0) > 0 else min_p
                         for it in range(self.n_items)}

    def _build_item_category_map(self) -> None:
        """graph item index → (subcategory_id, category_id) のマッピング"""
        rows = self.db.fetch_all(
            "SELECT i.item_id, i.subcategory_id, s.category_id "
            "FROM items i JOIN subcategories s ON i.subcategory_id = s.subcategory_id"
        )
        iid_map = self.builder.item_id_map
        self.item_subcat: dict[int, int] = {}
        self.item_cat: dict[int, int] = {}
        for r in rows:
            g_idx = iid_map.get(r["item_id"])
            if g_idx is not None:
                self.item_subcat[g_idx] = r["subcategory_id"]
                self.item_cat[g_idx] = r["category_id"]

        self.n_subcats = len(set(self.item_subcat.values()))
        self.n_cats = len(set(self.item_cat.values()))

    # --------------------------------------------------
    # スコア計算
    # --------------------------------------------------

    @torch.no_grad()
    def _get_topk(self, user_idx: int, k: int) -> list[int]:
        """ユーザーに対する top-k アイテムの graph index リスト"""
        z_u = self.z_user[user_idx].unsqueeze(0).expand(self.n_items, -1)
        scores = self.model.predict_score(z_u, self.z_item)

        # train で見たアイテムを除外
        for it in self.train_pos.get(user_idx, set()):
            scores[it] = float("-inf")

        _, topk = torch.topk(scores, k)
        return topk.cpu().tolist()

    # --------------------------------------------------
    # ランキング指標
    # --------------------------------------------------

    def compute_ranking_metrics(self, k_values: list[int]) -> dict[str, dict[str, float]]:
        """
        各 K について Precision/Recall/NDCG/HitRate/MRR/MAP を計算

        Returns: {k: {metric_name: value}}
        """
        max_k = max(k_values)
        results: dict[int, dict[str, list[float]]] = {
            k: {"precision": [], "recall": [], "ndcg": [], "hit": [], "mrr": [], "map": []}
            for k in k_values
        }

        for u, true_items in self.test_gt.items():
            topk_list = self._get_topk(u, max_k)

            for k in k_values:
                topk = topk_list[:k]
                topk_set = set(topk)
                n_hit = len(true_items & topk_set)

                # Precision@K
                results[k]["precision"].append(n_hit / k)

                # Recall@K
                results[k]["recall"].append(n_hit / len(true_items))

                # Hit Rate@K
                results[k]["hit"].append(1.0 if n_hit > 0 else 0.0)

                # NDCG@K
                dcg = sum(1.0 / math.log2(rank + 2)
                          for rank, item in enumerate(topk) if item in true_items)
                idcg = sum(1.0 / math.log2(i + 2)
                           for i in range(min(len(true_items), k)))
                results[k]["ndcg"].append(dcg / max(idcg, 1e-10))

                # MRR (最初の正解の逆数)
                rr = 0.0
                for rank, item in enumerate(topk):
                    if item in true_items:
                        rr = 1.0 / (rank + 1)
                        break
                results[k]["mrr"].append(rr)

                # MAP@K (各正解位置での Precision の平均)
                n_relevant = 0
                sum_precision = 0.0
                for rank, item in enumerate(topk):
                    if item in true_items:
                        n_relevant += 1
                        sum_precision += n_relevant / (rank + 1)
                ap = sum_precision / min(len(true_items), k)
                results[k]["map"].append(ap)

        # 平均化
        final: dict[str, dict[str, float]] = {}
        for k in k_values:
            final[k] = {name: float(np.mean(vals)) for name, vals in results[k].items()}
        return final

    # --------------------------------------------------
    # カバレッジ・多様性・新規性
    # --------------------------------------------------

    def compute_beyond_accuracy_metrics(self, k: int = 10) -> dict[str, float]:
        """
        ランキング以外の指標:
          Item Coverage, Category Coverage, ILD, Novelty
        """
        all_rec_items: set[int] = set()
        all_rec_cats: set[int] = set()
        all_rec_subcats: set[int] = set()
        ild_scores: list[float] = []
        novelty_scores: list[float] = []

        for u in self.test_gt:
            topk = self._get_topk(u, k)
            all_rec_items.update(topk)

            # カテゴリ収集
            cats_in_list = []
            for item in topk:
                cat = self.item_cat.get(item)
                subcat = self.item_subcat.get(item)
                if cat is not None:
                    all_rec_cats.add(cat)
                    cats_in_list.append(cat)
                if subcat is not None:
                    all_rec_subcats.add(subcat)

            # Intra-List Diversity (カテゴリの多様性)
            if len(cats_in_list) >= 2:
                n_pairs = 0
                n_diff = 0
                for i in range(len(cats_in_list)):
                    for j in range(i + 1, len(cats_in_list)):
                        n_pairs += 1
                        if cats_in_list[i] != cats_in_list[j]:
                            n_diff += 1
                ild_scores.append(n_diff / n_pairs)

            # Novelty (自己情報量: -log2(P(item)))
            user_novelty = []
            for item in topk:
                p = self.item_pop.get(item, 1e-10)
                user_novelty.append(-math.log2(max(p, 1e-10)))
            novelty_scores.append(np.mean(user_novelty))

        return {
            "item_coverage": len(all_rec_items) / self.n_items,
            "category_coverage": len(all_rec_cats) / max(self.n_cats, 1),
            "subcategory_coverage": len(all_rec_subcats) / max(self.n_subcats, 1),
            "avg_ILD": float(np.mean(ild_scores)) if ild_scores else 0.0,
            "avg_novelty": float(np.mean(novelty_scores)) if novelty_scores else 0.0,
            "unique_items_recommended": len(all_rec_items),
            "total_items": self.n_items,
        }

    # --------------------------------------------------
    # ユーザーセグメント別評価
    # --------------------------------------------------

    def compute_by_user_activity(self, k: int = 10) -> dict[str, dict[str, float]]:
        """
        ユーザーの train 購入数でセグメントし、セグメント別に Recall/NDCG を計算
        """
        segments = {
            "light (1-5)": [],
            "medium (6-15)": [],
            "heavy (16+)": [],
        }

        for u, true_items in self.test_gt.items():
            n_train = len(self.train_pos.get(u, set()))
            topk = set(self._get_topk(u, k))
            n_hit = len(true_items & topk)
            recall = n_hit / len(true_items)

            dcg = sum(1.0 / math.log2(r + 2)
                      for r, it in enumerate(self._get_topk(u, k)) if it in true_items)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(true_items), k)))
            ndcg = dcg / max(idcg, 1e-10)

            if n_train <= 5:
                segments["light (1-5)"].append((recall, ndcg))
            elif n_train <= 15:
                segments["medium (6-15)"].append((recall, ndcg))
            else:
                segments["heavy (16+)"].append((recall, ndcg))

        result = {}
        for seg, values in segments.items():
            if values:
                recalls = [v[0] for v in values]
                ndcgs = [v[1] for v in values]
                result[seg] = {
                    "n_users": len(values),
                    f"recall@{k}": float(np.mean(recalls)),
                    f"ndcg@{k}": float(np.mean(ndcgs)),
                }
            else:
                result[seg] = {"n_users": 0, f"recall@{k}": 0.0, f"ndcg@{k}": 0.0}
        return result

    # --------------------------------------------------
    # カテゴリ別評価
    # --------------------------------------------------

    def compute_by_category(self, k: int = 10) -> dict[str, dict[str, float]]:
        """
        テストの正解アイテムのカテゴリ別に Hit Rate を計算
        """
        # カテゴリ名マッピング
        cat_names = {}
        rows = self.db.fetch_all("SELECT category_id, name FROM categories")
        for r in rows:
            cat_names[r["category_id"]] = r["name"]

        cat_hits: dict[int, list[float]] = {}
        for u, true_items in self.test_gt.items():
            topk_set = set(self._get_topk(u, k))
            for item in true_items:
                cat = self.item_cat.get(item)
                if cat is not None:
                    hit = 1.0 if item in topk_set else 0.0
                    cat_hits.setdefault(cat, []).append(hit)

        result = {}
        for cat_id, hits in cat_hits.items():
            name = cat_names.get(cat_id, f"cat_{cat_id}")
            result[name] = {
                "n_test_items": len(hits),
                f"hit_rate@{k}": float(np.mean(hits)),
            }
        return result

    # --------------------------------------------------
    # レポート出力
    # --------------------------------------------------

    def full_report(self, k_values: list[int] = None) -> str:
        """全指標を計算して整形されたレポート文字列を返す"""
        if k_values is None:
            k_values = [5, 10, 20]

        lines: list[str] = []
        sep = "=" * 72

        lines.append("")
        lines.append(sep)
        lines.append("  GNN レコメンドシステム 精度評価レポート")
        lines.append(sep)
        lines.append("")

        # --- データ概要 ---
        lines.append("📊 データ概要")
        lines.append(f"  ユーザー数:     {self.n_users:>8,}")
        lines.append(f"  アイテム数:     {self.n_items:>8,}")
        lines.append(f"  Train 購入エッジ: {self.train_edges.shape[1]:>8,}")
        lines.append(f"  Test  購入エッジ: {self.test_edges.shape[1]:>8,}")
        lines.append(f"  テスト対象ユーザー: {len(self.test_gt):>8,}")
        avg_train = np.mean([len(v) for v in self.train_pos.values()]) if self.train_pos else 0
        lines.append(f"  平均 train 購入数/user: {avg_train:>8.1f}")
        lines.append("")

        # --- ランキング指標 ---
        lines.append("─" * 72)
        lines.append("📈 ランキング指標")
        lines.append("─" * 72)

        t0 = time.time()
        ranking = self.compute_ranking_metrics(k_values)
        dt = time.time() - t0
        lines.append(f"  (計算時間: {dt:.1f}s)")
        lines.append("")

        # ヘッダー
        header = f"  {'K':>4s} │ {'Precision':>10s} │ {'Recall':>10s} │ {'NDCG':>10s} │ {'HitRate':>10s} │ {'MRR':>10s} │ {'MAP':>10s}"
        lines.append(header)
        lines.append("  " + "─" * len(header.strip()))

        for k in k_values:
            m = ranking[k]
            lines.append(
                f"  {k:>4d} │ {m['precision']:>10.4f} │ {m['recall']:>10.4f} │ "
                f"{m['ndcg']:>10.4f} │ {m['hit']:>10.4f} │ {m['mrr']:>10.4f} │ {m['map']:>10.4f}"
            )

        lines.append("")

        # 主要指標の解釈
        best_k = k_values[1] if len(k_values) > 1 else k_values[0]
        m = ranking[best_k]
        lines.append(f"  💡 主要指標の読み方 (K={best_k}):")
        lines.append(f"     Precision@{best_k} = {m['precision']:.4f}")
        lines.append(f"       → top-{best_k} の推薦のうち {m['precision']*100:.1f}% が実際に購入された")
        lines.append(f"     Recall@{best_k} = {m['recall']:.4f}")
        lines.append(f"       → テスト購入の {m['recall']*100:.1f}% を top-{best_k} で捕捉")
        lines.append(f"     NDCG@{best_k} = {m['ndcg']:.4f}")
        lines.append(f"       → 正解が上位にあるほど高い (1.0 が理想)")
        lines.append(f"     HitRate@{best_k} = {m['hit']:.4f}")
        lines.append(f"       → {m['hit']*100:.1f}% のユーザーに1件以上的中")
        lines.append(f"     MRR = {m['mrr']:.4f}")
        lines.append(f"       → 最初の正解が平均 {1/max(m['mrr'],1e-10):.1f} 位に出現")
        lines.append("")

        # --- カバレッジ・多様性 ---
        lines.append("─" * 72)
        lines.append("🌐 カバレッジ・多様性・新規性")
        lines.append("─" * 72)

        ba = self.compute_beyond_accuracy_metrics(k=best_k)
        lines.append(f"  Item Coverage       : {ba['item_coverage']:.4f}  "
                     f"({ba['unique_items_recommended']:,} / {ba['total_items']:,} アイテム)")
        lines.append(f"  Category Coverage   : {ba['category_coverage']:.4f}")
        lines.append(f"  SubCategory Coverage: {ba['subcategory_coverage']:.4f}")
        lines.append(f"  Avg ILD (多様性)    : {ba['avg_ILD']:.4f}  "
                     f"(1.0=全ペアが異カテゴリ)")
        lines.append(f"  Avg Novelty         : {ba['avg_novelty']:.2f} bits  "
                     f"(高い=ニッチな推薦)")
        lines.append("")

        # --- ユーザーセグメント別 ---
        lines.append("─" * 72)
        lines.append(f"👥 ユーザーセグメント別 (K={best_k})")
        lines.append("─" * 72)

        segments = self.compute_by_user_activity(k=best_k)
        lines.append(f"  {'セグメント':>18s} │ {'ユーザー数':>10s} │ {'Recall':>10s} │ {'NDCG':>10s}")
        lines.append("  " + "─" * 58)
        for seg, sm in segments.items():
            lines.append(
                f"  {seg:>18s} │ {sm['n_users']:>10,} │ "
                f"{sm[f'recall@{best_k}']:>10.4f} │ {sm[f'ndcg@{best_k}']:>10.4f}"
            )
        lines.append("")

        # --- カテゴリ別 ---
        lines.append("─" * 72)
        lines.append(f"🏷️  カテゴリ別 Hit Rate (K={best_k})")
        lines.append("─" * 72)

        cat_metrics = self.compute_by_category(k=best_k)
        lines.append(f"  {'カテゴリ':>20s} │ {'テスト件数':>10s} │ {'HitRate':>10s}")
        lines.append("  " + "─" * 48)
        for cat_name, cm in sorted(cat_metrics.items(), key=lambda x: -x[1][f"hit_rate@{best_k}"]):
            lines.append(
                f"  {cat_name:>20s} │ {cm['n_test_items']:>10,} │ "
                f"{cm[f'hit_rate@{best_k}']:>10.4f}"
            )
        lines.append("")
        lines.append(sep)

        return "\n".join(lines)


# =================================================================
# CLI
# =================================================================

def main():
    parser = argparse.ArgumentParser(description="GNN Recommendation Evaluation")
    parser.add_argument("--top-k", type=str, default="5,10,20",
                        help="評価する K 値 (カンマ区切り)")
    parser.add_argument("--output", type=str, default=None,
                        help="レポート出力先ファイル (省略時は stdout)")
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", type=int, default=5432)
    parser.add_argument("--db-name", default="fashion_recommend")
    parser.add_argument("--db-user", default="postgres")
    parser.add_argument("--db-password", default="postgres")
    parser.add_argument("--encoder", choices=["gat", "appnp"], default="gat",
                        help="エンコーダー: gat / appnp")
    parser.add_argument("--teleport", type=float, default=0.15)
    parser.add_argument("--ppr-iters", type=int, default=10)
    args = parser.parse_args()

    cfg = Config()
    cfg.db.host = args.db_host
    cfg.db.port = args.db_port
    cfg.db.dbname = args.db_name
    cfg.db.user = args.db_user
    cfg.db.password = args.db_password
    cfg.model.encoder_type = args.encoder
    cfg.model.teleport_prob = args.teleport
    cfg.model.num_iterations = args.ppr_iters

    if torch.cuda.is_available():
        cfg.model.device = "cuda"

    k_values = [int(x.strip()) for x in args.top_k.split(",")]

    # モデルロード
    print("📐 グラフ構築 & モデルロード中...")
    builder = GraphBuilder(cfg.db.connect_kwargs)
    data = builder.build()

    in_ch = {nt: data[nt].x.shape[1] for nt in data.node_types}
    mc = cfg.model
    model = GNNRecommender(
        metadata=data.metadata(), in_channels_dict=in_ch,
        hidden_channels=mc.hidden_channels, out_channels=mc.out_channels,
        num_layers=mc.num_layers, heads=mc.heads, dropout=mc.dropout,
        encoder_type=mc.encoder_type,
        teleport_prob=mc.teleport_prob,
        num_iterations=mc.num_iterations,
    )

    ckpt = Path("checkpoints/best_model.pt")
    if ckpt.exists():
        model.load_state_dict(
            torch.load(ckpt, map_location=mc.device, weights_only=True)
        )
        print(f"  ✅ チェックポイント読込: {ckpt}")
    else:
        print("  ⚠️  チェックポイント未検出 — ランダム重みで評価します")

    # 評価
    print("\n🔬 評価開始...")
    evaluator = Evaluator(model, builder, data, cfg)
    report = evaluator.full_report(k_values)

    print(report)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"\n📄 レポート保存: {args.output}")


if __name__ == "__main__":
    main()
