"""
レコメンド推論エンジン

  recommend_for_user()   : 特定ユーザーへの top-k レコメンド
  find_similar_items()   : アイテム間のコサイン類似度で類似アイテム検索
  save_to_db()           : レコメンド結果を PostgreSQL に保存
"""
from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from config import Config
from db.connection import DatabaseManager
from models.graph_builder import GraphBuilder
from models.gnn_model import GNNRecommender


class Recommender:
    """学習済みモデルによるレコメンド"""

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

        self._precompute()

    @torch.no_grad()
    def _precompute(self) -> None:
        self.model.eval()
        x = {nt: self.data[nt].x.to(self.device) for nt in self.data.node_types}
        ei = {et: self.data[et].edge_index.to(self.device) for et in self.data.edge_types}
        self.z_user, self.z_item = self.model.get_embeddings(x, ei)
        print(f"  ✅ Embedding 事前計算完了  "
              f"user={self.z_user.shape}  item={self.z_item.shape}")

    # =========================================================
    # User → Item recommendation
    # =========================================================

    @torch.no_grad()
    def recommend_for_user(
        self, user_id: int, top_k: int = 10, exclude_purchased: bool = True,
    ) -> list[dict]:
        uid_map = self.builder.user_id_map
        iid_rev = self.builder.reverse_item_map

        if user_id not in uid_map:
            print(f"  ⚠️  user_id={user_id} not found")
            return []

        g_uid = uid_map[user_id]
        z_u = self.z_user[g_uid].unsqueeze(0).expand(self.z_item.shape[0], -1)
        scores = self.model.predict_score(z_u, self.z_item)

        # 購入済みを除外
        if exclude_purchased:
            rows = self.db.fetch_all(
                "SELECT DISTINCT item_id FROM purchases WHERE user_id=%s", (user_id,),
            )
            for r in rows:
                g_iid = self.builder.item_id_map.get(r["item_id"])
                if g_iid is not None:
                    scores[g_iid] = float("-inf")

        top_scores, top_idx = torch.topk(scores, top_k)

        results = []
        for score_val, idx in zip(top_scores.cpu().tolist(), top_idx.cpu().tolist()):
            db_iid = iid_rev.get(idx)
            if db_iid is None:
                continue
            info = self.db.fetch_one(
                "SELECT i.item_id, i.name, i.price, i.color, i.season, i.is_on_sale, "
                "       s.name AS subcategory, c.name AS category, b.name AS brand "
                "FROM items i "
                "JOIN subcategories s ON i.subcategory_id = s.subcategory_id "
                "JOIN categories c ON s.category_id = c.category_id "
                "LEFT JOIN brands b ON i.brand_id = b.brand_id "
                "WHERE i.item_id = %s",
                (db_iid,),
            )
            if info:
                results.append({
                    "rank": len(results) + 1,
                    "item_id": info["item_id"],
                    "name": info["name"],
                    "category": info["category"],
                    "subcategory": info["subcategory"],
                    "brand": info["brand"],
                    "price": int(info["price"]),
                    "color": info["color"],
                    "season": info["season"],
                    "is_on_sale": info["is_on_sale"],
                    "score": round(score_val, 5),
                })
        return results

    # =========================================================
    # Item → Item similarity
    # =========================================================

    @torch.no_grad()
    def find_similar_items(self, item_id: int, top_k: int = 10) -> list[dict]:
        iid_map = self.builder.item_id_map
        iid_rev = self.builder.reverse_item_map

        if item_id not in iid_map:
            print(f"  ⚠️  item_id={item_id} not found")
            return []

        g_iid = iid_map[item_id]
        z_t = self.z_item[g_iid]

        # cosine similarity
        norms = self.z_item.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z_normed = self.z_item / norms
        t_normed = z_t / z_t.norm().clamp(min=1e-8)
        sims = (z_normed @ t_normed.unsqueeze(1)).squeeze(1)
        sims[g_iid] = -1.0  # 自分を除外

        top_sim, top_idx = torch.topk(sims, top_k)
        results = []
        for sim_val, idx in zip(top_sim.cpu().tolist(), top_idx.cpu().tolist()):
            db_iid = iid_rev.get(idx)
            if db_iid is None:
                continue
            info = self.db.fetch_one(
                "SELECT i.item_id, i.name, i.price, i.color, "
                "       s.name AS subcategory, c.name AS category, b.name AS brand "
                "FROM items i "
                "JOIN subcategories s ON i.subcategory_id = s.subcategory_id "
                "JOIN categories c ON s.category_id = c.category_id "
                "LEFT JOIN brands b ON i.brand_id = b.brand_id "
                "WHERE i.item_id = %s",
                (db_iid,),
            )
            if info:
                results.append({
                    "rank": len(results) + 1,
                    "item_id": info["item_id"],
                    "name": info["name"],
                    "category": info["category"],
                    "subcategory": info["subcategory"],
                    "brand": info["brand"],
                    "price": int(info["price"]),
                    "color": info["color"],
                    "similarity": round(sim_val, 5),
                })
        return results

    # =========================================================
    # DB persistence
    # =========================================================

    def save_to_db(
        self,
        user_ids: list[int] | None = None,
        top_k: int = 20,
        model_version: str = "v1.0",
    ) -> None:
        if user_ids is None:
            rows = self.db.fetch_all("SELECT user_id FROM users ORDER BY user_id")
            user_ids = [r["user_id"] for r in rows]

        self.db.execute(
            "DELETE FROM recommendations WHERE model_version=%s",
            (model_version,),
        )

        total = 0
        for uid in user_ids:
            recs = self.recommend_for_user(uid, top_k)
            if recs:
                vals = [(uid, r["item_id"], r["score"], r["rank"], model_version) for r in recs]
                self.db.execute_values(
                    "INSERT INTO recommendations "
                    "(user_id, item_id, score, rank, model_version) VALUES %s",
                    vals,
                )
                total += len(vals)

        print(f"  ✅ {len(user_ids):,} ユーザー × top-{top_k} → {total:,} 件を DB に保存")

    # =========================================================
    # Pretty print
    # =========================================================

    def print_user_recommendations(self, user_id: int, top_k: int = 10) -> None:
        # ユーザー情報
        u = self.db.fetch_one(
            "SELECT username, age, gender, prefecture FROM users WHERE user_id=%s",
            (user_id,),
        )
        if u:
            print(f"\n  👤 {u['username']}  (ID={user_id}, {u['age']}歳, "
                  f"{u['gender']}, {u['prefecture']})")

        # 購入履歴
        history = self.db.fetch_all(
            "SELECT i.name, s.name AS sub, b.name AS brand, i.price "
            "FROM purchases p "
            "JOIN items i ON p.item_id = i.item_id "
            "JOIN subcategories s ON i.subcategory_id = s.subcategory_id "
            "LEFT JOIN brands b ON i.brand_id = b.brand_id "
            "WHERE p.user_id = %s ORDER BY p.purchased_at DESC LIMIT 5",
            (user_id,),
        )
        if history:
            print("  📦 最近の購入:")
            for h in history:
                print(f"     ・{h['name']}  ({h['sub']})  ¥{h['price']:,}")

        # レコメンド
        recs = self.recommend_for_user(user_id, top_k)
        if recs:
            print(f"\n  🎯 レコメンド Top-{top_k}:")
            for r in recs:
                sale = " 🔥SALE" if r.get("is_on_sale") else ""
                print(
                    f"     {r['rank']:2d}. {r['name']}"
                    f"  [{r['category']}>{r['subcategory']}]"
                    f"  ¥{r['price']:,}  score={r['score']:.4f}{sale}"
                )
        else:
            print("  レコメンドなし")
        print()
