"""
GraphBuilder  ─  PostgreSQL → PyG HeteroData 変換

ヘテロジニアスグラフを構築:
  ノード: user / item / subcategory / category
  エッジ: purchased / viewed / favorited / belongs_to / child_of
         + 逆エッジ (ToUndirected)

特徴量:
  user : [age_norm, gender_onehot, prefecture_onehot]
  item : [price_norm, subcategory_onehot, brand_onehot, color_onehot, season_onehot]
  subcategory / category : identity matrix
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from db.connection import DatabaseManager


class GraphBuilder:
    """DB → HeteroData 変換 + ID マッピング管理"""

    def __init__(self, connect_kwargs: dict):
        self.db = DatabaseManager(connect_kwargs)
        # DB id → graph index のマップ
        self._user_map: dict[int, int] = {}
        self._item_map: dict[int, int] = {}
        self._subcat_map: dict[int, int] = {}
        self._cat_map: dict[int, int] = {}

    # =========================================================
    # Public
    # =========================================================

    def build(self) -> HeteroData:
        """グラフ全体を構築して返す"""
        data = HeteroData()

        self._build_category_nodes(data)
        self._build_subcategory_nodes(data)
        self._build_user_nodes(data)
        self._build_item_nodes(data)

        self._build_purchase_edges(data)
        self._build_view_edges(data)
        self._build_favorite_edges(data)
        self._build_taxonomy_edges(data)

        # 逆エッジの自動生成 → メッセージパッシングの双方向化
        data = T.ToUndirected()(data)

        self._print_summary(data)
        return data

    # --- ID maps ---
    @property
    def user_id_map(self) -> dict[int, int]:
        return self._user_map

    @property
    def item_id_map(self) -> dict[int, int]:
        return self._item_map

    @property
    def reverse_item_map(self) -> dict[int, int]:
        return {v: k for k, v in self._item_map.items()}

    @property
    def reverse_user_map(self) -> dict[int, int]:
        return {v: k for k, v in self._user_map.items()}

    # =========================================================
    # Node builders
    # =========================================================

    def _build_category_nodes(self, data: HeteroData) -> None:
        rows = self.db.fetch_all("SELECT category_id FROM categories ORDER BY category_id")
        self._cat_map = {r["category_id"]: i for i, r in enumerate(rows)}
        n = len(rows)
        data["category"].x = torch.eye(n, dtype=torch.float)
        data["category"].num_nodes = n

    def _build_subcategory_nodes(self, data: HeteroData) -> None:
        rows = self.db.fetch_all(
            "SELECT subcategory_id, category_id FROM subcategories ORDER BY subcategory_id"
        )
        self._subcat_map = {r["subcategory_id"]: i for i, r in enumerate(rows)}
        self._subcat_to_cat = {r["subcategory_id"]: r["category_id"] for r in rows}
        n = len(rows)
        data["subcategory"].x = torch.eye(n, dtype=torch.float)
        data["subcategory"].num_nodes = n

    def _build_user_nodes(self, data: HeteroData) -> None:
        rows = self.db.fetch_all(
            "SELECT user_id, age, gender, prefecture FROM users ORDER BY user_id"
        )
        self._user_map = {r["user_id"]: i for i, r in enumerate(rows)}

        genders = sorted({r["gender"] for r in rows})
        g_map = {g: i for i, g in enumerate(genders)}
        prefs = sorted({r["prefecture"] for r in rows})
        p_map = {p: i for i, p in enumerate(prefs)}

        feats = []
        for r in rows:
            f = [r["age"] / 100.0]
            g_oh = [0.0] * len(genders)
            g_oh[g_map[r["gender"]]] = 1.0
            f.extend(g_oh)
            p_oh = [0.0] * len(prefs)
            p_oh[p_map[r["prefecture"]]] = 1.0
            f.extend(p_oh)
            feats.append(f)

        data["user"].x = torch.tensor(feats, dtype=torch.float)
        data["user"].num_nodes = len(rows)

    def _build_item_nodes(self, data: HeteroData) -> None:
        rows = self.db.fetch_all(
            "SELECT item_id, subcategory_id, brand_id, price, color, season "
            "FROM items ORDER BY item_id"
        )
        self._item_map = {r["item_id"]: i for i, r in enumerate(rows)}

        subcats = sorted({r["subcategory_id"] for r in rows})
        sc_map = {s: i for i, s in enumerate(subcats)}
        brands = sorted({r["brand_id"] for r in rows if r["brand_id"]})
        b_map = {b: i for i, b in enumerate(brands)}
        colors = sorted({r["color"] for r in rows if r["color"]})
        c_map = {c: i for i, c in enumerate(colors)}
        seasons = sorted({r["season"] for r in rows if r["season"]})
        s_map = {s: i for i, s in enumerate(seasons)}

        max_price = max(r["price"] for r in rows) or 1

        feats = []
        for r in rows:
            f = [r["price"] / max_price]

            sc_oh = [0.0] * len(subcats)
            sc_oh[sc_map[r["subcategory_id"]]] = 1.0
            f.extend(sc_oh)

            b_oh = [0.0] * len(brands)
            if r["brand_id"] in b_map:
                b_oh[b_map[r["brand_id"]]] = 1.0
            f.extend(b_oh)

            c_oh = [0.0] * len(colors)
            if r["color"] in c_map:
                c_oh[c_map[r["color"]]] = 1.0
            f.extend(c_oh)

            s_oh = [0.0] * len(seasons)
            if r["season"] in s_map:
                s_oh[s_map[r["season"]]] = 1.0
            f.extend(s_oh)

            feats.append(f)

        data["item"].x = torch.tensor(feats, dtype=torch.float)
        data["item"].num_nodes = len(rows)

    # =========================================================
    # Edge builders
    # =========================================================

    def _build_purchase_edges(self, data: HeteroData) -> None:
        rows = self.db.fetch_all("SELECT user_id, item_id, quantity FROM purchases")
        src, dst, w = [], [], []
        for r in rows:
            u = self._user_map.get(r["user_id"])
            it = self._item_map.get(r["item_id"])
            if u is not None and it is not None:
                src.append(u); dst.append(it)
                w.append(float(r["quantity"]))
        data["user", "purchased", "item"].edge_index = torch.tensor([src, dst], dtype=torch.long)
        data["user", "purchased", "item"].edge_attr = torch.tensor(w, dtype=torch.float).unsqueeze(1)

    def _build_view_edges(self, data: HeteroData) -> None:
        rows = self.db.fetch_all("SELECT user_id, item_id, duration_sec FROM views")
        src, dst, w = [], [], []
        for r in rows:
            u = self._user_map.get(r["user_id"])
            it = self._item_map.get(r["item_id"])
            if u is not None and it is not None:
                src.append(u); dst.append(it)
                w.append(np.log1p(float(r["duration_sec"] or 0)))
        data["user", "viewed", "item"].edge_index = torch.tensor([src, dst], dtype=torch.long)
        data["user", "viewed", "item"].edge_attr = torch.tensor(w, dtype=torch.float).unsqueeze(1)

    def _build_favorite_edges(self, data: HeteroData) -> None:
        rows = self.db.fetch_all("SELECT user_id, item_id FROM favorites")
        src, dst = [], []
        for r in rows:
            u = self._user_map.get(r["user_id"])
            it = self._item_map.get(r["item_id"])
            if u is not None and it is not None:
                src.append(u); dst.append(it)
        data["user", "favorited", "item"].edge_index = torch.tensor([src, dst], dtype=torch.long)

    def _build_taxonomy_edges(self, data: HeteroData) -> None:
        # item → subcategory
        rows = self.db.fetch_all("SELECT item_id, subcategory_id FROM items")
        src, dst = [], []
        for r in rows:
            it = self._item_map.get(r["item_id"])
            sc = self._subcat_map.get(r["subcategory_id"])
            if it is not None and sc is not None:
                src.append(it); dst.append(sc)
        data["item", "belongs_to", "subcategory"].edge_index = torch.tensor([src, dst], dtype=torch.long)

        # subcategory → category
        src, dst = [], []
        for sub_id, cat_id in self._subcat_to_cat.items():
            sc = self._subcat_map.get(sub_id)
            ca = self._cat_map.get(cat_id)
            if sc is not None and ca is not None:
                src.append(sc); dst.append(ca)
        data["subcategory", "child_of", "category"].edge_index = torch.tensor([src, dst], dtype=torch.long)

    # =========================================================
    # Utils
    # =========================================================

    @staticmethod
    def _print_summary(data: HeteroData) -> None:
        print("\n📊 グラフ構築完了:")
        for nt in data.node_types:
            x = data[nt].x
            print(f"  [{nt:15s}]  nodes={x.shape[0]:>6,}  features={x.shape[1]}")
        for et in data.edge_types:
            n = data[et].edge_index.shape[1]
            print(f"  {str(et):55s}  edges={n:>8,}")
        print()
