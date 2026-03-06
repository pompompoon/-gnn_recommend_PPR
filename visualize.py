"""
GNN レコメンド結果の可視化モジュール

  PostgreSQL → NetworkX グラフ構築 → pyvis / matplotlib で可視化

使い方:
  # 1. インタラクティブ HTML (pyvis)
  python visualize.py --user-id 1 --top-k 5 --output network.html

  # 2. 静的画像 (matplotlib)
  python visualize.py --user-id 1 --top-k 5 --output network.png --static

  # 3. 複数ユーザー比較
  python visualize.py --user-ids 1,2,3 --top-k 3 --output compare.html

  # 4. 類似アイテムネットワーク
  python visualize.py --item-id 42 --similar --top-k 8 --output similar.html

必要パッケージ:
  pip install networkx pyvis matplotlib japanize-matplotlib
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx

from config import Config
from db.connection import DatabaseManager


# =================================================================
# グラフ構築: PostgreSQL → NetworkX
# =================================================================

class RecommendationGraphBuilder:
    """DB のレコメンド結果 + 行動ログから NetworkX グラフを構築"""

    # ノードタイプ別の色とサイズ
    STYLE = {
        "user":        {"color": "#f59e0b", "size": 40, "shape": "dot",     "font_color": "#f59e0b"},
        "item":        {"color": "#3b82f6", "size": 20, "shape": "dot",     "font_color": "#93c5fd"},
        "item_rec":    {"color": "#10b981", "size": 25, "shape": "star",    "font_color": "#6ee7b7"},
        "subcategory": {"color": "#8b5cf6", "size": 18, "shape": "diamond", "font_color": "#c4b5fd"},
        "category":    {"color": "#ec4899", "size": 30, "shape": "triangle","font_color": "#f9a8d4"},
    }

    EDGE_STYLE = {
        "purchased":  {"color": "#f59e0b", "width": 2.5, "dashes": False},
        "viewed":     {"color": "#64748b", "width": 1.0, "dashes": [5, 5]},
        "favorited":  {"color": "#f472b6", "width": 1.5, "dashes": [2, 2]},
        "recommend":  {"color": "#10b981", "width": 3.0, "dashes": [8, 4]},
        "belongs_to": {"color": "#475569", "width": 0.8, "dashes": False},
        "child_of":   {"color": "#475569", "width": 0.8, "dashes": False},
    }

    def __init__(self, config: Config):
        self.db = DatabaseManager(config.db.connect_kwargs)

    def build_user_recommendation_graph(
        self,
        user_id: int,
        recommendations: list[dict],
        show_views: bool = False,
        show_favorites: bool = True,
        show_categories: bool = True,
    ) -> nx.DiGraph:
        """
        1人のユーザーのレコメンドネットワークを構築

        Parameters
        ----------
        user_id          : 対象ユーザーの DB ID
        recommendations  : Recommender.recommend_for_user() の返り値
        show_views       : 閲覧エッジを表示するか
        show_favorites   : お気に入りエッジを表示するか
        show_categories  : カテゴリ階層を表示するか
        """
        G = nx.DiGraph()

        # --- ユーザーノード ---
        user = self.db.fetch_one(
            "SELECT user_id, username, age, gender, prefecture "
            "FROM users WHERE user_id = %s", (user_id,),
        )
        if not user:
            print(f"  ⚠️  user_id={user_id} not found")
            return G

        uid = f"user_{user_id}"
        G.add_node(uid,
            label=user["username"],
            title=f"👤 {user['username']}\n{user['age']}歳 / {user['gender']}\n{user['prefecture']}",
            node_type="user",
            **self.STYLE["user"],
        )

        # --- 購入アイテム ---
        purchases = self.db.fetch_all(
            "SELECT DISTINCT i.item_id, i.name, i.price, i.color, "
            "       s.name AS subcategory, s.subcategory_id, "
            "       c.name AS category, c.category_id "
            "FROM purchases p "
            "JOIN items i ON p.item_id = i.item_id "
            "JOIN subcategories s ON i.subcategory_id = s.subcategory_id "
            "JOIN categories c ON s.category_id = c.category_id "
            "WHERE p.user_id = %s",
            (user_id,),
        )
        purchased_ids = set()
        for it in purchases:
            nid = f"item_{it['item_id']}"
            purchased_ids.add(it["item_id"])
            G.add_node(nid,
                label=self._short_name(it["name"]),
                title=f"📦 {it['name']}\n¥{it['price']:,} / {it['color']}\n[{it['category']} > {it['subcategory']}]",
                node_type="item",
                **self.STYLE["item"],
            )
            G.add_edge(uid, nid,
                label="購入",
                title="purchased",
                edge_type="purchased",
                **self.EDGE_STYLE["purchased"],
            )

            if show_categories:
                self._add_category_nodes(G, it)

        # --- レコメンドアイテム ---
        rec_ids = set()
        for rec in recommendations:
            nid = f"item_{rec['item_id']}"
            rec_ids.add(rec["item_id"])
            if rec["item_id"] in purchased_ids:
                continue  # 購入済みと重複する場合はスキップ

            score_str = f"{rec['score']:.3f}" if isinstance(rec['score'], float) else str(rec['score'])
            G.add_node(nid,
                label=f"#{rec['rank']} {self._short_name(rec['name'])}",
                title=(
                    f"🎯 レコメンド #{rec['rank']}\n{rec['name']}\n"
                    f"¥{rec['price']:,} / {rec['color']}\n"
                    f"[{rec['category']} > {rec['subcategory']}]\n"
                    f"score: {score_str}"
                ),
                node_type="item_rec",
                **self.STYLE["item_rec"],
            )
            G.add_edge(uid, nid,
                label=f"rec #{rec['rank']}",
                title=f"recommend (score={score_str})",
                edge_type="recommend",
                **self.EDGE_STYLE["recommend"],
            )

            if show_categories:
                # レコメンドアイテムのカテゴリ情報を取得
                item_info = self.db.fetch_one(
                    "SELECT s.name AS subcategory, s.subcategory_id, "
                    "       c.name AS category, c.category_id "
                    "FROM items i "
                    "JOIN subcategories s ON i.subcategory_id = s.subcategory_id "
                    "JOIN categories c ON s.category_id = c.category_id "
                    "WHERE i.item_id = %s",
                    (rec["item_id"],),
                )
                if item_info:
                    self._add_category_nodes(G, {**rec, **item_info})

        # --- 閲覧エッジ (オプション) ---
        if show_views:
            views = self.db.fetch_all(
                "SELECT DISTINCT i.item_id, i.name, i.price, i.color, "
                "       s.name AS subcategory, s.subcategory_id, "
                "       c.name AS category, c.category_id, "
                "       v.duration_sec "
                "FROM views v "
                "JOIN items i ON v.item_id = i.item_id "
                "JOIN subcategories s ON i.subcategory_id = s.subcategory_id "
                "JOIN categories c ON s.category_id = c.category_id "
                "WHERE v.user_id = %s "
                "ORDER BY v.duration_sec DESC LIMIT 15",
                (user_id,),
            )
            for it in views:
                nid = f"item_{it['item_id']}"
                if not G.has_node(nid):
                    G.add_node(nid,
                        label=self._short_name(it["name"]),
                        title=f"👁️ {it['name']}\n¥{it['price']:,}\n閲覧 {it['duration_sec']}秒",
                        node_type="item",
                        color="#64748b", size=15, shape="dot", font_color="#94a3b8",
                    )
                if not G.has_edge(uid, nid):
                    G.add_edge(uid, nid,
                        label="閲覧",
                        title=f"viewed ({it['duration_sec']}s)",
                        edge_type="viewed",
                        **self.EDGE_STYLE["viewed"],
                    )

        # --- お気に入りエッジ (オプション) ---
        if show_favorites:
            favs = self.db.fetch_all(
                "SELECT DISTINCT i.item_id, i.name, i.price, i.color, "
                "       s.name AS subcategory, s.subcategory_id, "
                "       c.name AS category, c.category_id "
                "FROM favorites f "
                "JOIN items i ON f.item_id = i.item_id "
                "JOIN subcategories s ON i.subcategory_id = s.subcategory_id "
                "JOIN categories c ON s.category_id = c.category_id "
                "WHERE f.user_id = %s LIMIT 10",
                (user_id,),
            )
            for it in favs:
                nid = f"item_{it['item_id']}"
                if not G.has_node(nid):
                    G.add_node(nid,
                        label=self._short_name(it["name"]),
                        title=f"❤️ {it['name']}\n¥{it['price']:,}",
                        node_type="item",
                        color="#f472b6", size=18, shape="dot", font_color="#f9a8d4",
                    )
                if not G.has_edge(uid, nid):
                    G.add_edge(uid, nid,
                        label="❤️",
                        title="favorited",
                        edge_type="favorited",
                        **self.EDGE_STYLE["favorited"],
                    )

                if show_categories:
                    self._add_category_nodes(G, it)

        return G

    def build_multi_user_graph(
        self,
        user_recommendations: dict[int, list[dict]],
        show_categories: bool = True,
    ) -> nx.DiGraph:
        """
        複数ユーザーのレコメンドを1つのグラフに統合
        共有アイテム・カテゴリの関係が見える
        """
        G = nx.DiGraph()
        for user_id, recs in user_recommendations.items():
            sub_g = self.build_user_recommendation_graph(
                user_id, recs,
                show_views=False, show_favorites=False,
                show_categories=show_categories,
            )
            G = nx.compose(G, sub_g)
        return G

    def build_similar_items_graph(
        self, item_id: int, similar_items: list[dict],
    ) -> nx.DiGraph:
        """類似アイテムネットワークを構築"""
        G = nx.DiGraph()

        # 基準アイテム
        base = self.db.fetch_one(
            "SELECT i.item_id, i.name, i.price, i.color, "
            "       s.name AS subcategory, c.name AS category "
            "FROM items i "
            "JOIN subcategories s ON i.subcategory_id = s.subcategory_id "
            "JOIN categories c ON s.category_id = c.category_id "
            "WHERE i.item_id = %s", (item_id,),
        )
        if not base:
            return G

        center = f"item_{item_id}"
        G.add_node(center,
            label=self._short_name(base["name"]),
            title=f"🔍 基準アイテム\n{base['name']}\n¥{base['price']:,}\n{base['category']} > {base['subcategory']}",
            color="#f59e0b", size=35, shape="star", font_color="#fbbf24",
        )

        for sim in similar_items:
            nid = f"item_{sim['item_id']}"
            sim_val = sim.get("similarity", 0)
            G.add_node(nid,
                label=f"#{sim['rank']} {self._short_name(sim['name'])}",
                title=f"{sim['name']}\n¥{sim['price']:,}\n{sim['category']} > {sim['subcategory']}\nsim={sim_val:.4f}",
                color="#3b82f6", size=15 + int(sim_val * 20),
                shape="dot", font_color="#93c5fd",
            )
            G.add_edge(center, nid,
                label=f"{sim_val:.2f}",
                title=f"cosine similarity = {sim_val:.4f}",
                color="#3b82f6", width=max(0.5, sim_val * 4),
            )

        return G

    # --- private helpers ---

    def _add_category_nodes(self, G: nx.DiGraph, item_info: dict) -> None:
        """アイテムからサブカテゴリ → カテゴリ のエッジを追加"""
        item_nid = f"item_{item_info['item_id']}"
        sub_nid = f"sub_{item_info['subcategory_id']}"
        cat_nid = f"cat_{item_info['category_id']}"

        if not G.has_node(sub_nid):
            G.add_node(sub_nid,
                label=item_info["subcategory"],
                title=f"📂 {item_info['subcategory']}",
                node_type="subcategory",
                **self.STYLE["subcategory"],
            )
        if not G.has_node(cat_nid):
            G.add_node(cat_nid,
                label=item_info["category"],
                title=f"🏷️ {item_info['category']}",
                node_type="category",
                **self.STYLE["category"],
            )

        if not G.has_edge(item_nid, sub_nid):
            G.add_edge(item_nid, sub_nid,
                edge_type="belongs_to",
                **self.EDGE_STYLE["belongs_to"],
            )
        if not G.has_edge(sub_nid, cat_nid):
            G.add_edge(sub_nid, cat_nid,
                edge_type="child_of",
                **self.EDGE_STYLE["child_of"],
            )

    @staticmethod
    def _short_name(name: str, max_len: int = 18) -> str:
        return name[:max_len] + "…" if len(name) > max_len else name


# =================================================================
# 可視化: pyvis (インタラクティブ HTML)
# =================================================================

def visualize_pyvis(
    G: nx.DiGraph,
    output: str = "network.html",
    title: str = "GNN Recommendation Network",
    height: str = "900px",
    width: str = "100%",
) -> str:
    """
    NetworkX グラフ → pyvis インタラクティブ HTML

    Returns: 出力ファイルパス
    """
    from pyvis.network import Network

    net = Network(
        height=height, width=width,
        directed=True,
        bgcolor="#0a0e17",
        font_color="#e2e8f0",
        notebook=False,
    )

    # 物理エンジン設定
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -120,
          "centralGravity": 0.015,
          "springLength": 160,
          "springConstant": 0.04,
          "damping": 0.5
        },
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 200}
      },
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
        "smooth": {"type": "curvedCW", "roundness": 0.15}
      },
      "nodes": {
        "font": {"size": 12, "face": "Noto Sans JP, sans-serif"},
        "borderWidth": 2
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)

    # ノード追加
    for node_id, attrs in G.nodes(data=True):
        net.add_node(
            node_id,
            label=attrs.get("label", node_id),
            title=attrs.get("title", ""),
            color=attrs.get("color", "#888"),
            size=attrs.get("size", 20),
            shape=attrs.get("shape", "dot"),
            font={"color": attrs.get("font_color", "#e2e8f0")},
        )

    # エッジ追加
    for src, dst, attrs in G.edges(data=True):
        net.add_edge(
            src, dst,
            title=attrs.get("title", ""),
            label=attrs.get("label", ""),
            color=attrs.get("color", "#475569"),
            width=attrs.get("width", 1),
            dashes=attrs.get("dashes", False),
        )

    # タイトル注入
    net.html = f"""<h2 style="
        text-align:center; color:#10b981; font-family:'Noto Sans JP',sans-serif;
        margin:10px 0; background:#0a0e17; padding:8px 0;
    ">{title}</h2>\n""" + (net.html if hasattr(net, 'html') else "")

    net.save_graph(output)
    print(f"  ✅ pyvis HTML 保存: {output}")
    return output


# =================================================================
# 可視化: matplotlib (静的画像)
# =================================================================

def visualize_matplotlib(
    G: nx.DiGraph,
    output: str = "network.png",
    title: str = "GNN Recommendation Network",
    figsize: tuple = (18, 12),
) -> str:
    """
    NetworkX グラフ → matplotlib 静的画像

    Returns: 出力ファイルパス
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import japanize_matplotlib  # noqa: F401 – 日本語フォント対応
    except ImportError:
        plt.rcParams["font.family"] = "Noto Sans JP"

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="#0a0e17")
    ax.set_facecolor("#0a0e17")

    # レイアウト: 階層的に配置
    pos = _hierarchical_layout(G)

    # ノードタイプ別に描画
    node_types_order = ["category", "subcategory", "item", "item_rec", "user"]
    for ntype in node_types_order:
        nodelist = [n for n, d in G.nodes(data=True) if d.get("node_type", "") == ntype]
        if not nodelist:
            # node_type 未設定のノードも拾う
            if ntype == "item":
                nodelist = [n for n, d in G.nodes(data=True)
                            if "node_type" not in d and n.startswith("item_")]
            else:
                continue
        if not nodelist:
            continue

        style = RecommendationGraphBuilder.STYLE.get(ntype, {"color": "#888", "size": 20})
        colors = [G.nodes[n].get("color", style["color"]) for n in nodelist]
        sizes = [G.nodes[n].get("size", style["size"]) * 15 for n in nodelist]

        nx.draw_networkx_nodes(
            G, pos, nodelist=nodelist, node_color=colors,
            node_size=sizes, alpha=0.9, ax=ax,
        )

    # エッジ描画
    edge_types = {"purchased", "recommend", "viewed", "favorited", "belongs_to", "child_of"}
    for etype in edge_types:
        edgelist = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == etype]
        if not edgelist:
            continue
        style = RecommendationGraphBuilder.EDGE_STYLE.get(etype, {"color": "#475569", "width": 1})
        line_style = "dashed" if etype in ("recommend", "viewed", "favorited") else "solid"

        nx.draw_networkx_edges(
            G, pos, edgelist=edgelist,
            edge_color=style["color"], width=style["width"],
            alpha=0.6, style=line_style,
            arrows=True, arrowsize=12, ax=ax,
            connectionstyle="arc3,rad=0.08",
        )

    # ラベル描画
    labels = {n: d.get("label", n) for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(
        G, pos, labels, font_size=8, font_color="#e2e8f0",
        font_family="Noto Sans JP", ax=ax,
    )

    # エッジラベル (レコメンドスコア)
    rec_edge_labels = {}
    for u, v, d in G.edges(data=True):
        if d.get("edge_type") == "recommend" and "label" in d:
            rec_edge_labels[(u, v)] = d["label"]
    if rec_edge_labels:
        nx.draw_networkx_edge_labels(
            G, pos, rec_edge_labels, font_size=7,
            font_color="#10b981", ax=ax,
        )

    # 凡例
    legend_items = [
        ("●", "#f59e0b", "User"),
        ("●", "#3b82f6", "購入 Item"),
        ("★", "#10b981", "レコメンド Item"),
        ("◆", "#8b5cf6", "SubCategory"),
        ("▲", "#ec4899", "Category"),
        ("─", "#f59e0b", "purchased"),
        ("╌", "#10b981", "recommend"),
    ]
    for i, (marker, color, label) in enumerate(legend_items):
        ax.text(0.02, 0.98 - i * 0.035, f" {marker} {label}",
                transform=ax.transAxes, fontsize=9, color=color,
                verticalalignment="top", fontfamily="Noto Sans JP")

    ax.set_title(title, fontsize=16, color="#10b981",
                 fontfamily="Noto Sans JP", pad=15)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=150, facecolor="#0a0e17",
                edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"  ✅ matplotlib 画像保存: {output}")
    return output


def _hierarchical_layout(G: nx.DiGraph) -> dict:
    """ノードタイプに基づく階層的レイアウト"""
    layer_x = {
        "user": 0.0,
        "item": 1.5,
        "item_rec": 1.5,
        "subcategory": 3.0,
        "category": 4.5,
    }

    nodes_by_type: dict[str, list] = {}
    for n, d in G.nodes(data=True):
        ntype = d.get("node_type", "item")
        nodes_by_type.setdefault(ntype, []).append(n)

    pos = {}
    for ntype, nodes in nodes_by_type.items():
        x = layer_x.get(ntype, 1.5)
        n = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            y = (i - n / 2) * 0.8
            pos[node] = (x, y)

    return pos


# =================================================================
# グラフ統計情報の出力
# =================================================================

def print_graph_stats(G: nx.DiGraph) -> None:
    """NetworkX グラフの統計情報を表示"""
    print(f"\n📊 NetworkX グラフ統計:")
    print(f"  ノード数: {G.number_of_nodes()}")
    print(f"  エッジ数: {G.number_of_edges()}")

    # ノードタイプ別
    type_counts: dict[str, int] = {}
    for _, d in G.nodes(data=True):
        t = d.get("node_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"    {t:15s}: {c}")

    # エッジタイプ別
    edge_counts: dict[str, int] = {}
    for _, _, d in G.edges(data=True):
        t = d.get("edge_type", "unknown")
        edge_counts[t] = edge_counts.get(t, 0) + 1
    for t, c in sorted(edge_counts.items()):
        print(f"    {t:15s}: {c}")

    # 次数統計
    degrees = [d for _, d in G.degree()]
    if degrees:
        print(f"  次数: min={min(degrees)}  max={max(degrees)}  "
              f"avg={sum(degrees)/len(degrees):.1f}")
    print()


# =================================================================
# CLI Entry Point
# =================================================================

def main():
    parser = argparse.ArgumentParser(description="GNN Recommendation Network Visualization")
    parser.add_argument("--user-id", type=int, default=None, help="対象ユーザーID")
    parser.add_argument("--user-ids", type=str, default=None, help="複数ユーザー (カンマ区切り, e.g. 1,2,3)")
    parser.add_argument("--item-id", type=int, default=None, help="類似アイテム検索の基準アイテムID")
    parser.add_argument("--similar", action="store_true", help="類似アイテムモード")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=str, default="network.html")
    parser.add_argument("--static", action="store_true", help="matplotlib 静的画像で出力")
    parser.add_argument("--show-views", action="store_true", help="閲覧エッジを表示")
    parser.add_argument("--show-favorites", action="store_true", default=True)
    parser.add_argument("--no-categories", action="store_true", help="カテゴリ階層を非表示")
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", type=int, default=5432)
    parser.add_argument("--db-name", default="fashion_recommend")
    parser.add_argument("--db-user", default="postgres")
    parser.add_argument("--db-password", default="postgres")
    parser.add_argument("--encoder", choices=["gat", "appnp"], default="gat")
    parser.add_argument("--teleport", type=float, default=0.15)
    parser.add_argument("--ppr-iters", type=int, default=10)
    args = parser.parse_args()

    # Config
    cfg = Config()
    cfg.db.host = args.db_host
    cfg.db.port = args.db_port
    cfg.db.dbname = args.db_name
    cfg.db.user = args.db_user
    cfg.db.password = args.db_password
    cfg.model.encoder_type = args.encoder
    cfg.model.teleport_prob = args.teleport
    cfg.model.num_iterations = args.ppr_iters

    import torch
    from models.graph_builder import GraphBuilder
    from models.gnn_model import GNNRecommender
    from models.recommend import Recommender

    # モデルのロード
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
        print(f"  ⚠️  チェックポイント未検出 — ランダム重みで推論")

    rec = Recommender(model, builder, data, cfg)
    graph_builder = RecommendationGraphBuilder(cfg)

    # --- 類似アイテムモード ---
    if args.similar and args.item_id:
        similar = rec.find_similar_items(args.item_id, args.top_k)
        G = graph_builder.build_similar_items_graph(args.item_id, similar)
        title = f"類似アイテム: item_id={args.item_id}"

    # --- 複数ユーザーモード ---
    elif args.user_ids:
        uids = [int(x.strip()) for x in args.user_ids.split(",")]
        user_recs = {}
        for uid in uids:
            user_recs[uid] = rec.recommend_for_user(uid, args.top_k)
        G = graph_builder.build_multi_user_graph(
            user_recs, show_categories=not args.no_categories,
        )
        title = f"複数ユーザー比較: {uids}"

    # --- 単一ユーザーモード ---
    else:
        uid = args.user_id
        if uid is None:
            row = graph_builder.db.fetch_one("SELECT user_id FROM users ORDER BY user_id LIMIT 1")
            uid = row["user_id"] if row else 1
        recs = rec.recommend_for_user(uid, args.top_k)
        G = graph_builder.build_user_recommendation_graph(
            uid, recs,
            show_views=args.show_views,
            show_favorites=args.show_favorites,
            show_categories=not args.no_categories,
        )
        user_info = graph_builder.db.fetch_one(
            "SELECT username FROM users WHERE user_id=%s", (uid,)
        )
        uname = user_info["username"] if user_info else f"ID={uid}"
        title = f"GNN レコメンド: {uname} (top-{args.top_k})"

    # 統計情報
    print_graph_stats(G)

    # 出力
    if args.static or args.output.endswith((".png", ".jpg", ".pdf", ".svg")):
        visualize_matplotlib(G, args.output, title)
    else:
        visualize_pyvis(G, args.output, title)

    print(f"\n🎉 完了! → {args.output}")


if __name__ == "__main__":
    main()
