#!/usr/bin/env python3
"""
GNN Fashion Recommendation System ─ メインエントリーポイント

Usage
-----
  # 全パイプライン (データ生成 → 学習 → レコメンド表示)
  python main.py --all

  # 個別ステップ
  python main.py --generate-data
  python main.py --train --epochs 100
  python main.py --recommend --user-id 1 --top-k 10
  python main.py --similar-items --item-id 42 --top-k 5
  python main.py --save-recommendations
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import Config
from data.generate_sample_data import generate_all
from models.graph_builder import GraphBuilder
from models.gnn_model import GNNRecommender
from models.train import Trainer
from models.recommend import Recommender


# =================================================================
# CLI
# =================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GNN Fashion Recommender")

    # Actions
    p.add_argument("--generate-data", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--recommend", action="store_true")
    p.add_argument("--similar-items", action="store_true")
    p.add_argument("--save-recommendations", action="store_true")
    p.add_argument("--all", action="store_true",
                   help="generate-data → train → recommend")

    # Params
    p.add_argument("--user-id", type=int, default=1)
    p.add_argument("--item-id", type=int, default=1)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--epochs", type=int, default=None)

    # Encoder
    p.add_argument("--encoder", choices=["gat", "appnp"], default="gat",
                   help="エンコーダー: gat (Attention) / appnp (PPR)")
    p.add_argument("--teleport", type=float, default=0.15,
                   help="APPNP: テレポート確率 α")
    p.add_argument("--ppr-iters", type=int, default=10,
                   help="APPNP: PPR 反復回数 K")

    # DB
    p.add_argument("--db-host", default="localhost")
    p.add_argument("--db-port", type=int, default=5432)
    p.add_argument("--db-name", default="fashion_recommend")
    p.add_argument("--db-user", default="postgres")
    p.add_argument("--db-password", default="postgres")

    return p.parse_args()


# =================================================================
# Builders
# =================================================================

def make_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.db.host = args.db_host
    cfg.db.port = args.db_port
    cfg.db.dbname = args.db_name
    cfg.db.user = args.db_user
    cfg.db.password = args.db_password
    if args.epochs is not None:
        cfg.model.epochs = args.epochs
    cfg.model.encoder_type = args.encoder
    cfg.model.teleport_prob = args.teleport
    cfg.model.num_iterations = args.ppr_iters
    # auto-detect CUDA
    if torch.cuda.is_available():
        cfg.model.device = "cuda"
    return cfg


def build_graph_and_model(cfg: Config):
    """Graph 構築 → Model 初期化"""
    print("\n📐 グラフ構築中 …")
    builder = GraphBuilder(cfg.db.connect_kwargs)
    data = builder.build()

    in_ch = {nt: data[nt].x.shape[1] for nt in data.node_types}
    mc = cfg.model
    enc = mc.encoder_type.upper()
    if mc.encoder_type == "appnp":
        print(f"🧠 モデル初期化  encoder={enc}  α={mc.teleport_prob}  K={mc.num_iterations}  "
              f"hidden={mc.hidden_channels}  out={mc.out_channels}")
    else:
        print(f"🧠 モデル初期化  encoder={enc}  hidden={mc.hidden_channels}  out={mc.out_channels}  "
              f"layers={mc.num_layers}  heads={mc.heads}")

    model = GNNRecommender(
        metadata=data.metadata(),
        in_channels_dict=in_ch,
        hidden_channels=mc.hidden_channels,
        out_channels=mc.out_channels,
        num_layers=mc.num_layers,
        heads=mc.heads,
        dropout=mc.dropout,
        encoder_type=mc.encoder_type,
        teleport_prob=mc.teleport_prob,
        num_iterations=mc.num_iterations,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  パラメータ数: {n_params:,}")

    return model, data, builder


def load_checkpoint(model: GNNRecommender, device: str, path: str = "checkpoints/best_model.pt"):
    ckpt = Path(path)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        print(f"  ✅ チェックポイント読込: {ckpt}")
    else:
        print(f"  ⚠️  チェックポイント未検出 ({ckpt}) — ランダム重みで推論します")


# =================================================================
# Main
# =================================================================

def main():
    args = parse_args()
    cfg = make_config(args)

    print("=" * 72)
    print("  GNN Fashion Recommendation System")
    print("=" * 72)

    # --- generate ---
    if args.generate_data or args.all:
        generate_all(cfg)

    # --- train ---
    if args.train or args.all:
        model, data, builder = build_graph_and_model(cfg)
        trainer = Trainer(model, data, cfg.model)
        trainer.train(epochs=args.epochs)

    # --- recommend ---
    if args.recommend or args.all:
        model, data, builder = build_graph_and_model(cfg)
        load_checkpoint(model, cfg.model.device)
        rec = Recommender(model, builder, data, cfg)

        if args.all:
            # 全パイプライン時は先頭 3 ユーザーを表示
            from db.connection import DatabaseManager
            db = DatabaseManager(cfg.db.connect_kwargs)
            sample_users = db.fetch_all("SELECT user_id FROM users ORDER BY user_id LIMIT 3")
            for row in sample_users:
                rec.print_user_recommendations(row["user_id"], args.top_k)
        else:
            rec.print_user_recommendations(args.user_id, args.top_k)

    # --- similar items ---
    if args.similar_items:
        model, data, builder = build_graph_and_model(cfg)
        load_checkpoint(model, cfg.model.device)
        rec = Recommender(model, builder, data, cfg)

        items = rec.find_similar_items(args.item_id, args.top_k)
        # 対象アイテム情報
        from db.connection import DatabaseManager
        db = DatabaseManager(cfg.db.connect_kwargs)
        target = db.fetch_one(
            "SELECT i.name, s.name AS sub, c.name AS cat "
            "FROM items i "
            "JOIN subcategories s ON i.subcategory_id = s.subcategory_id "
            "JOIN categories c ON s.category_id = c.category_id "
            "WHERE i.item_id = %s", (args.item_id,),
        )
        if target:
            print(f"\n  🔍 基準: {target['name']}  [{target['cat']}>{target['sub']}]")
        if items:
            print(f"  🎯 類似アイテム Top-{args.top_k}:")
            for it in items:
                print(f"     {it['rank']:2d}. {it['name']}"
                      f"  [{it['category']}>{it['subcategory']}]"
                      f"  ¥{it['price']:,}  sim={it['similarity']:.4f}")
        print()

    # --- save ---
    if args.save_recommendations:
        model, data, builder = build_graph_and_model(cfg)
        load_checkpoint(model, cfg.model.device)
        rec = Recommender(model, builder, data, cfg)
        rec.save_to_db(top_k=20)

    # --- no action ---
    if not any([
        args.generate_data, args.train, args.recommend,
        args.similar_items, args.save_recommendations, args.all,
    ]):
        print("""
使い方:
  python main.py --all                           # 全パイプライン一括実行
  python main.py --generate-data                 # サンプルデータ生成
  python main.py --train --epochs 100            # モデル学習
  python main.py --recommend --user-id 5         # レコメンド表示
  python main.py --similar-items --item-id 42    # 類似アイテム検索
  python main.py --save-recommendations          # DB にレコメンド結果を保存
  python main.py --help                          # ヘルプ
""")


if __name__ == "__main__":
    main()
