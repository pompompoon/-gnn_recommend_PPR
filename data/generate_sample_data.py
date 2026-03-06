"""
サンプルデータ生成 → PostgreSQL インサート

各ユーザーに嗜好プロファイル（好みカテゴリ・ブランド・価格帯）を割り当て、
プロファイルに基づいた確率でインタラクションを生成する。
  - 閲覧 → 購入は 70% の確率で閲覧済みアイテムから発生
  - お気に入り → 閲覧・購入アイテムからの比率が高い
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

import numpy as np
from faker import Faker

from config import Config
from db.connection import DatabaseManager
from data.categories import (
    CATEGORY_HIERARCHY, BRANDS, COLORS, SEASONS,
    PRICE_RANGES, ITEM_TEMPLATES,
)

fake = Faker("ja_JP")


# =================================================================
# Public entry point
# =================================================================

def generate_all(config: Config) -> None:
    """全サンプルデータを生成して DB にインサート"""
    db = DatabaseManager(config.db.connect_kwargs)
    rng = np.random.default_rng(config.data.seed)
    random.seed(config.data.seed)
    Faker.seed(config.data.seed)

    print("\n📦 サンプルデータ生成")
    print("=" * 60)

    db.init_schema()
    db.truncate_all()

    cat_map, subcat_map = _insert_categories(db)
    brand_map = _insert_brands(db)
    item_rows = _generate_and_insert_items(
        db, config.data.n_items, subcat_map, brand_map, rng,
    )
    _generate_and_insert_users(db, config.data.n_users, rng)
    _generate_interactions(db, config.data, item_rows, rng)
    _print_stats(db)


# =================================================================
# Internal helpers
# =================================================================

def _insert_categories(db: DatabaseManager) -> tuple[dict[str, int], dict[str, int]]:
    cat_map: dict[str, int] = {}
    subcat_map: dict[str, int] = {}

    for sort_idx, (cat_name, info) in enumerate(CATEGORY_HIERARCHY.items()):
        db.execute(
            "INSERT INTO categories (name, name_en, sort_order) "
            "VALUES (%s, %s, %s) ON CONFLICT (name) DO NOTHING",
            (cat_name, info["name_en"], sort_idx),
        )
        row = db.fetch_one("SELECT category_id FROM categories WHERE name=%s", (cat_name,))
        cat_id = row["category_id"]
        cat_map[cat_name] = cat_id

        for sub_idx, (sub_name, sub_en) in enumerate(info["subcategories"]):
            db.execute(
                "INSERT INTO subcategories (category_id, name, name_en, sort_order) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT (category_id, name) DO NOTHING",
                (cat_id, sub_name, sub_en, sub_idx),
            )
            row = db.fetch_one(
                "SELECT subcategory_id FROM subcategories "
                "WHERE category_id=%s AND name=%s", (cat_id, sub_name),
            )
            subcat_map[sub_name] = row["subcategory_id"]

    print(f"  📁 カテゴリ {len(cat_map)} / サブカテゴリ {len(subcat_map)}")
    return cat_map, subcat_map


def _insert_brands(db: DatabaseManager) -> dict[str, int]:
    brand_map: dict[str, int] = {}
    for name in BRANDS:
        db.execute(
            "INSERT INTO brands (name) VALUES (%s) ON CONFLICT (name) DO NOTHING",
            (name,),
        )
        row = db.fetch_one("SELECT brand_id FROM brands WHERE name=%s", (name,))
        brand_map[name] = row["brand_id"]
    print(f"  🏷️  ブランド {len(brand_map)}")
    return brand_map


def _generate_and_insert_items(
    db: DatabaseManager,
    n: int,
    subcat_map: dict[str, int],
    brand_map: dict[str, int],
    rng: np.random.Generator,
) -> list[dict]:
    """アイテムを生成して INSERT、全行情報を返す"""
    subcategories = list(subcat_map.keys())
    brands = list(brand_map.keys())
    rows = []

    for _ in range(n):
        sub_name = str(rng.choice(subcategories))
        brand_name = str(rng.choice(brands))
        color = str(rng.choice(COLORS))
        season = str(rng.choice(SEASONS))

        templates = ITEM_TEMPLATES.get(sub_name, ["{b} " + sub_name])
        name = str(rng.choice(templates)).format(b=brand_name)
        name = f"{name} ({color})"

        lo, hi = PRICE_RANGES.get(sub_name, (5_000, 30_000))
        price = int(np.clip(rng.normal((lo + hi) / 2, (hi - lo) / 4), lo, hi))
        price = round(price, -2)

        is_on_sale = bool(rng.random() < 0.15)

        rows.append(dict(
            name=name,
            subcategory_id=subcat_map[sub_name],
            brand_id=brand_map[brand_name],
            price=price,
            color=color,
            season=season,
            is_on_sale=is_on_sale,
            description=f"{brand_name} {season}。{color}の{sub_name}。",
        ))

    values = [
        (r["name"], r["subcategory_id"], r["brand_id"],
         r["price"], r["color"], r["season"], r["is_on_sale"], r["description"])
        for r in rows
    ]
    db.execute_values(
        "INSERT INTO items "
        "(name, subcategory_id, brand_id, price, color, season, is_on_sale, description) "
        "VALUES %s",
        values,
    )
    # item_id を取得して rows に追加
    all_items = db.fetch_all("SELECT item_id, subcategory_id, brand_id, price FROM items ORDER BY item_id")
    for i, item in enumerate(all_items):
        if i < len(rows):
            rows[i]["item_id"] = item["item_id"]

    print(f"  👕 アイテム {len(rows)}")
    return rows


def _generate_and_insert_users(
    db: DatabaseManager, n: int, rng: np.random.Generator,
) -> None:
    prefectures = [
        "東京都", "神奈川県", "大阪府", "愛知県", "埼玉県",
        "千葉県", "兵庫県", "北海道", "福岡県", "京都府",
        "静岡県", "広島県", "宮城県",
    ]
    values = []
    for _ in range(n):
        values.append((
            fake.unique.user_name(),
            int(rng.integers(18, 60)),
            str(rng.choice(["male", "female", "other"], p=[0.50, 0.45, 0.05])),
            str(rng.choice(prefectures)),
        ))
    db.execute_values(
        "INSERT INTO users (username, age, gender, prefecture) VALUES %s",
        values,
    )
    print(f"  👤 ユーザー {n}")


# -----------------------------------------------------------------
# インタラクション生成
# -----------------------------------------------------------------

def _build_user_profiles(
    user_ids: list[int],
    cat_ids: list[int],
    brand_ids: list[int],
    rng: np.random.Generator,
) -> dict[int, dict]:
    """ユーザーごとの嗜好プロファイルを生成"""
    profiles = {}
    for uid in user_ids:
        fav_cats = set(int(x) for x in rng.choice(cat_ids, size=int(rng.integers(1, 4)), replace=False))
        fav_brands = set(int(x) for x in rng.choice(brand_ids, size=int(rng.integers(2, 6)), replace=False))
        price_pref = float(rng.beta(2, 2))
        profiles[uid] = dict(fav_cats=fav_cats, fav_brands=fav_brands, price_pref=price_pref)
    return profiles


def _affinity_scores(
    profile: dict,
    items: list[dict],
    max_price: float,
) -> np.ndarray:
    """プロファイルに基づくアイテムごとの嗜好スコア → 確率分布"""
    n = len(items)
    scores = np.ones(n, dtype=np.float64)
    for idx, it in enumerate(items):
        if it["category_id"] in profile["fav_cats"]:
            scores[idx] *= 3.0
        if it["brand_id"] in profile["fav_brands"]:
            scores[idx] *= 2.0
        price_norm = min(it["price"] / max_price, 1.0)
        scores[idx] *= max(0.2, 1.0 - abs(price_norm - profile["price_pref"]))
    scores /= scores.sum()
    return scores


def _generate_interactions(
    db: DatabaseManager,
    data_cfg,
    item_rows: list[dict],
    rng: np.random.Generator,
) -> None:
    """閲覧 → 購入 → お気に入りの順で生成"""
    # マスタ取得
    users = db.fetch_all("SELECT user_id FROM users ORDER BY user_id")
    user_ids = [u["user_id"] for u in users]

    items_full = db.fetch_all(
        "SELECT i.item_id, i.subcategory_id, i.brand_id, i.price, "
        "       s.category_id "
        "FROM items i "
        "JOIN subcategories s ON i.subcategory_id = s.subcategory_id "
        "ORDER BY i.item_id"
    )
    item_ids = [it["item_id"] for it in items_full]
    n_items = len(item_ids)
    max_price = max(it["price"] for it in items_full)

    cats = db.fetch_all("SELECT category_id FROM categories")
    cat_ids = [c["category_id"] for c in cats]
    brands = db.fetch_all("SELECT brand_id FROM brands")
    brand_ids = [b["brand_id"] for b in brands]

    profiles = _build_user_profiles(user_ids, cat_ids, brand_ids, rng)
    base_time = datetime(2025, 4, 1)
    span_sec = 240 * 86400  # 240日分

    # ----- 閲覧 -----
    print("  📖 閲覧データ生成中...")
    view_values = []
    viewed_map: dict[int, list[int]] = {}

    for _ in range(data_cfg.n_views):
        uid = int(rng.choice(user_ids))
        probs = _affinity_scores(profiles[uid], items_full, max_price)
        idx = int(rng.choice(n_items, p=probs))
        iid = int(item_ids[idx])
        dur = max(3, int(rng.exponential(35)))
        ts = base_time + timedelta(seconds=int(rng.integers(0, span_sec)))
        view_values.append((uid, iid, dur, ts))
        viewed_map.setdefault(uid, []).append(iid)

    db.execute_values(
        "INSERT INTO views (user_id, item_id, duration_sec, viewed_at) VALUES %s",
        view_values,
    )

    # ----- 購入 -----
    print("  🛒 購入データ生成中...")
    purchase_values = []
    for _ in range(data_cfg.n_purchases):
        uid = int(rng.choice(user_ids))
        # 70% は閲覧済みアイテムから購入
        if uid in viewed_map and rng.random() < 0.70:
            iid = int(rng.choice(viewed_map[uid]))
        else:
            probs = _affinity_scores(profiles[uid], items_full, max_price)
            idx = int(rng.choice(n_items, p=probs))
            iid = int(item_ids[idx])
        qty = int(rng.choice([1, 1, 1, 2], p=[0.80, 0.10, 0.05, 0.05]))
        ts = base_time + timedelta(seconds=int(rng.integers(0, span_sec)))
        purchase_values.append((uid, iid, qty, ts))

    db.execute_values(
        "INSERT INTO purchases (user_id, item_id, quantity, purchased_at) VALUES %s",
        purchase_values,
    )

    # ----- お気に入り -----
    print("  ❤️  お気に入りデータ生成中...")
    fav_set: set[tuple[int, int]] = set()
    attempts = 0
    while len(fav_set) < data_cfg.n_favorites and attempts < data_cfg.n_favorites * 5:
        uid = int(rng.choice(user_ids))
        probs = _affinity_scores(profiles[uid], items_full, max_price)
        idx = int(rng.choice(n_items, p=probs))
        iid = int(item_ids[idx])
        fav_set.add((uid, iid))
        attempts += 1

    fav_values = [
        (uid, iid, base_time + timedelta(seconds=int(rng.integers(0, span_sec))))
        for uid, iid in fav_set
    ]
    db.execute_values(
        "INSERT INTO favorites (user_id, item_id, favorited_at) VALUES %s",
        fav_values,
    )


def _print_stats(db: DatabaseManager) -> None:
    tables = ["categories", "subcategories", "brands", "items",
              "users", "purchases", "views", "favorites"]
    print("\n✅ データ生成完了:")
    for t in tables:
        print(f"  {t:20s}  {db.count(t):>8,}")
    print()
