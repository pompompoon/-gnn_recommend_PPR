-- =================================================================
-- GNN Fashion Recommendation System - PostgreSQL Schema
-- =================================================================

-- -----------------------------------------------
-- マスタテーブル
-- -----------------------------------------------

-- 大分類カテゴリ
CREATE TABLE IF NOT EXISTS categories (
    category_id   SERIAL PRIMARY KEY,
    name          VARCHAR(100) NOT NULL UNIQUE,
    name_en       VARCHAR(100),
    sort_order    INTEGER DEFAULT 0,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 小分類サブカテゴリ
CREATE TABLE IF NOT EXISTS subcategories (
    subcategory_id  SERIAL PRIMARY KEY,
    category_id     INTEGER NOT NULL REFERENCES categories(category_id) ON DELETE CASCADE,
    name            VARCHAR(100) NOT NULL,
    name_en         VARCHAR(100),
    sort_order      INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (category_id, name)
);

-- ブランド
CREATE TABLE IF NOT EXISTS brands (
    brand_id    SERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL UNIQUE,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------
-- アイテム（商品）
-- -----------------------------------------------
CREATE TABLE IF NOT EXISTS items (
    item_id         SERIAL PRIMARY KEY,
    name            VARCHAR(300) NOT NULL,
    subcategory_id  INTEGER NOT NULL REFERENCES subcategories(subcategory_id),
    brand_id        INTEGER REFERENCES brands(brand_id),
    price           INTEGER NOT NULL CHECK (price > 0),
    color           VARCHAR(50),
    season          VARCHAR(80),
    description     TEXT,
    is_on_sale      BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------
-- ユーザー
-- -----------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    user_id     SERIAL PRIMARY KEY,
    username    VARCHAR(100) NOT NULL UNIQUE,
    age         INTEGER CHECK (age BETWEEN 10 AND 100),
    gender      VARCHAR(20),
    prefecture  VARCHAR(50),
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------
-- インタラクション (行動ログ)
-- -----------------------------------------------

-- 購入
CREATE TABLE IF NOT EXISTS purchases (
    purchase_id   SERIAL PRIMARY KEY,
    user_id       INTEGER NOT NULL REFERENCES users(user_id),
    item_id       INTEGER NOT NULL REFERENCES items(item_id),
    quantity      INTEGER DEFAULT 1 CHECK (quantity > 0),
    purchased_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 閲覧
CREATE TABLE IF NOT EXISTS views (
    view_id       SERIAL PRIMARY KEY,
    user_id       INTEGER NOT NULL REFERENCES users(user_id),
    item_id       INTEGER NOT NULL REFERENCES items(item_id),
    duration_sec  INTEGER DEFAULT 0,
    viewed_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- お気に入り
CREATE TABLE IF NOT EXISTS favorites (
    favorite_id   SERIAL PRIMARY KEY,
    user_id       INTEGER NOT NULL REFERENCES users(user_id),
    item_id       INTEGER NOT NULL REFERENCES items(item_id),
    favorited_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, item_id)
);

-- -----------------------------------------------
-- レコメンド結果キャッシュ
-- -----------------------------------------------
CREATE TABLE IF NOT EXISTS recommendations (
    rec_id         SERIAL PRIMARY KEY,
    user_id        INTEGER NOT NULL REFERENCES users(user_id),
    item_id        INTEGER NOT NULL REFERENCES items(item_id),
    score          DOUBLE PRECISION NOT NULL,
    rank           INTEGER,
    model_version  VARCHAR(50),
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------
-- インデックス
-- -----------------------------------------------
CREATE INDEX IF NOT EXISTS idx_items_subcategory    ON items(subcategory_id);
CREATE INDEX IF NOT EXISTS idx_items_brand          ON items(brand_id);
CREATE INDEX IF NOT EXISTS idx_purchases_user       ON purchases(user_id);
CREATE INDEX IF NOT EXISTS idx_purchases_item       ON purchases(item_id);
CREATE INDEX IF NOT EXISTS idx_purchases_time       ON purchases(purchased_at);
CREATE INDEX IF NOT EXISTS idx_views_user           ON views(user_id);
CREATE INDEX IF NOT EXISTS idx_views_item           ON views(item_id);
CREATE INDEX IF NOT EXISTS idx_favorites_user       ON favorites(user_id);
CREATE INDEX IF NOT EXISTS idx_favorites_item       ON favorites(item_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_user ON recommendations(user_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_ver  ON recommendations(model_version);
