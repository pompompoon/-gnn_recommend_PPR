"""
プロジェクト全体の設定
"""
from dataclasses import dataclass, field


@dataclass
class DBConfig:
    """PostgreSQL 接続設定"""
    host: str = "localhost"
    port: int = 5432
    dbname: str = "fashion_recommend"
    user: str = "postgres"
    password: str = "postgres"

    @property
    def connect_kwargs(self) -> dict:
        """psycopg2.connect() に渡すキーワード引数（DSN文字列より安全）"""
        return dict(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password,
        )


@dataclass
class DataConfig:
    """サンプルデータ生成の設定"""
    n_users: int = 500
    n_items: int = 350          # 800 → 350 (密度 UP)
    n_purchases: int = 25000    # 10000 → 25000
    n_views: int = 60000        # 30000 → 60000
    n_favorites: int = 12000    # 6000  → 12000
    seed: int = 42


@dataclass
class ModelConfig:
    """GNN モデルのハイパーパラメータ"""
    embedding_dim: int = 64
    hidden_channels: int = 128
    out_channels: int = 64
    num_layers: int = 3
    heads: int = 4
    dropout: float = 0.2
    learning_rate: float = 5e-3
    weight_decay: float = 1e-5
    epochs: int = 200
    batch_size: int = 1024
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    early_stop_patience: int = 50  # 5エポック毎に評価するため、実質10回の評価猶予
    device: str = "cpu"  # "cuda" if available
    encoder_type: str = "gat"       # "gat" or "appnp"
    teleport_prob: float = 0.15     # APPNP: PPR テレポート確率 α
    num_iterations: int = 10        # APPNP: PPR 反復回数 K


@dataclass
class Config:
    """統合設定"""
    db: DBConfig = field(default_factory=DBConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
