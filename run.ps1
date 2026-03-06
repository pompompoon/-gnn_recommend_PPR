#!/usr/bin/env pwsh
# ============================================================
# run.ps1 - PostgreSQL 環境変数を設定してから Python を実行
#
# Windows 日本語環境で psycopg2 (libpq) が cp932 エラーを
# 起こす問題を、プロセス起動前に環境変数を設定して回避する。
#
# 使い方:
#   .\run.ps1 main.py --all --db-password パスワード
#   .\run.ps1 evaluate.py --db-password パスワード
#   .\run.ps1 visualize.py --user-id 1 --top-k 5 --output network.html --db-password パスワード
#   .\run.ps1 main.py --train --encoder appnp --db-password パスワード
# ============================================================

# PostgreSQL 環境変数を設定（プロセスレベル）
$env:PGCLIENTENCODING = "UTF8"
$env:PGPASSFILE = "NUL"
$env:PGSERVICEFILE = "NUL"

# 引数をそのまま python に渡す
python @args
