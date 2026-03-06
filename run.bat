@echo off
REM ============================================================
REM run.bat - PostgreSQL 環境変数を設定してから Python を実行
REM
REM 使い方:
REM   run.bat main.py --all --db-password パスワード
REM   run.bat evaluate.py --db-password パスワード
REM   run.bat visualize.py --user-id 1 --output network.html --db-password パスワード
REM ============================================================

set PGCLIENTENCODING=UTF8
set PGPASSFILE=NUL
set PGSERVICEFILE=NUL

python %*
