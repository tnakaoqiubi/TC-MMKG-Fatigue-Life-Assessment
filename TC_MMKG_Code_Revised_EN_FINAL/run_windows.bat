@echo off
cd /d %~dp0
if not exist .env (
  echo [INFO] .env not found. Copy .env.example to .env and fill API/Neo4j credentials.
)
python app.py
pause
