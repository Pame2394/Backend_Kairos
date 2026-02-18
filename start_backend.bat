@echo off
cd /d %~dp0
call "%~dp0\.venv\Scripts\activate.bat"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
pause
