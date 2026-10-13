@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\streamlit.exe" (
    echo Virtual environment not found in .venv!
    pause
    exit /b 1
)
echo ============================================================
echo   Starting Cartoon Violence Detection Web Application...
echo   URL: http://localhost:8501
echo ============================================================
".venv\Scripts\streamlit.exe" run app.py
pause
