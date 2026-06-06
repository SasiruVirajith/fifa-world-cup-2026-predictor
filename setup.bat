@echo off
REM World Cup Predictor - environment setup (Windows)
REM Requires Python 3.12: py -3.12

py -3.12 -m venv venv 2>nul || python -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup complete. Activate with: venv\Scripts\activate
echo Run dashboard: streamlit run app.py
