@echo off
cd /d "%~dp0"
echo Membuka simulator di browser... (tutup jendela ini untuk menghentikan)
python -m streamlit run kediri_modul5_simulator.py
pause
