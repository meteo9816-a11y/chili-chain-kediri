@echo off
cd /d "%~dp0"
echo === MODUL 2 (Earth Engine) ===
python kediri_modul2_gee.py
if errorlevel 1 goto gagal
echo === MODUL 3 (waktu tempuh) ===
python kediri_modul3_waktu_tempuh.py
if errorlevel 1 goto gagal
echo === MODUL 1 (SAIPI) ===
python kediri_modul1_analisis.py
if errorlevel 1 goto gagal
echo.
echo SEMUA SELESAI. Lihat folder hasil_modul1, hasil_modul2, hasil_modul3.
pause
exit /b 0
:gagal
echo.
echo Ada langkah yang gagal. Baca pesan error di atas lalu kirim ke Claude.
pause
