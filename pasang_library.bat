@echo off
cd /d "%~dp0"
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib scipy networkx osmnx geopandas requests earthengine-api streamlit folium streamlit-folium
echo.
echo Selesai memasang library.
pause
