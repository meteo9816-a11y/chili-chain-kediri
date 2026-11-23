#!/usr/bin/env python3
"""
KEDIRI VALUECHAIN - Modul 2 (Geospatial & Agro-Climatic Intelligence) dengan Google Earth Engine

Keluaran utama: data_modul2.csv (satu baris per kecamatan), otomatis dibaca skrip Modul 1/SAIPI.
Pengolahan SPI-3 dilakukan di pandas (bukan di GEE) supaya transparan dan bisa diaudit.

PERSIAPAN (sekali saja):
    pip install earthengine-api geopandas pandas numpy scipy
    earthengine authenticate          # login akun Google yang terdaftar di Earth Engine

JALANKAN (cukup tanpa argumen jika semua file ada di D:\\LOMBA_BI):
    python kediri_modul2_gee.py

OPSI TAMBAHAN:
    --ndvi             tambah NDVI median Sentinel-2 SR pada lahan pertanian (lebih lambat)
    --simplify 0.0002  penyederhanaan poligon (derajat) agar permintaan ke GEE tidak terlalu besar
    --project ID_GCP   override ID Google Cloud project Earth Engine
    --batas PATH_SHP   override file batas kecamatan

Dataset:
    CHIRPS Daily     UCSB-CHG/CHIRPS/DAILY            (~5,5 km)
    ERA5-Land        ECMWF/ERA5_LAND/MONTHLY_AGGR     (~11 km)
    Dynamic World    GOOGLE/DYNAMICWORLD/V1           (10 m)
    Copernicus DEM   COPERNICUS/DEM/GLO30             (30 m)
    Sentinel-2 SR    COPERNICUS/S2_SR_HARMONIZED      (opsional)

CATATAN METODOLOGI (tuliskan di proposal):
- "Cropland" Dynamic World = semua tanaman semusim, BUKAN khusus cabai rawit.
    Statistik cabai tetap bersumber dari BPS; citra hanya memberi konteks lahan.
- CHIRPS dan ERA5-Land berresolusi kasar; nilai tiap kecamatan adalah estimasi rata-rata
    wilayah, dan kecamatan bertetangga bisa memiliki nilai hampir sama.
"""
import argparse
import calendar
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --- util lokasi file & konsol (otomatis; tidak perlu diubah) ---------------
BASE = Path(__file__).resolve().parent     # folder tempat skrip ini berada (mis. D:\LOMBA_BI)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def cari_csv():
    for nama in ("Rekap_Cabai_Rawit_Kediri_2021_2025-v3.csv", "Rekap_Cabai_Rawit_Kediri_2021_2025.csv"):
        if (BASE / nama).exists():
            return str(BASE / nama)
    sisa = sorted(BASE.glob("Rekap_Cabai_Rawit*.csv"))
    if sisa:
        return str(sisa[-1])
    sys.exit(f"CSV data cabai tidak ditemukan di {BASE}. Taruh file Rekap_Cabai_Rawit_*.csv di folder ini.")


# ----------------------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------------------
BASELINE = (1981, 2020)          # klimatologi referensi SPI
EVALUASI = (2021, 2025)          # periode evaluasi (selaras dengan data BPS)
SPI_AMBANG = 1.5                 # |SPI-3| >= 1,5 dianggap ekstrem
AMBANG_CROPS = 0.40              # rata-rata probabilitas 'crops' Dynamic World utk mask lahan tanam
PROJECT_GCP = "earth-engine-research-459913"
MIN_HARI_CHIRPS = 0.9            # bulan CHIRPS dianggap lengkap bila >= 90% hari ada datanya


def _norm(nama):
    n = str(nama).upper().replace("KECAMATAN", "").replace("KEC.", "")
    return "".join(ch for ch in n if ch.isalnum())


def cari_batas():
    prioritas = [
        BASE / "KAB_KEDIRI_26_KEC.shp",
        Path(r"D:\LOMBA_BI\KAB_KEDIRI_26_KEC.shp"),
    ]
    for p in prioritas:
        if p.exists():
            return str(p)

    kand = [p for ext in ("*.shp", "*.geojson", "*.gpkg", "*.json") for p in BASE.glob(ext)]
    if not kand:
        return None
    pilih = ([p for p in kand if "kec" in p.name.lower()]
             or [p for p in kand if "kediri" in p.name.lower()] or kand)
    return str(sorted(pilih)[0])


def deteksi_kolom(gdf, nama_list):
    """Cari kolom yang isinya paling banyak cocok dengan nama kecamatan di CSV."""
    if "KECAMATAN" in gdf.columns:
        skor_def = sum(_norm(v) in {_norm(n) for n in nama_list} for v in gdf["KECAMATAN"].dropna().astype(str))
        if skor_def >= 10:
            return "KECAMATAN", skor_def

    kunci = {_norm(n) for n in nama_list}
    terbaik, skor = None, 0
    for c in gdf.columns:
        if c == "geometry":
            continue
        s = sum(_norm(v) in kunci for v in gdf[c].dropna().astype(str))
        if s > skor:
            terbaik, skor = c, s
    return terbaik, skor


# ----------------------------------------------------------------------------
# FUNGSI MURNI PYTHON (bisa diuji tanpa GEE)
# ----------------------------------------------------------------------------
def hitung_spi3(df, baseline=BASELINE):
    """
    df: kolom [Kecamatan, tahun, bulan, p_mm] (curah hujan bulanan).
    Return: df dengan kolom tambahan p3_mm (jumlah 3 bulan) dan spi3.
    SPI-3: jumlah 3 bulan, fit gamma per bulan kalender pada periode baseline,
    dengan koreksi probabilitas nol (H = q + (1-q) G(x)), lalu transformasi ke normal baku.
    """
    hasil = []
    for kec, g in df.groupby("Kecamatan"):
        g = g.sort_values(["tahun", "bulan"]).copy()
        g["tanggal"] = pd.to_datetime(dict(year=g["tahun"], month=g["bulan"], day=1))
        g = g.set_index("tanggal")
        idx = pd.date_range(g.index.min(), g.index.max(), freq="MS")
        g = g.reindex(idx)
        g["Kecamatan"] = kec
        g["tahun"], g["bulan"] = g.index.year, g.index.month
        g["p3_mm"] = g["p_mm"].rolling(3, min_periods=3).sum()
        g["spi3"] = np.nan
        for m in range(1, 13):
            sel = g["bulan"] == m
            x_all = g.loc[sel, "p3_mm"]
            x_base = g.loc[sel & g["tahun"].between(*baseline), "p3_mm"].dropna()
            if len(x_base) < 20:
                continue
            q = (x_base <= 0).mean()
            pos = x_base[x_base > 0]
            if len(pos) < 10:
                continue
            a, loc, scale = stats.gamma.fit(pos, floc=0)
            H = q + (1 - q) * stats.gamma.cdf(x_all.clip(lower=0), a, loc=loc, scale=scale)
            H = np.clip(H, 1e-6, 1 - 1e-6)
            g.loc[sel, "spi3"] = stats.norm.ppf(H)
        hasil.append(g.reset_index(drop=True))
    return pd.concat(hasil, ignore_index=True)


def kelembapan_relatif(t_c, td_c):
    """RH (%) dari suhu dan titik embun (Magnus). Dari rata-rata bulanan = pendekatan kasar."""
    a, b = 17.625, 243.04
    return 100 * np.exp(a * td_c / (b + td_c)) / np.exp(a * t_c / (b + t_c))


def ringkas_spi(spi_df, evaluasi=EVALUASI, ambang=SPI_AMBANG):
    d = spi_df[spi_df["tahun"].between(*evaluasi)].dropna(subset=["spi3"])
    out = d.groupby("Kecamatan")["spi3"].agg(
        spi_n_bulan="count",
        spi_min="min",
        spi_max="max",
    )
    out["spi_kering_share"] = d.groupby("Kecamatan")["spi3"].apply(lambda s: (s <= -ambang).mean())
    out["spi_basah_share"] = d.groupby("Kecamatan")["spi3"].apply(lambda s: (s >= ambang).mean())
    out["spi_ekstrem"] = out["spi_kering_share"] + out["spi_basah_share"]
    return out.reset_index()


# ----------------------------------------------------------------------------
# GEE
# ----------------------------------------------------------------------------
def inisialisasi_ee(project):
    import ee
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
    return ee


def muat_kecamatan(ee, batas, kolom, nama_list, simplify):
    import geopandas as gpd
    gdf = gpd.read_file(batas)
    if not kolom:
        kolom, skor = deteksi_kolom(gdf, nama_list)
        if not kolom or skor < 10:
            sys.exit(f"Kolom nama kecamatan tidak terdeteksi di {batas}. Kolom tersedia: "
                     f"{list(gdf.columns)}. Jalankan dengan --kolom-nama NAMA_KOLOM.")
        print(f"Kolom nama kecamatan terdeteksi otomatis: {kolom} ({skor} nama cocok)")
    if kolom not in gdf.columns:
        sys.exit(f"Kolom '{kolom}' tidak ada di {batas}. Kolom tersedia: {list(gdf.columns)}")
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)
    gdf["_key"] = gdf[kolom].map(_norm)
    peta = {_norm(n): n for n in nama_list}
    gdf = gdf[gdf["_key"].isin(peta)].copy()
    hilang = [peta[k] for k in peta if k not in set(gdf["_key"])]
    if hilang:
        print("PERINGATAN: kecamatan tidak ditemukan di file batas:", hilang)
    gdf["Kecamatan"] = gdf["_key"].map(peta)
    gdf = gdf.dissolve(by="Kecamatan", as_index=False)
    gdf["luas_wilayah_ha"] = gdf.to_crs(32749).area / 10_000
    gdf_simple = gdf.copy()
    gdf_simple["geometry"] = gdf_simple.geometry.simplify(simplify)
    gj = json.loads(gdf_simple[["Kecamatan", "geometry"]].to_json())
    return ee.FeatureCollection(gj), gdf[["Kecamatan", "luas_wilayah_ha"]], gdf


def chirps_bulanan(ee, fc, th0, th1):
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")
    rows = []
    for y in range(th0, th1 + 1):
        imgs = []
        for m in range(1, 13):
            start = ee.Date.fromYMD(y, m, 1)
            col = chirps.filterDate(start, start.advance(1, "month"))
            imgs.append(col.sum().set({"ym": y * 100 + m, "n_hari": col.size()}))
        ic = ee.ImageCollection(imgs)

        def red(img):
            return img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=5566).map(
                lambda f: ee.Feature(None, {"Kecamatan": f.get("Kecamatan"), "ym": img.get("ym"),
                                            "n_hari": img.get("n_hari"), "p_mm": f.get("mean")}))

        res = ic.map(red).flatten().getInfo()
        rows += [f["properties"] for f in res["features"]]
        print(f"  CHIRPS {y} selesai")
    df = pd.DataFrame(rows)
    df["tahun"], df["bulan"] = df["ym"] // 100, df["ym"] % 100
    hari = df.apply(lambda r: calendar.monthrange(int(r["tahun"]), int(r["bulan"]))[1], axis=1)
    lengkap = df["n_hari"] >= MIN_HARI_CHIRPS * hari
    if (~lengkap).any():
        bln = sorted(set(zip(df.loc[~lengkap, "tahun"], df.loc[~lengkap, "bulan"])))
        print(f"  PERINGATAN: bulan CHIRPS tidak lengkap dibuang (data terbaru belum rilis): {bln[:6]}...")
    df = df[lengkap & df["p_mm"].notna()]
    return df[["Kecamatan", "tahun", "bulan", "p_mm"]]


def era5_bulanan(ee, fc, th0, th1):
    ic = (ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
          .select(["temperature_2m", "dewpoint_temperature_2m"])
          .filterDate(f"{th0}-01-01", f"{th1 + 1}-01-01"))

    def red(img):
        ym = ee.Date(img.get("system:time_start")).format("YYYYMM")
        return img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=11132).map(
            lambda f: ee.Feature(None, {"Kecamatan": f.get("Kecamatan"), "ym": ym,
                                        "t_k": f.get("temperature_2m"),
                                        "td_k": f.get("dewpoint_temperature_2m")}))

    res = ic.map(red).flatten().getInfo()
    df = pd.DataFrame([f["properties"] for f in res["features"]])
    df["ym"] = df["ym"].astype(int)
    df["tahun"], df["bulan"] = df["ym"] // 100, df["ym"] % 100
    df["t_c"] = df["t_k"] - 273.15
    df["td_c"] = df["td_k"] - 273.15
    df["rh_pct"] = kelembapan_relatif(df["t_c"], df["td_c"])
    return df[["Kecamatan", "tahun", "bulan", "t_c", "td_c", "rh_pct"]]


def fisik_lahan(ee, fc, th0, th1, ambang_crops, pakai_ndvi):
    """Lahan pertanian aktif (Dynamic World), lereng (Copernicus DEM), jarak ke kawasan terbangun."""
    aoi = fc.geometry()
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
          .filterDate(f"{th0}-01-01", f"{th1 + 1}-01-01").filterBounds(aoi))
    crops_p = dw.select("crops").mean()
    mask = crops_p.gte(ambang_crops).rename("mask")
    label = dw.select("label").mode()
    built = label.eq(6)                                     # 6 = built area

    dem_col = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM")
    proj = dem_col.first().select(0).projection()
    dem = dem_col.mosaic().setDefaultProjection(proj)
    slope = ee.Terrain.slope(dem).rename("lereng_deg")

    jarak = (built.fastDistanceTransform(256).sqrt()
             .multiply(ee.Image.pixelArea().sqrt()).rename("jarak_terbangun_m"))

    luas_ha = mask.multiply(ee.Image.pixelArea()).divide(10_000).rename("cropland_ha")
    ll = ee.Image.pixelLonLat()

    bands_mean = (slope.addBands(jarak).addBands(ll.select("longitude").rename("cropland_lon"))
                  .addBands(ll.select("latitude").rename("cropland_lat")).updateMask(mask))
    if pakai_ndvi:
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterDate(f"{th1}-01-01", f"{th1 + 1}-01-01").filterBounds(aoi)
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
              .map(lambda im: im.updateMask(im.select("SCL").neq(3).And(im.select("SCL").neq(8))
                                            .And(im.select("SCL").neq(9)).And(im.select("SCL").neq(10)))
                   .normalizedDifference(["B8", "B4"]).rename("ndvi")))
        bands_mean = bands_mean.addBands(s2.median().updateMask(mask).rename("ndvi_median"))

    r_sum = luas_ha.reduceRegions(collection=fc, reducer=ee.Reducer.sum(), scale=30).getInfo()
    r_mean = bands_mean.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=30).getInfo()
    a = pd.DataFrame([{"Kecamatan": f["properties"]["Kecamatan"], "cropland_ha": f["properties"].get("sum")}
                      for f in r_sum["features"]])
    b = pd.DataFrame([{k: v for k, v in f["properties"].items()} for f in r_mean["features"]])
    b = b.rename(columns={"lereng_deg": "lereng_cropland_deg"})
    return a.merge(b, on="Kecamatan", how="outer")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None, help="ID Google Cloud project untuk Earth Engine")
    ap.add_argument("--batas", default=None, help="file poligon batas kecamatan (otomatis KAB_KEDIRI_26_KEC.shp)")
    ap.add_argument("--kolom-nama", default=None, help="kolom nama kecamatan (otomatis dideteksi)")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--simplify", type=float, default=0.0002)
    ap.add_argument("--ndvi", action="store_true")
    args = ap.parse_args()

    csv = args.csv or cari_csv()
    print(f"Data cabai: {csv}")
    project = args.project or PROJECT_GCP or input("Masukkan ID Google Cloud project untuk Earth Engine: ").strip()
    batas = args.batas or cari_batas()
    if not batas:
        sys.exit(f"File batas kecamatan tidak ditemukan. Taruh file KAB_KEDIRI_26_KEC.shp di {BASE} "
                 "atau pakai --batas path_file.")
    print(f"File batas kecamatan: {batas}")
    folder = Path(args.out) if args.out else BASE / "hasil_modul2"
    folder.mkdir(parents=True, exist_ok=True)
    nama = [k for k in pd.read_csv(csv, dtype=str)["Kecamatan"] if k != "Kabupaten Kediri"]

    ee = inisialisasi_ee(project)
    fc, luas, gdf = muat_kecamatan(ee, batas, args.kolom_nama, nama, args.simplify)

    print("1/4 CHIRPS bulanan (baseline + evaluasi)...")
    p = chirps_bulanan(ee, fc, BASELINE[0], EVALUASI[1])
    p.to_csv(folder / "chirps_bulanan.csv", index=False)
    spi = hitung_spi3(p)
    spi.to_csv(folder / "spi3_bulanan.csv", index=False)
    spi_ring = ringkas_spi(spi)

    print("2/4 ERA5-Land bulanan (evaluasi)...")
    era = era5_bulanan(ee, fc, EVALUASI[0], EVALUASI[1])
    era.to_csv(folder / "era5_bulanan.csv", index=False)
    era_ring = era.groupby("Kecamatan").agg(t_mean_c=("t_c", "mean"), rh_mean_pct=("rh_pct", "mean"),
                                            rh_bulan_lembap_share=("rh_pct", lambda s: (s >= 85).mean())
                                            ).reset_index()

    print("3/4 Lahan pertanian, lereng, jarak ke kawasan terbangun...")
    fis = fisik_lahan(ee, fc, EVALUASI[0], EVALUASI[1], AMBANG_CROPS, args.ndvi)
    fis = fis.merge(luas, on="Kecamatan", how="left")
    fis["cropland_pct_wilayah"] = fis["cropland_ha"] / fis["luas_wilayah_ha"] * 100

    print("4/4 Menyatukan hasil...")
    out = (pd.DataFrame({"Kecamatan": nama}).merge(spi_ring, on="Kecamatan", how="left")
           .merge(era_ring, on="Kecamatan", how="left").merge(fis, on="Kecamatan", how="left"))
    out.round(4).to_csv(BASE / "data_modul2.csv", index=False)

    # Simpan juga output spasial (GeoJSON & SHP) di hasil_modul2/
    gdf_out = gdf[["Kecamatan", "geometry"]].merge(out, on="Kecamatan", how="left")
    gdf_out.to_file(folder / "data_modul2_spasial.geojson", driver="GeoJSON")
    gdf_out.to_file(folder / "data_modul2_spasial.shp")

    # titik pusat kawasan pertanian aktif -> input Modul 3
    rep = gdf.to_crs(32749).geometry.representative_point().to_crs(4326)
    fallback = pd.DataFrame({"Kecamatan": gdf["Kecamatan"], "f_lat": rep.y.values, "f_lon": rep.x.values})
    t = pd.DataFrame({"Kecamatan": nama}).merge(out[["Kecamatan", "cropland_lat", "cropland_lon"]],
                                                on="Kecamatan", how="left").merge(fallback, on="Kecamatan", how="left")
    ada = t["cropland_lat"].notna()
    titik = pd.DataFrame({"Kecamatan": t["Kecamatan"],
                          "lat": np.where(ada, t["cropland_lat"], t["f_lat"]),
                          "lon": np.where(ada, t["cropland_lon"], t["f_lon"]),
                          "sumber": np.where(ada, "centroid cropland Dynamic World", "titik representatif poligon")})
    tujuan = BASE / ("titik_kecamatan.csv" if not (BASE / "titik_kecamatan.csv").exists() else "titik_kecamatan_cropland.csv")
    titik.round(6).to_csv(tujuan, index=False)

    print("\nRINGKASAN (urut tekanan SPI ekstrem):")
    cols = ["Kecamatan", "spi_ekstrem", "spi_kering_share", "spi_basah_share", "t_mean_c",
            "rh_mean_pct", "cropland_ha", "lereng_cropland_deg", "jarak_terbangun_m"]
    print(out.sort_values("spi_ekstrem", ascending=False)[[c for c in cols if c in out.columns]]
          .round(2).to_string(index=False))
    print(f"\nSelesai. data_modul2.csv, {tujuan.name}, dan output spasial di {folder.name}/ telah dibuat.")


if __name__ == "__main__":
    main()