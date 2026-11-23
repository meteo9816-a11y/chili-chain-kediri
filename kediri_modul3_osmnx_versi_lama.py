#!/usr/bin/env python3
"""
KEDIRI VALUECHAIN - Modul 3 (Network Accessibility & Spatial Logistics)

Menghitung waktu tempuh dari titik tiap kecamatan ke Pasar Induk Pare di jaringan
jalan OpenStreetMap, dengan penalti tanjakan dari DEM (grade per ruas jalan).
Hasilnya ditulis ke data_tambahan.csv, yang otomatis dibaca skrip Modul 1/SAIPI.

CARA PAKAI (butuh internet, jalankan di komputermu):
    pip install osmnx networkx pandas numpy matplotlib requests rasterio

    # 1) Pertama kali: geocoding titik kecamatan + unduh jaringan jalan + elevasi
    python kediri_modul3_waktu_tempuh.py

    # 2) Cek titik_kecamatan.csv (perbaiki koordinat yang salah), lalu jalankan lagi.
    #    Jaringan jalan dan elevasi di-cache, jadi run ulang cepat.

    # Opsi:
    #   --dem tile1.tif tile2.tif   pakai Copernicus DEM GLO-30 lokal (disarankan)
    #   --batas kecamatan.geojson   poligon batas kecamatan (BIG/BPS/OSM/GADM); titik tiap
    #   --kolom-nama NAMOBJ         kecamatan = titik representatif poligon, bukan geocoding
    #   --pare lat,lon              ganti koordinat Pasar Induk Pare (default sudah diisi)
    #   --faktor 0.8                skala kecepatan global (kalibrasi, 1.0 = asumsi default)
    #   --selftest                  uji logika dengan graf sintetis (tanpa internet)

Tanpa --dem, elevasi diambil dari API publik OpenTopoData (SRTM 30 m): gratis, tapi
dibatasi 100 titik/permintaan dan 1.000 permintaan/hari. Hasil di-cache, jadi bisa
dilanjutkan keesokan harinya kalau terpotong. Untuk klaim "Copernicus DEM" di proposal,
gunakan --dem dengan tile Copernicus DEM GLO-30.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

# --- util lokasi file & konsol (otomatis; tidak perlu diubah) ---------------
BASE = Path(__file__).resolve().parent     # folder tempat skrip ini berada (mis. D:\\LOMBA_BI)
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
WILAYAH = "Kabupaten Kediri, Jawa Timur, Indonesia"
BUFFER_M = 3000                  # buffer batas kabupaten agar rute tepi tidak terpotong
PARE_DEFAULT = (-7.7472121, 112.1992859)   # Pasar Induk Pare (koordinat dari pengguna)

# Kecepatan jelajah asumsi per kelas jalan (km/jam), truk/pickup bermuatan di jalan pedesaan.
# INI ASUMSI, kalibrasi dengan waktu nyata (lihat kalibrasi_waktu.csv di bawah).
KECEPATAN_KMH = {
    "motorway": 70, "motorway_link": 40,
    "trunk": 50, "trunk_link": 35,
    "primary": 40, "primary_link": 30,
    "secondary": 35, "secondary_link": 25,
    "tertiary": 30, "tertiary_link": 25,
    "unclassified": 25, "residential": 20,
    "living_street": 15, "service": 15, "track": 15,
}
KECEPATAN_DEFAULT = 20

# Penalti tanjakan: faktor = max(F_MIN, 1 - K_TANJAKAN * grade) untuk grade > 0.
# grade 0.06 (6%) -> 0.70; grade 0.10 -> 0.50. Turunan tidak dipercepat.
K_TANJAKAN = 5.0
F_MIN = 0.40
GRADE_CLIP = 0.15                # DEM 30 m kasar pada ruas pendek -> batasi +/-15%
BATAS_SNAP_KM = 3.0              # titik kecamatan > 3 km dari jalan terdekat ditandai


# ----------------------------------------------------------------------------
# UTILITAS GEOMETRI
# ----------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def simpul_terdekat(G, lat, lon):
    """Simpul graf terdekat (haversine). Return (node_id, jarak_snap_km)."""
    ids = np.array(list(G.nodes))
    ys = np.array([G.nodes[n]["y"] for n in ids], float)
    xs = np.array([G.nodes[n]["x"] for n in ids], float)
    d = haversine_km(lat, lon, ys, xs)
    i = int(np.argmin(d))
    return ids[i], float(d[i])


# ----------------------------------------------------------------------------
# ELEVASI
# ----------------------------------------------------------------------------
def elevasi_raster(G, paths):
    """Ambil elevasi simpul dari satu/lebih raster DEM (mis. tile Copernicus GLO-30)."""
    import rasterio
    ids = list(G.nodes)
    elev = {n: np.nan for n in ids}
    for p in paths:
        with rasterio.open(p) as ds:
            b = ds.bounds
            sel = [n for n in ids if np.isnan(elev[n])
                   and b.left <= G.nodes[n]["x"] <= b.right
                   and b.bottom <= G.nodes[n]["y"] <= b.top]
            if not sel:
                continue
            coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in sel]
            vals = [v[0] for v in ds.sample(coords)]
            for n, v in zip(sel, vals):
                if ds.nodata is not None and v == ds.nodata:
                    continue
                elev[n] = float(v)
    return elev


def elevasi_opentopodata(G, cache_path, dataset="srtm30m"):
    """Elevasi via API publik OpenTopoData, dengan cache agar bisa dilanjutkan."""
    import requests
    cache = {}
    if Path(cache_path).exists():
        cache = {int(k): v for k, v in json.loads(Path(cache_path).read_text()).items()}
    ids = [n for n in G.nodes if n not in cache]
    n_batch = math.ceil(len(ids) / 100)
    print(f"  Elevasi: {len(cache)} simpul sudah di-cache, {len(ids)} perlu diambil "
          f"({n_batch} permintaan API; batas harian 1.000)")
    url = f"https://api.opentopodata.org/v1/{dataset}"
    for b in range(n_batch):
        chunk = ids[b * 100:(b + 1) * 100]
        loc = "|".join(f"{G.nodes[n]['y']:.6f},{G.nodes[n]['x']:.6f}" for n in chunk)
        for percobaan in range(5):
            r = requests.get(url, params={"locations": loc}, timeout=60)
            if r.status_code == 200:
                for n, res in zip(chunk, r.json()["results"]):
                    cache[int(n)] = res["elevation"]
                break
            time.sleep(2 * (percobaan + 1))
        else:
            print("  API gagal/terbatas. Simpan cache dan hentikan; jalankan lagi nanti.")
            break
        if b % 20 == 0:
            Path(cache_path).write_text(json.dumps(cache))
            print(f"  ...{b + 1}/{n_batch}")
        time.sleep(1.1)
    Path(cache_path).write_text(json.dumps(cache))
    return {n: (np.nan if cache.get(int(n)) is None else float(cache[int(n)])) for n in G.nodes}


def pasang_elevasi_dan_grade(G, elev):
    for n, z in elev.items():
        G.nodes[n]["elevation"] = z
    tanpa = 0
    for u, v, k, d in G.edges(keys=True, data=True):
        zu, zv, L = elev.get(u, np.nan), elev.get(v, np.nan), d.get("length", 0)
        if L > 0 and not (np.isnan(zu) or np.isnan(zv)):
            d["grade"] = float(np.clip((zv - zu) / L, -GRADE_CLIP, GRADE_CLIP))
        else:
            d["grade"] = 0.0
            tanpa += 1
    if tanpa:
        print(f"  Peringatan: {tanpa} ruas tanpa elevasi lengkap dianggap datar.")


# ----------------------------------------------------------------------------
# WAKTU TEMPUH
# ----------------------------------------------------------------------------
def kecepatan_kelas(d):
    hw = d.get("highway", "unclassified")
    if isinstance(hw, str) and hw.startswith("["):      # hasil load_graphml: "['a', 'b']"
        hw = hw.strip("[]").split(",")[0].strip().strip("'\"")
    if isinstance(hw, (list, tuple)):
        hw = hw[0]
    return KECEPATAN_KMH.get(hw, KECEPATAN_DEFAULT)


def faktor_tanjakan(grade):
    return max(F_MIN, 1 - K_TANJAKAN * grade) if grade > 0 else 1.0


def hitung_waktu_ruas(G, faktor_global=1.0):
    """Isi 'travel_time' (dengan tanjakan) dan 'travel_time_flat' (datar), dalam detik."""
    for u, v, k, d in G.edges(keys=True, data=True):
        v0 = kecepatan_kelas(d) * faktor_global
        ms_flat = v0 * 1000 / 3600
        ms = ms_flat * faktor_tanjakan(d.get("grade", 0.0))
        L = d.get("length", 0)
        d["travel_time_flat"] = L / ms_flat
        d["travel_time"] = L / ms


def waktu_ke_tujuan(G, tujuan, bobot):
    """Waktu terpendek dari SEMUA simpul KE 'tujuan' (arah asal -> pasar)."""
    return nx.single_source_dijkstra_path_length(G.reverse(copy=True), tujuan, weight=bobot)


def panjang_rute(G, asal, tujuan, bobot):
    path = nx.shortest_path(G, asal, tujuan, weight=bobot)
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        total += min(dd.get("length", 0) for dd in G[a][b].values())
    return total / 1000


# ----------------------------------------------------------------------------
# TITIK KECAMATAN
# ----------------------------------------------------------------------------
def _norm(nama):
    n = str(nama).upper().replace("KECAMATAN", "").replace("KEC.", "")
    return "".join(ch for ch in n if ch.isalnum())


def cari_batas():
    kand = [p for ext in ("*.geojson", "*.gpkg", "*.shp", "*.json") for p in BASE.glob(ext)]
    if not kand:
        return None
    pilih = ([p for p in kand if "kec" in p.name.lower()]
             or [p for p in kand if "kediri" in p.name.lower()] or kand)
    return str(sorted(pilih)[0])


def deteksi_kolom(gdf, nama_list):
    """Cari kolom yang isinya paling banyak cocok dengan nama kecamatan di CSV."""
    kunci = {_norm(n) for n in nama_list}
    terbaik, skor = None, 0
    for c in gdf.columns:
        if c == "geometry":
            continue
        s = sum(_norm(v) in kunci for v in gdf[c].dropna().astype(str))
        if s > skor:
            terbaik, skor = c, s
    return terbaik, skor


def titik_dari_batas(nama_list, batas, kolom):
    """Titik representatif (selalu berada DI DALAM poligon) tiap kecamatan dari file batas."""
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
    gdf["_key"] = gdf[kolom].map(_norm)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    pts = gdf.to_crs(32749).geometry.representative_point().to_crs(4326)
    gdf["lat"], gdf["lon"] = pts.y, pts.x
    if gdf["_key"].duplicated().any():
        dup = gdf.loc[gdf["_key"].duplicated(keep=False), kolom].unique()
        print(f"Peringatan: nama kecamatan ganda di file batas: {list(dup)}. "
              "Filter file ke Kabupaten Kediri saja.")
    peta = gdf.drop_duplicates("_key").set_index("_key")[["lat", "lon"]]
    rows = []
    for nama in nama_list:
        k = _norm(nama)
        if k in peta.index:
            rows.append({"Kecamatan": nama, "lat": peta.loc[k, "lat"], "lon": peta.loc[k, "lon"],
                         "sumber": "titik representatif poligon batas"})
        else:
            rows.append({"Kecamatan": nama, "lat": np.nan, "lon": np.nan,
                         "sumber": "NAMA TIDAK COCOK"})
    return pd.DataFrame(rows)


def titik_kecamatan(nama_list, path_csv, G, batas=None, kolom=None):
    path_csv = Path(path_csv)
    if path_csv.exists():
        df = pd.read_csv(path_csv)
        print(f"Memakai titik dari {path_csv} (edit file ini untuk memperbaiki koordinat).")
        return df
    if batas:
        df = titik_dari_batas(nama_list, batas, kolom)
        df.to_csv(path_csv, index=False)
        gagal = df[df["lat"].isna()]
        if len(gagal):
            print("Nama kecamatan ini tidak cocok dengan file batas. Samakan nama di "
                  f"{path_csv} (isi lat/lon manual) atau di file batas, lalu jalankan ulang:")
            print(gagal[["Kecamatan"]].to_string(index=False))
            sys.exit(0)
        print(f"Titik kecamatan dari poligon batas ditulis ke {path_csv}.")
        return df
    import osmnx as ox
    ys = [G.nodes[n]["y"] for n in G.nodes]
    xs = [G.nodes[n]["x"] for n in G.nodes]
    rows = []
    for nama in nama_list:
        lat = lon = np.nan
        sumber = "geocode"
        try:
            lat, lon = ox.geocode(f"Kecamatan {nama}, Kabupaten Kediri, Jawa Timur, Indonesia")
        except Exception:
            sumber = "GAGAL: isi manual"
        ok = (not np.isnan(lat)) and min(ys) <= lat <= max(ys) and min(xs) <= lon <= max(xs)
        if not ok and sumber == "geocode":
            sumber = "DI LUAR AREA: periksa"
        rows.append({"Kecamatan": nama, "lat": lat, "lon": lon, "sumber": sumber})
        time.sleep(1.1)  # kebijakan Nominatim: maks 1 permintaan/detik
    df = pd.DataFrame(rows)
    df.to_csv(path_csv, index=False)
    bermasalah = df[df["sumber"] != "geocode"]
    print(f"Titik kecamatan ditulis ke {path_csv}.")
    if len(bermasalah):
        print("PERIKSA/ISI MANUAL koordinat berikut, lalu jalankan ulang:")
        print(bermasalah.to_string(index=False))
        print("Catatan: hasil geocoding adalah titik pusat kecamatan administratif. Untuk klaim\n"
              "'pusat kawasan pertanian aktif', ganti dengan centroid cropland dari Modul 2.")
        sys.exit(0)
    print("Semua titik berhasil di-geocode. Periksa sekilas di titik_kecamatan.csv, "
          "lalu jalankan ulang.")
    sys.exit(0)


# ----------------------------------------------------------------------------
# PIPELINE UTAMA
# ----------------------------------------------------------------------------
def analisis(G, df_titik, pare_latlon, faktor, folder):
    pare_node, snap_pare = simpul_terdekat(G, *pare_latlon)
    print(f"Pare dipetakan ke simpul {pare_node} (jarak snap {snap_pare * 1000:.0f} m)")
    if snap_pare > 0.5:
        print("  Peringatan: titik Pare jauh dari jalan; periksa koordinatnya.")

    hitung_waktu_ruas(G, faktor)
    t_grade = waktu_ke_tujuan(G, pare_node, "travel_time")
    t_flat = waktu_ke_tujuan(G, pare_node, "travel_time_flat")

    rows = []
    for _, r in df_titik.iterrows():
        nama = r["Kecamatan"]
        if pd.isna(r["lat"]):
            rows.append({"Kecamatan": nama})
            continue
        n, snap = simpul_terdekat(G, r["lat"], r["lon"])
        if n not in t_grade:
            rows.append({"Kecamatan": nama, "catatan": "tidak terhubung ke Pare"})
            continue
        rows.append({
            "Kecamatan": nama,
            "waktu_tempuh_pare_menit": t_grade[n] / 60,
            "waktu_tempuh_datar_menit": t_flat[n] / 60,
            "jarak_rute_km": panjang_rute(G, n, pare_node, "travel_time"),
            "snap_km": snap,
            "lat": r["lat"], "lon": r["lon"],
            "catatan": "snap>%.0fkm" % BATAS_SNAP_KM if snap > BATAS_SNAP_KM else "",
        })
    out = pd.DataFrame(rows)
    out["rasio_tanjakan"] = out["waktu_tempuh_pare_menit"] / out["waktu_tempuh_datar_menit"]
    return out, pare_node


def grafik(out, pare_latlon, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = out.dropna(subset=["waktu_tempuh_pare_menit"])
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(d["lon"], d["lat"], c=d["waktu_tempuh_pare_menit"], cmap="YlOrRd", s=90,
                    edgecolor="k")
    for _, r in d.iterrows():
        ax.annotate(r["Kecamatan"], (r["lon"], r["lat"]), fontsize=7, xytext=(4, 3),
                    textcoords="offset points")
    ax.scatter([pare_latlon[1]], [pare_latlon[0]], marker="*", s=250, color="blue", label="Pare")
    plt.colorbar(sc, label="Waktu tempuh ke Pare (menit)")
    ax.set_title("Waktu tempuh dari titik kecamatan ke Pasar Induk Pare")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(folder) / "04_waktu_tempuh_pare.png", dpi=150)
    plt.close(fig)


def kalibrasi(out, path=None):
    path = path or (BASE / "kalibrasi_waktu.csv")
    if not Path(path).exists():
        print(f"\n(Opsional) Buat {path} berisi kolom Kecamatan,waktu_nyata_menit dari Google Maps/survei "
              "untuk mengkalibrasi kecepatan.")
        return
    k = pd.read_csv(path).merge(out[["Kecamatan", "waktu_tempuh_pare_menit"]], on="Kecamatan")
    k["rasio_model_per_nyata"] = k["waktu_tempuh_pare_menit"] / k["waktu_nyata_menit"]
    print("\nKALIBRASI (model / nyata):")
    print(k.round(2).to_string(index=False))
    med = k["rasio_model_per_nyata"].median()
    print(f"Median rasio = {med:.2f}. Jalankan ulang dengan --faktor {med:.2f} "
          "untuk menyelaraskan skala waktu.")


# ----------------------------------------------------------------------------
# SELF-TEST (graf sintetis, tanpa internet)
# ----------------------------------------------------------------------------
def graf_sintetis():
    G = nx.MultiDiGraph()
    n = 8
    lat0, lon0, step = -7.80, 112.10, 0.01
    idx = lambda i, j: i * n + j
    for i in range(n):
        for j in range(n):
            G.add_node(idx(i, j), y=lat0 + i * step, x=lon0 + j * step)
    for i in range(n):
        for j in range(n):
            for di, dj in [(0, 1), (1, 0)]:
                a, b = (i, j), (i + di, j + dj)
                if b[0] < n and b[1] < n:
                    L = haversine_km(lat0 + a[0] * step, lon0 + a[1] * step,
                                     lat0 + b[0] * step, lon0 + b[1] * step) * 1000
                    for u, v in [(idx(*a), idx(*b)), (idx(*b), idx(*a))]:
                        G.add_edge(u, v, length=L, highway="tertiary")
    # elevasi naik ke utara: 0 m (i=0) sampai 350 m (i=7)
    elev = {idx(i, j): 50.0 * i for i in range(n) for j in range(n)}
    return G, elev


def selftest():
    print("SELF-TEST dengan graf sintetis 8x8 (lereng naik ke utara)")
    G, elev = graf_sintetis()
    pasang_elevasi_dan_grade(G, elev)
    pare = (-7.80 + 7 * 0.01, 112.10 + 3 * 0.01)       # di puncak lereng (utara)
    titik = pd.DataFrame({
        "Kecamatan": ["Selatan_bawah", "Dekat_pare", "Barat_datar"],
        "lat": [-7.80, -7.80 + 6 * 0.01, -7.80 + 7 * 0.01],
        "lon": [112.13, 112.13, 112.10],
    })
    out, _ = analisis(G, titik, pare, 1.0, ".")
    print(out[["Kecamatan", "waktu_tempuh_pare_menit", "waktu_tempuh_datar_menit",
               "jarak_rute_km", "rasio_tanjakan"]].round(2).to_string(index=False))
    a = out.set_index("Kecamatan")
    assert a.loc["Selatan_bawah", "rasio_tanjakan"] > 1.2, "tanjakan harus menambah waktu"
    assert abs(a.loc["Barat_datar", "rasio_tanjakan"] - 1) < 0.05, "jalur datar tidak boleh kena penalti"
    assert a.loc["Dekat_pare", "waktu_tempuh_pare_menit"] < a.loc["Selatan_bawah", "waktu_tempuh_pare_menit"]
    print("SELF-TEST LULUS")


# ----------------------------------------------------------------------------
def muat_atau_bangun_graf(args, folder):
    import osmnx as ox
    gpath = folder / "kediri_graph.graphml"
    if gpath.exists():
        print(f"Memuat graf cache: {gpath}")
        G = ox.load_graphml(gpath)
        for u, v, k, d in G.edges(keys=True, data=True):      # pastikan tipe numerik
            d["length"] = float(d.get("length", 0) or 0)
            d["grade"] = float(d.get("grade", 0) or 0)
        return G
    print("Mengunduh jaringan jalan OSM (bisa beberapa menit)...")
    gdf = ox.geocode_to_gdf(WILAYAH)
    poly = gdf.to_crs(32749).geometry.buffer(BUFFER_M).to_crs(4326).iloc[0]
    G = ox.graph_from_polygon(poly, network_type="drive")
    print(f"  {len(G.nodes)} simpul, {len(G.edges)} ruas")
    if args.dem:
        print("Mengambil elevasi dari raster DEM...")
        elev = elevasi_raster(G, args.dem)
    else:
        print("Mengambil elevasi dari OpenTopoData (SRTM 30 m)...")
        elev = elevasi_opentopodata(G, folder / "elevasi_cache.json")
    kosong = sum(1 for v in elev.values() if np.isnan(v))
    print(f"  Simpul tanpa elevasi: {kosong}")
    pasang_elevasi_dan_grade(G, elev)
    ox.save_graphml(G, gpath)
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dem", nargs="+", help="satu/lebih raster DEM (Copernicus GLO-30)")
    ap.add_argument("--pare", default=None, help="lat,lon Pasar Induk Pare")
    ap.add_argument("--faktor", type=float, default=1.0)
    ap.add_argument("--batas", default=None, help="file poligon batas kecamatan (geojson/shp/gpkg)")
    ap.add_argument("--kolom-nama", default=None, help="kolom nama kecamatan di file batas (otomatis dideteksi)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    folder = Path(args.out) if args.out else BASE / "hasil_modul3"
    folder.mkdir(parents=True, exist_ok=True)
    pare = PARE_DEFAULT if not args.pare else tuple(float(x) for x in args.pare.split(","))
    print(f"Titik Pasar Induk Pare: {pare[0]}, {pare[1]}")

    csv = args.csv or cari_csv()
    print(f"Data cabai: {csv}")
    nama = [k for k in pd.read_csv(csv, dtype=str)["Kecamatan"] if k != "Kabupaten Kediri"]
    G = muat_atau_bangun_graf(args, folder)
    file_titik = BASE / "titik_kecamatan.csv"
    batas = args.batas or (cari_batas() if not file_titik.exists() else None)
    if batas:
        print(f"File batas kecamatan: {batas}")
    titik = titik_kecamatan(nama, file_titik, G, batas, args.kolom_nama)

    out, _ = analisis(G, titik, pare, args.faktor, folder)

    print("\nWAKTU TEMPUH KE PARE (menit), diurutkan terjauh:")
    tampil = out.sort_values("waktu_tempuh_pare_menit", ascending=False)
    print(tampil[["Kecamatan", "waktu_tempuh_pare_menit", "waktu_tempuh_datar_menit",
                  "jarak_rute_km", "rasio_tanjakan", "snap_km", "catatan"]].round(2).to_string(index=False))

    out_cols = ["Kecamatan", "waktu_tempuh_pare_menit", "waktu_tempuh_datar_menit",
                "jarak_rute_km", "rasio_tanjakan", "snap_km", "lat", "lon"]
    out[out_cols].round(3).to_csv(BASE / "data_tambahan.csv", index=False)
    out.round(3).to_csv(folder / "waktu_tempuh_detail.csv", index=False)
    grafik(out, pare, folder)
    kalibrasi(out)
    print("\nSelesai. data_tambahan.csv siap dibaca kediri_modul1_analisis.py.")


if __name__ == "__main__":
    main()
