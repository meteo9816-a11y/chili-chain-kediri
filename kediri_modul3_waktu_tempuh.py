#!/usr/bin/env python3
"""
KEDIRI VALUECHAIN - Modul 3 (Network Accessibility & Spatial Logistics)
Versi Cepat & Stabil: Rute Jalan Raya OSRM (berbasis OSM) + Profil Tanjakan DEM GEE

Menghitung waktu tempuh dari titik tiap kecamatan ke Pasar Induk Pare dengan penalti
tanjakan (grade per segmen rute) dari Copernicus DEM GLO-30.
Hasilnya ditulis ke data_tambahan.csv untuk langsung dibaca kediri_modul1_analisis.py.

Perubahan dari versi sebelumnya: pengambilan elevasi dari GEE dipecah per batch, karena
getInfo() GEE dibatasi 5.000 elemen per permintaan dan titik rute seluruh kecamatan bisa melebihinya.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# --- util lokasi file & konsol ----------------------------------------------
BASE = Path(__file__).resolve().parent
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ----------------------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------------------
PARE_DEFAULT = (-7.7472121, 112.1992859)   # Pasar Induk Pare (lat, lon)
PROJECT_GCP = "earth-engine-research-459913"

# Kecepatan rata-rata angkutan logistik cabai (pikap/truk ringan) vs kecepatan mobil standar OSRM
FAKTOR_ANGKUTAN_DEFAULT = 0.85   # kendaraan bermuatan ~15% lebih lambat dari mobil kosong
K_TANJAKAN = 5.0                 # penalti tanjakan: faktor = max(F_MIN, 1 - K * grade)
F_MIN = 0.40
GRADE_CLIP = 0.15
BATCH_GEE = 1500                 # jumlah titik per permintaan GEE (batas getInfo 5.000 elemen)


def cari_csv():
    for nama in ("Rekap_Cabai_Rawit_Kediri_2021_2025-v3.csv", "Rekap_Cabai_Rawit_Kediri_2021_2025.csv"):
        if (BASE / nama).exists():
            return str(BASE / nama)
    sisa = sorted(BASE.glob("Rekap_Cabai_Rawit*.csv"))
    if sisa:
        return str(sisa[-1])
    sys.exit(f"CSV data cabai tidak ditemukan di {BASE}.")


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
    sisa = sorted(BASE.glob("*.shp"))
    return str(sisa[0]) if sisa else None


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def faktor_tanjakan(grade):
    return max(F_MIN, 1.0 - K_TANJAKAN * grade) if grade > 0 else 1.0


# ----------------------------------------------------------------------------
# 1. ROUTING JALAN RAYA VIA OSRM (BERBASIS OPENSTREETMAP)
# ----------------------------------------------------------------------------
def ambil_rute_osrm(lat_asal, lon_asal, lat_tujuan, lon_tujuan):
    """Ambil geometri rute, jarak (m), dan durasi dasar (detik) dari OSRM."""
    servers = [
        "https://router.project-osrm.org/route/v1/driving",
        "https://routing.openstreetmap.de/routed-car/route/v1/driving",
    ]
    coords_str = f"{lon_asal:.6f},{lat_asal:.6f};{lon_tujuan:.6f},{lat_tujuan:.6f}"
    params = {"overview": "full", "geometries": "geojson"}

    for base_url in servers:
        try:
            r = requests.get(f"{base_url}/{coords_str}", params=params, timeout=20)
            if r.status_code == 200:
                data = r.json()
                if data.get("code") == "Ok" and data.get("routes"):
                    route = data["routes"][0]
                    # Waypoint pertama adalah titik jalan terdekat (snap) dari koordinat asal
                    wp_asal = data["waypoints"][0]["location"]  # [lon, lat]
                    snap_km = haversine_m(lat_asal, lon_asal, wp_asal[1], wp_asal[0]) / 1000.0
                    coords = route["geometry"]["coordinates"]   # list of [lon, lat]
                    return {
                        "jarak_m": float(route["distance"]),
                        "durasi_s": float(route["duration"]),
                        "snap_km": float(snap_km),
                        "coords": coords,
                    }
        except Exception:
            time.sleep(0.5)
    return None


# ----------------------------------------------------------------------------
# 2. EKSTRAKSI ELEVASI RUTE DARI GOOGLE EARTH ENGINE (COPERNICUS DEM 30M)
# ----------------------------------------------------------------------------
def ambil_elevasi_rute_gee(rute_dict, project):
    """Ambil elevasi tiap titik simpul rute menggunakan Copernicus DEM di GEE (per batch)."""
    import ee
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)

    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic()

    # Kumpulkan semua titik unik dari semua rute (di-downsample tiap ~150 m agar super cepat)
    titik_unik = []          # daftar (lon, lat); indeks = kid
    idx_map = {}
    for kec, info in rute_dict.items():
        if info is None:
            continue
        coords = info["coords"]
        sampled = [coords[0]]
        akum_m = 0.0
        for i in range(1, len(coords)):
            d = haversine_m(coords[i - 1][1], coords[i - 1][0], coords[i][1], coords[i][0])
            akum_m += d
            if akum_m >= 150.0 or i == len(coords) - 1:
                sampled.append(coords[i])
                akum_m = 0.0
        info["sampled_coords"] = sampled
        for lon, lat in sampled:
            key = (round(lon, 5), round(lat, 5))
            if key not in idx_map:
                idx_map[key] = len(titik_unik)
                titik_unik.append((lon, lat))

    n = len(titik_unik)
    n_batch = (n + BATCH_GEE - 1) // BATCH_GEE
    print(f"Mengekstrak elevasi DEM 30 m untuk {n} titik rute di GEE ({n_batch} batch)...")

    elev_lookup = {}
    for b in range(n_batch):
        awal, akhir = b * BATCH_GEE, min((b + 1) * BATCH_GEE, n)
        feats = [ee.Feature(ee.Geometry.Point([titik_unik[k][0], titik_unik[k][1]]), {"kid": k})
                 for k in range(awal, akhir)]
        hasil = dem.sampleRegions(collection=ee.FeatureCollection(feats), scale=30,
                                  geometries=False).getInfo()
        for f in hasil["features"]:
            props = f["properties"]
            elev_lookup[props["kid"]] = props.get("DEM", np.nan)
        print(f"  batch {b + 1}/{n_batch} selesai")

    for kec, info in rute_dict.items():
        if info is None:
            continue
        elevs = []
        for lon, lat in info["sampled_coords"]:
            key = (round(lon, 5), round(lat, 5))
            elevs.append(elev_lookup.get(idx_map[key], np.nan))
        info["elevs"] = elevs


# ----------------------------------------------------------------------------
# 3. GRAFIK & SPASIAL
# ----------------------------------------------------------------------------
def simpan_visual_dan_spasial(out, rute_dict, pare_latlon, folder, batas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot rute jalan + titik kecamatan
    fig, ax = plt.subplots(figsize=(8, 7))
    for kec, info in rute_dict.items():
        if info and "coords" in info:
            xs = [c[0] for c in info["coords"]]
            ys = [c[1] for c in info["coords"]]
            ax.plot(xs, ys, color="#78909C", lw=1.0, alpha=0.65, zorder=2)

    d = out.dropna(subset=["waktu_tempuh_pare_menit"])
    sc = ax.scatter(d["lon"], d["lat"], c=d["waktu_tempuh_pare_menit"], cmap="YlOrRd",
                    s=95, edgecolor="k", zorder=3)
    for _, r in d.iterrows():
        ax.annotate(r["Kecamatan"], (r["lon"], r["lat"]), fontsize=7, xytext=(4, 3),
                    textcoords="offset points", zorder=4)
    ax.scatter([pare_latlon[1]], [pare_latlon[0]], marker="*", s=260, color="#1565C0",
               edgecolor="white", label="Pasar Induk Pare", zorder=5)
    plt.colorbar(sc, label="Waktu tempuh ke Pare (menit)")
    ax.set_title("Rute & Waktu Tempuh Logistik Cabai Rawit ke Pasar Induk Pare")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(folder) / "04_waktu_tempuh_pare.png", dpi=150)
    plt.close(fig)

    if batas and Path(batas).exists():
        import geopandas as gpd
        gdf = gpd.read_file(batas)
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        gdf = gdf.to_crs(4326)
        gdf["_key"] = gdf["KECAMATAN"].map(_norm)
        out_sp = out.copy()
        out_sp["_key"] = out_sp["Kecamatan"].map(_norm)
        merged = gdf.merge(out_sp.drop(columns=["lat", "lon"]), on="_key", how="left").drop(columns=["_key"])

        merged.to_file(Path(folder) / "waktu_tempuh_spasial.geojson", driver="GeoJSON")
        merged.to_file(Path(folder) / "waktu_tempuh_spasial.shp")

        fig, ax = plt.subplots(figsize=(8, 7))
        merged.plot(
            column="waktu_tempuh_pare_menit",
            cmap="YlOrRd",
            linewidth=0.7,
            edgecolor="black",
            legend=True,
            legend_kwds={"label": "Waktu tempuh ke Pasar Induk Pare (menit)", "shrink": 0.75},
            ax=ax,
        )
        for kec, info in rute_dict.items():
            if info and "coords" in info:
                xs = [c[0] for c in info["coords"]]
                ys = [c[1] for c in info["coords"]]
                ax.plot(xs, ys, color="#263238", lw=0.8, alpha=0.45, zorder=3)
        ax.scatter([pare_latlon[1]], [pare_latlon[0]], marker="*", s=230, color="#1565C0",
                   edgecolor="white", linewidth=0.8, label="Pasar Induk Pare", zorder=5)
        pts = merged.to_crs(32749).geometry.representative_point().to_crs(4326)
        for idx, row in merged.iterrows():
            ax.annotate(row["KECAMATAN"].title(), xy=(pts.loc[idx].x, pts.loc[idx].y),
                        ha="center", va="center", fontsize=6.5, color="#111111")
        ax.set_title("Aksesibilitas Logistik Cabai Rawit ke Pasar Induk Pare")
        ax.legend(loc="lower left")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(Path(folder) / "05_peta_waktu_tempuh_pare.png", dpi=180)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pare", default=None, help="lat,lon Pasar Induk Pare")
    ap.add_argument("--faktor", type=float, default=FAKTOR_ANGKUTAN_DEFAULT)
    ap.add_argument("--project", default=PROJECT_GCP)
    ap.add_argument("--batas", default=None)
    args = ap.parse_args()

    folder = Path(args.out) if args.out else BASE / "hasil_modul3"
    folder.mkdir(parents=True, exist_ok=True)
    pare = PARE_DEFAULT if not args.pare else tuple(float(x) for x in args.pare.split(","))
    batas = args.batas or cari_batas()

    file_titik = BASE / "titik_kecamatan.csv"
    if not file_titik.exists():
        sys.exit("File titik_kecamatan.csv tidak ditemukan. Jalankan kediri_modul2_gee.py lebih dulu.")

    df_titik = pd.read_csv(file_titik)
    print(f"Menghitung rute jaringan jalan OSM untuk {len(df_titik)} kecamatan ke Pasar Induk Pare...")

    rute_dict = {}
    for _, r in df_titik.iterrows():
        kec = r["Kecamatan"]
        info = ambil_rute_osrm(r["lat"], r["lon"], pare[0], pare[1])
        rute_dict[kec] = info
        status = f"{info['jarak_m']/1000:.1f} km" if info else "GAGAL"
        print(f"  {kec:15s}: {status}")
        time.sleep(0.15)

    # Ekstrak elevasi DEM sepanjang rute via GEE
    ambil_elevasi_rute_gee(rute_dict, args.project)

    # Hitung waktu tempuh dengan penalti tanjakan per segmen rute
    rows = []
    for _, r in df_titik.iterrows():
        kec = r["Kecamatan"]
        info = rute_dict.get(kec)
        if not info:
            rows.append({"Kecamatan": kec, "catatan": "rute gagal"})
            continue

        durasi_datar_s = info["durasi_s"] / args.faktor
        coords = info["sampled_coords"]
        elevs = info["elevs"]

        # Hitung panjang tiap segmen dan proporsi penalti tanjakan
        seg_lengths = []
        seg_penalties = []
        for i in range(1, len(coords)):
            L = haversine_m(coords[i - 1][1], coords[i - 1][0], coords[i][1], coords[i][0])
            if L <= 0:
                continue
            z1, z2 = elevs[i - 1], elevs[i]
            if not (np.isnan(z1) or np.isnan(z2)):
                grade = float(np.clip((z2 - z1) / L, -GRADE_CLIP, GRADE_CLIP))
            else:
                grade = 0.0
            seg_lengths.append(L)
            seg_penalties.append(1.0 / faktor_tanjakan(grade))

        if seg_lengths:
            faktor_rute_tanjakan = float(np.average(seg_penalties, weights=seg_lengths))
        else:
            faktor_rute_tanjakan = 1.0

        durasi_tanjakan_s = durasi_datar_s * faktor_rute_tanjakan
        rows.append({
            "Kecamatan": kec,
            "waktu_tempuh_pare_menit": durasi_tanjakan_s / 60.0,
            "waktu_tempuh_datar_menit": durasi_datar_s / 60.0,
            "jarak_rute_km": info["jarak_m"] / 1000.0,
            "rasio_tanjakan": faktor_rute_tanjakan,
            "snap_km": info["snap_km"],
            "lat": r["lat"],
            "lon": r["lon"],
            "catatan": "",
        })

    out = pd.DataFrame(rows)
    print("\nWAKTU TEMPUH KE PARE (menit), diurutkan terjauh:")
    tampil = out.sort_values("waktu_tempuh_pare_menit", ascending=False)
    print(tampil[["Kecamatan", "waktu_tempuh_pare_menit", "waktu_tempuh_datar_menit",
                  "jarak_rute_km", "rasio_tanjakan", "snap_km"]].round(2).to_string(index=False))

    out_cols = ["Kecamatan", "waktu_tempuh_pare_menit", "waktu_tempuh_datar_menit",
                "jarak_rute_km", "rasio_tanjakan", "snap_km", "lat", "lon"]
    out[out_cols].round(3).to_csv(BASE / "data_tambahan.csv", index=False)
    out.round(3).to_csv(folder / "waktu_tempuh_detail.csv", index=False)
    simpan_visual_dan_spasial(out, rute_dict, pare, folder, batas)
    print("\nSelesai dalam hitungan detik! data_tambahan.csv siap dibaca kediri_modul1_analisis.py.")


if __name__ == "__main__":
    main()
