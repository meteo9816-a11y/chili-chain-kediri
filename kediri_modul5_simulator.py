#!/usr/bin/env python3
"""
CHILI-CHAIN (KEDIRI VALUECHAIN) - Sistem Pendukung Keputusan & Layanan Rantai Pasok Cabai Rawit
Mobile-First & Responsive Edition dengan Welcome Portal Profesional
Untuk Dinas Pertanian, TPID, Petani, dan Pelaku Usaha Kabupaten Kediri

Jalankan:
    python -m streamlit run kediri_modul5_simulator.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FOLDER_M1 = BASE / "hasil_modul1"
FOLDER_M3 = BASE / "hasil_modul3"
FOLDER_M5 = BASE / "hasil_modul5"

FILE_MATRIKS = FOLDER_M5 / "matriks_waktu.csv"
FILE_GRAF = FOLDER_M3 / "kediri_graph.graphml"
FILE_TITIK = BASE / "titik_kecamatan.csv"
FILE_TAMBAHAN = BASE / "data_tambahan.csv"
FILE_MODUL2 = BASE / "data_modul2.csv"
FILE_INDIKATOR = FOLDER_M1 / "indikator_modul1.csv"
FILE_SHP = BASE / "KAB_KEDIRI_26_KEC.shp"
PARE = "PARE"

SKENARIO_SUSUT = {
    "Rendah (5% — Distribusi lokal cepat)": 0.05,
    "Sedang (15% — Rantai pasok regional)": 0.15,
    "Tinggi (30% — Angkutan jauh / tanpa pendingin)": 0.30,
}


# ----------------------------------------------------------------------------
# 1. MATRIKS WAKTU TEMPUH
# ----------------------------------------------------------------------------
def bangun_matriks(G, titik, pare_latlon, faktor=1.0):
    import networkx as nx
    from kediri_modul3_waktu_tempuh import hitung_waktu_ruas, simpul_terdekat

    for u, v, k, d in G.edges(keys=True, data=True):
        d["length"] = float(d.get("length", 0) or 0)
        d["grade"] = float(d.get("grade", 0) or 0)
    hitung_waktu_ruas(G, faktor)
    Gr = G.reverse(copy=True)

    nodes = {r["Kecamatan"]: simpul_terdekat(G, r["lat"], r["lon"])[0] for _, r in titik.iterrows()}
    nodes[PARE] = simpul_terdekat(G, *pare_latlon)[0]

    cache, kolom = {}, {}
    for nama, n in nodes.items():
        if n not in cache:
            cache[n] = nx.single_source_dijkstra_path_length(Gr, n, weight="travel_time")
        kolom[nama] = cache[n]
    asal = [k for k in nodes if k != PARE]
    T = pd.DataFrame(index=asal, columns=list(nodes), dtype=float)
    for nama_j in nodes:
        dist = kolom[nama_j]
        T[nama_j] = [dist.get(nodes[i], np.inf) / 60 for i in asal]
    return T


def bangun_matriks_osrm(titik, pare_latlon, faktor, rasio):
    import requests
    nama = list(titik["Kecamatan"])
    pts = [(r.lat, r.lon) for r in titik.itertuples()] + [tuple(pare_latlon)]
    koord = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in pts)
    servers = [
        "https://router.project-osrm.org/table/v1/driving",
        "https://routing.openstreetmap.de/routed-car/table/v1/driving",
    ]
    D = None
    for url in servers:
        try:
            r = requests.get(f"{url}/{koord}", params={"annotations": "duration"}, timeout=90)
            if r.status_code == 200 and r.json().get("code") == "Ok":
                D = np.array([[np.nan if v is None else v for v in baris]
                              for baris in r.json()["durations"]], float)
                break
        except Exception as e:
            print(f"  OSRM {url} gagal: {e}")
    if D is None:
        raise RuntimeError("Layanan rute OSRM tidak dapat dihubungi. Periksa koneksi internet.")
    rasio_v = np.array([float(rasio.get(n, 1.0)) if pd.notna(rasio.get(n, 1.0)) else 1.0 for n in nama])
    n = len(nama)
    T = pd.DataFrame(D[:n, :] / 60.0 / faktor * rasio_v[:, None], index=nama, columns=nama + [PARE])
    return T.replace(np.nan, np.inf)


def muat_atau_bangun_matriks(faktor=1.0, paksa=False):
    FOLDER_M5.mkdir(parents=True, exist_ok=True)
    punya_graf = FILE_GRAF.exists()
    punya_tamb = FILE_TAMBAHAN.exists()
    pakai_osrm = punya_tamb and (not punya_graf or FILE_TAMBAHAN.stat().st_mtime > FILE_GRAF.stat().st_mtime)

    sumber = [f for f in ([FILE_TAMBAHAN] if pakai_osrm else [FILE_GRAF]) + [FILE_TITIK] if f.exists()]
    perlu = paksa or not FILE_MATRIKS.exists()
    if not perlu and sumber:
        acuan = FILE_MATRIKS.stat().st_mtime
        perlu = any(f.stat().st_mtime > acuan for f in sumber)
    if perlu:
        if not FILE_TITIK.exists() or not (punya_graf or punya_tamb):
            raise FileNotFoundError(
                f"Butuh {FILE_TITIK} dan {FILE_TAMBAHAN} (Modul 3) untuk membangun matriks."
            )
        titik = pd.read_csv(FILE_TITIK)
        from kediri_modul3_waktu_tempuh import PARE_DEFAULT
        if pakai_osrm:
            tamb = pd.read_csv(FILE_TAMBAHAN).set_index("Kecamatan")
            try:
                from kediri_modul3_waktu_tempuh import FAKTOR_ANGKUTAN_DEFAULT as f_angkutan
            except Exception:
                f_angkutan = 0.85
            T = bangun_matriks_osrm(titik, PARE_DEFAULT, f_angkutan, tamb["rasio_tanjakan"].to_dict())
        else:
            import osmnx as ox
            T = bangun_matriks(ox.load_graphml(FILE_GRAF), titik, PARE_DEFAULT, faktor)
        T.to_csv(FILE_MATRIKS)
    return pd.read_csv(FILE_MATRIKS, index_col=0)


# ----------------------------------------------------------------------------
# 2. MODEL SIMULASI KEBIJAKAN
# ----------------------------------------------------------------------------
def simulasi(T, prod_ton, fasilitas, radius, susut=0.15, efek=0.50, kapasitas=5000.0, porsi_pare=1.0):
    asal = T.index
    w_total = prod_ton.reindex(asal).fillna(0.0).to_numpy(float)
    w = w_total * porsi_pare
    total = w.sum()
    t0 = T[PARE].to_numpy(float)
    fasilitas = [f for f in fasilitas if f in T.columns]
    if fasilitas:
        Ts = T[fasilitas].to_numpy(float)
        tmin = Ts.min(axis=1)
        arg = Ts.argmin(axis=1)
        t1 = np.minimum(t0, tmin)
        terdekat = np.where(tmin < t0, np.array(fasilitas)[arg], PARE)
    else:
        tmin = np.full(len(asal), np.inf)
        arg = np.zeros(len(asal), int)
        t1 = t0.copy()
        terdekat = np.array([PARE] * len(asal))

    def rata(t):
        ok = np.isfinite(t)
        return float((t[ok] * w[ok]).sum() / w[ok].sum()) if w[ok].sum() > 0 else np.nan

    cov0 = float(w[t0 <= radius].sum() / total) if total else np.nan
    cov1 = float(w[t1 <= radius].sum() / total) if total else np.nan

    ditangani, per_fasilitas = 0.0, {}
    for j, f in enumerate(fasilitas):
        masuk = (arg == j) & (tmin <= radius)
        vol = float(w[masuk].sum())
        per_fasilitas[f] = {
            "kecamatan_dilayani": int(masuk.sum()),
            "volume_masuk_ton": vol,
            "volume_ditangani_ton": min(vol, kapasitas),
        }
        ditangani += min(vol, kapasitas)

    hasil = {
        "total_produksi_kab_ton": float(w_total.sum()),
        "total_produksi_ton": float(total),
        "t_rata_baseline": rata(t0),
        "t_rata_skenario": rata(t1),
        "cakupan_baseline": cov0,
        "cakupan_skenario": cov1,
        "ton_baru_terjangkau": float(w[(t1 <= radius) & ~(t0 <= radius)].sum()),
        "ditangani_ton": ditangani,
        "diselamatkan_ton": ditangani * susut * efek,
        "diselamatkan_rendah": ditangani * 0.05 * efek,
        "diselamatkan_sedang": ditangani * 0.15 * efek,
        "diselamatkan_tinggi": ditangani * 0.30 * efek,
        "per_fasilitas": per_fasilitas,
    }
    hasil["turun_menit"] = hasil["t_rata_baseline"] - hasil["t_rata_skenario"]
    hasil["turun_persen"] = (
        hasil["turun_menit"] / hasil["t_rata_baseline"] * 100 if hasil["t_rata_baseline"] else np.nan
    )
    detail = pd.DataFrame({
        "Kecamatan": asal,
        "produksi_total_ton": w_total,
        "produksi_ton": w,
        "t_ke_pare_menit": t0,
        "t_skenario_menit": t1,
        "turun_menit": t0 - t1,
        "dilayani": terdekat,
    })
    return hasil, detail


def skor_tujuan(hasil, tujuan):
    if tujuan == "STA":
        return hasil["turun_menit"] if np.isfinite(hasil["turun_menit"]) else -1e9
    return hasil["diselamatkan_ton"] + 1e-6 * hasil["turun_menit"]


def penempatan_optimal(T, prod_ton, kandidat, k, tujuan, radius, susut, efek, kapasitas, porsi_pare):
    pilih = []
    for _ in range(k):
        terbaik, skor_terbaik = None, -np.inf
        for c in kandidat:
            if c in pilih or c not in T.columns:
                continue
            h, _ = simulasi(T, prod_ton, pilih + [c], radius, susut, efek, kapasitas, porsi_pare)
            s = skor_tujuan(h, tujuan)
            if s > skor_terbaik:
                terbaik, skor_terbaik = c, s
        if terbaik is None:
            break
        pilih.append(terbaik)
    return pilih


# ----------------------------------------------------------------------------
# 3. DATA PENDUKUNG & POLIGON SHP
# ----------------------------------------------------------------------------
def _norm(nama):
    n = str(nama).upper().replace("KECAMATAN", "").replace("KEC.", "")
    return "".join(ch for ch in n if ch.isalnum())


def muat_pendukung():
    ind = pd.read_csv(FILE_INDIKATOR)
    prod_ton = ind.set_index("Kecamatan")["prod_mean_kw"].fillna(0.0) / 10.0
    lolos = ind.loc[ind.get("lolos_ambang", True) == True, "Kecamatan"].tolist()  # noqa: E712
    rank = {}
    for nama, f in [("STA", "ranking_sta.csv"), ("Cold Storage", "ranking_cold_storage.csv")]:
        p = FOLDER_M1 / f
        rank[nama] = pd.read_csv(p).sort_values("rank")["Kecamatan"].tolist() if p.exists() else []
    titik = pd.read_csv(FILE_TITIK).set_index("Kecamatan")

    gdf = None
    if FILE_SHP.exists():
        import geopandas as gpd
        gdf = gpd.read_file(FILE_SHP)
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        gdf = gdf.to_crs(4326)
        gdf["_key"] = gdf["KECAMATAN"].map(_norm)
        peta_nama = {_norm(k): k for k in ind["Kecamatan"]}
        gdf["Kecamatan"] = gdf["_key"].map(peta_nama).fillna(gdf["KECAMATAN"])

        gdf_utm = gdf.to_crs(32749)
        cent_utm = gdf_utm.geometry.centroid
        med_x, med_y = cent_utm.x.median(), cent_utm.y.median()
        jarak_m = np.hypot(cent_utm.x - med_x, cent_utm.y - med_y)
        gdf = gdf[jarak_m <= 35_000].copy()
        gdf["geometry"] = gdf.geometry.simplify(0.0003)

    return ind, prod_ton, lolos, rank, titik, gdf


# ----------------------------------------------------------------------------
# 4. DESAIN CSS & WELCOME PORTAL PROFESIONAL
# ----------------------------------------------------------------------------
def inject_custom_css(st):
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #0F172A;
            -webkit-tap-highlight-color: transparent;
        }
        .stApp { background-color: #F1F5F9; }
        .block-container {
            padding-top: 0.75rem !important;
            padding-bottom: 2.5rem !important;
            padding-left: 0.9rem !important;
            padding-right: 0.9rem !important;
            max-width: 1180px;
        }
        #MainMenu, footer {visibility: hidden;}
        header[data-testid="stHeader"] {
            background: rgba(241, 245, 249, 0.92);
            backdrop-filter: blur(8px);
            height: 2.6rem;
        }

        /* Welcome Screen Hero */
        .welcome-shell {
            background: linear-gradient(135deg, #052E16 0%, #065F46 52%, #991B1B 115%);
            border-radius: 22px;
            padding: 2.1rem 1.8rem;
            color: #FFFFFF;
            box-shadow: 0 20px 35px -10px rgba(6, 78, 59, 0.35);
            margin-bottom: 1.15rem;
            position: relative;
            overflow: hidden;
        }
        .welcome-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(255, 255, 255, 0.14);
            border: 1px solid rgba(255, 255, 255, 0.25);
            padding: 0.3rem 0.75rem;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.85rem;
        }
        .welcome-title {
            font-size: 1.85rem;
            font-weight: 800;
            letter-spacing: -0.025em;
            line-height: 1.2;
            margin: 0 0 0.65rem 0;
            color: #FFFFFF;
        }
        .welcome-desc {
            font-size: 0.94rem;
            color: #E2E8F0;
            line-height: 1.55;
            max-width: 760px;
            margin: 0 0 1.25rem 0;
        }
        .welcome-stats-bar {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.65rem;
        }
        .w-stat-item {
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 12px;
            padding: 0.65rem 0.85rem;
        }
        .w-stat-num {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.15rem;
            font-weight: 700;
            color: #FFFFFF;
        }
        .w-stat-lbl {
            font-size: 0.72rem;
            color: #CBD5E1;
            margin-top: 0.1rem;
        }

        /* Role Selection Cards on Welcome Screen */
        .role-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 1.1rem 1.15rem;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
            height: 100%;
            margin-bottom: 0.45rem;
        }
        .role-icon {
            width: 40px; height: 40px; border-radius: 10px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.25rem; margin-bottom: 0.65rem;
        }
        .role-title { font-size: 0.98rem; font-weight: 800; color: #0F172A; margin-bottom: 0.25rem; }
        .role-desc { font-size: 0.80rem; color: #475569; line-height: 1.45; margin-bottom: 0.5rem; }

        /* Mobile App Header Bar */
        .mobile-appbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 0.65rem 0.9rem;
            margin-bottom: 0.7rem;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        }
        .brand-left { display: flex; align-items: center; gap: 0.65rem; }
        .chili-logo-box {
            width: 38px; height: 38px; border-radius: 10px; flex-shrink: 0;
            background: linear-gradient(135deg, #DC2626 0%, #059669 100%);
            color: #FFFFFF; display: flex; align-items: center; justify-content: center;
            font-size: 1.25rem; box-shadow: 0 3px 8px rgba(220, 38, 38, 0.22);
        }
        .live-pill {
            font-size: 0.68rem; font-weight: 700; color: #065F46;
            background: #ECFDF5; border: 1px solid #A7F3D0;
            padding: 0.22rem 0.55rem; border-radius: 999px;
            white-space: nowrap;
        }

        /* Compact Hero Card */
        .hero-mobile {
            background: linear-gradient(135deg, #064E3B 0%, #047857 52%, #B91C1C 115%);
            border-radius: 16px;
            padding: 1.05rem 1.15rem;
            color: #FFFFFF;
            margin-bottom: 0.8rem;
            box-shadow: 0 8px 18px -4px rgba(6, 78, 59, 0.28);
        }
        .hero-badge {
            display: inline-block;
            background: rgba(255, 255, 255, 0.16);
            font-size: 0.68rem; font-weight: 700; letter-spacing: 0.04em;
            padding: 0.18rem 0.55rem; border-radius: 6px; margin-bottom: 0.4rem;
            text-transform: uppercase;
        }
        .hero-title {
            font-size: 1.18rem; font-weight: 800; letter-spacing: -0.02em;
            margin: 0 0 0.25rem 0; color: #FFFFFF; line-height: 1.3;
        }
        .hero-sub {
            font-size: 0.80rem; color: #F1F5F9; margin: 0; line-height: 1.45;
        }

        /* 2x2 Mobile KPI Grid */
        .kpi-grid-2x2 {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.6rem;
            margin-bottom: 0.8rem;
        }
        .kpi-card {
            background: #FFFFFF; border-radius: 14px; padding: 0.8rem 0.85rem;
            border: 1px solid #E2E8F0; box-shadow: 0 2px 5px rgba(15, 23, 42, 0.03);
        }
        .kpi-label {
            font-size: 0.67rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.03em; color: #64748B; margin-bottom: 0.25rem;
            line-height: 1.25;
        }
        .kpi-value {
            font-family: 'JetBrains Mono', monospace; font-size: 1.28rem; font-weight: 700;
            color: #0F172A; line-height: 1.15; margin-bottom: 0.3rem;
        }
        .kpi-delta-pos {
            display: inline-block; background: #DCFCE7; color: #166534;
            font-size: 0.68rem; font-weight: 700; padding: 0.16rem 0.45rem; border-radius: 6px;
            line-height: 1.2;
        }
        .kpi-delta-neutral {
            display: inline-block; background: #F1F5F9; color: #475569;
            font-size: 0.68rem; font-weight: 600; padding: 0.16rem 0.45rem; border-radius: 6px;
            line-height: 1.2;
        }

        /* Policy Card */
        .policy-brief {
            background: #FFFFFF; border-left: 4px solid #059669;
            border-top: 1px solid #E2E8F0; border-right: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0;
            border-radius: 12px; padding: 0.8rem 0.95rem; margin: 0.65rem 0 0.85rem 0;
        }
        .policy-brief-title {
            font-size: 0.72rem; font-weight: 800; text-transform: uppercase;
            letter-spacing: 0.04em; color: #065F46; margin-bottom: 0.2rem;
        }
        .policy-brief-text { font-size: 0.82rem; color: #1E293B; line-height: 1.48; margin: 0; }

        /* Farmer Mobile Card */
        .farmer-card {
            background: linear-gradient(145deg, #FFFFFF 0%, #ECFDF5 100%);
            border: 1.5px solid #6EE7B7; border-radius: 16px; padding: 1rem;
            box-shadow: 0 4px 12px rgba(5, 150, 105, 0.08);
            margin-top: 0.4rem;
        }
        .farmer-metrics-3 {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.45rem;
            margin-top: 0.65rem;
        }
        .farmer-metric-box {
            background: #FFFFFF; padding: 0.55rem 0.5rem; border-radius: 10px;
            border: 1px solid #E2E8F0; text-align: center;
        }
        .panel-header { font-size: 0.95rem; font-weight: 800; color: #0F172A; margin-bottom: 0.1rem; }
        .panel-sub { font-size: 0.76rem; color: #64748B; margin-bottom: 0.65rem; line-height: 1.35; }
        .loc-pill {
            display: inline-block; background: #FEF2F2; color: #991B1B;
            border: 1px solid #FECACA; font-size: 0.74rem; font-weight: 700;
            padding: 0.22rem 0.55rem; border-radius: 8px; margin: 0.15rem 0.2rem 0.15rem 0;
        }

        .stButton > button {
            min-height: 44px !important;
            border-radius: 11px !important;
            font-weight: 700 !important;
            font-size: 0.85rem !important;
        }
        div[data-testid="stExpander"] {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.03);
            margin-bottom: 0.75rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            overflow-x: auto;
            flex-wrap: nowrap;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: none;
            padding-bottom: 2px;
            border-bottom: 2px solid #E2E8F0;
        }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            white-space: nowrap;
            flex-shrink: 0;
            border-radius: 10px 10px 0 0;
            padding: 0 13px;
            font-weight: 700;
            font-size: 0.80rem;
            color: #64748B;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FFFFFF !important;
            color: #065F46 !important;
            border: 1px solid #E2E8F0 !important;
            border-bottom: 2.5px solid #059669 !important;
        }

        @media (min-width: 768px) {
            .block-container { padding-left: 1.5rem !important; padding-right: 1.5rem !important; }
            .welcome-title { font-size: 2.35rem; }
            .welcome-stats-bar { grid-template-columns: repeat(4, 1fr); }
            .hero-title { font-size: 1.5rem; }
            .kpi-grid-2x2 { grid-template-columns: repeat(4, 1fr); gap: 0.85rem; }
            .kpi-value { font-size: 1.5rem; }
        }
    </style>
    """, unsafe_allow_html=True)


def gaya_plotly_mobile(fig, tinggi=320):
    fig.update_layout(
        font=dict(family="Plus Jakarta Sans, Arial, sans-serif", size=11, color="#0F172A"),
        title_font=dict(size=12.5, color="#064E3B"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=tinggi,
        margin=dict(t=48, b=38, l=42, r=14),
        xaxis=dict(showline=True, linecolor="#334155", linewidth=1, showgrid=False, ticks="outside"),
        yaxis=dict(showline=True, linecolor="#334155", linewidth=1, gridcolor="#E2E8F0", gridwidth=0.6, ticks="outside"),
    )
    return fig


def render_welcome_portal(st):
    st.markdown("""
    <div class="welcome-shell">
        <div class="welcome-pill">🌶️ CHILI-CHAIN • KEDIRI VALUECHAIN v2.0</div>
        <h1 class="welcome-title">Selamat Datang di Sistem Pendukung Keputusan Rantai Pasok Cabai Rawit</h1>
        <p class="welcome-desc">
            Platform analitik spasial dan simulasi infrastruktur pascapanen (<b>Sub Terminal Agribisnis &amp; Cold Storage</b>)
            berbasis penginderaan jauh, elevasi DEM 30 m, dan jaringan jalan nyata untuk pengendalian inflasi pangan Kabupaten Kediri.
        </p>
        <div class="welcome-stats-bar">
            <div class="w-stat-item">
                <div class="w-stat-num">26 Kec.</div>
                <div class="w-stat-lbl">Cakupan Analisis Kabupaten</div>
            </div>
            <div class="w-stat-item">
                <div class="w-stat-num">95,6 Rb t</div>
                <div class="w-stat-lbl">Produksi Rata-Rata / Tahun</div>
            </div>
            <div class="w-stat-item">
                <div class="w-stat-num">-59,5%</div>
                <div class="w-stat-lbl">Pemangkasan Waktu (2 Unit)</div>
            </div>
            <div class="w-stat-item">
                <div class="w-stat-num">3.000 t</div>
                <div class="w-stat-lbl">Potensi Susut Dicegah / Th</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:0.95rem;font-weight:800;color:#0F172A;margin:0.2rem 0 0.65rem 0;'>"
        "Pilih Mode Akses Sesuai Kebutuhan Anda:</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="role-card">
            <div class="role-icon" style="background:#DCFCE7;color:#166534;">🏛️</div>
            <div class="role-title">TPID &amp; Dinas Pertanian</div>
            <div class="role-desc">Simulasi penempatan 1–4 unit STA/Cold Storage, cakupan layanan ≤30 menit, dan peta pengalihan arus pasokan.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Masuk Dashboard Kebijakan →", key="btn_tpid", type="primary", use_container_width=True):
            st.session_state["entered_app"] = True
            st.session_state["user_role"] = "TPID & Dinas Pertanian"
            st.rerun()

    with c2:
        st.markdown("""
        <div class="role-card">
            <div class="role-icon" style="background:#FEE2E2;color:#991B1B;">🚜</div>
            <div class="role-title">Petani &amp; Pengepul Cabai</div>
            <div class="role-desc">Cek titik setor panen terdekat dari kecamatan kebun Anda, estimasi waktu angkut, dan penghematan susut bobot.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Buka Kalkulator Rute Panen →", key="btn_petani", use_container_width=True):
            st.session_state["entered_app"] = True
            st.session_state["user_role"] = "Petani & Pengepul"
            st.rerun()

    with c3:
        st.markdown("""
        <div class="role-card">
            <div class="role-icon" style="background:#E0F2FE;color:#075985;">📊</div>
            <div class="role-title">Peneliti &amp; Evaluasi Data</div>
            <div class="role-desc">Eksplorasi indeks multikriteria SAIPI, sensitivitas 2.000 iterasi Monte Carlo, dan unduh dataset terpadu 26 kecamatan.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Lihat Data & Metodologi →", key="btn_riset", use_container_width=True):
            st.session_state["entered_app"] = True
            st.session_state["user_role"] = "Peneliti & Akademisi"
            st.rerun()

    st.markdown(
        "<div style='text-align:center;font-size:0.74rem;color:#64748B;margin-top:1.1rem;'>"
        "Dikembangkan oleh Tim Peneliti <b>CHILI-CHAIN</b> • LKTI GenBI 15th Anniversary 2026 • KPw Bank Indonesia Kediri"
        "</div>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# 5. DASHBOARD UTAMA
# ----------------------------------------------------------------------------
def main():
    import streamlit as st
    import folium
    import plotly.graph_objects as go
    from streamlit_folium import st_folium
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx() is None:
        print("\nJalankan dengan perintah:\n    python -m streamlit run kediri_modul5_simulator.py\n")
        sys.exit(0)

    st.set_page_config(
        page_title="CHILI-CHAIN | Rantai Pasok Cabai Rawit Kediri",
        page_icon="🌶️",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    inject_custom_css(st)

    if "entered_app" not in st.session_state:
        st.session_state["entered_app"] = False
    if "user_role" not in st.session_state:
        st.session_state["user_role"] = "TPID & Dinas Pertanian"
    if "preset_k" not in st.session_state:
        st.session_state["preset_k"] = 2

    if not st.session_state["entered_app"]:
        render_welcome_portal(st)
        return

    try:
        T_base = muat_atau_bangun_matriks()
        ind, prod_ton, lolos, rank, titik, gdf = muat_pendukung()
    except Exception as e:
        st.error(f"Data belum siap: {e}")
        st.stop()
        return

    # --- TOP BAR DENGAN TOMBOL KEMBALI KE WELCOME ---
    col_brand, col_home = st.columns([4, 1.3])
    with col_brand:
        st.markdown(f"""
        <div class="mobile-appbar" style="margin-bottom:0;">
            <div class="brand-left">
                <div class="chili-logo-box">🌶️</div>
                <div>
                    <div style="font-weight:800;font-size:0.95rem;color:#0F172A;line-height:1.15;">CHILI-CHAIN</div>
                    <div style="font-size:0.70rem;color:#64748B;">Mode: {st.session_state["user_role"]}</div>
                </div>
            </div>
            <div class="live-pill">● 26 Kec • Live</div>
        </div>
        """, unsafe_allow_html=True)
    with col_home:
        if st.button("🏠 Beranda", use_container_width=True):
            st.session_state["entered_app"] = False
            st.rerun()

    st.markdown("<div style='height:0.55rem;'></div>", unsafe_allow_html=True)

    # --- COMPACT HERO BANNER ---
    st.markdown("""
    <div class="hero-mobile">
        <div class="hero-badge">📱 Kediri ValueChain • Kab. Kediri</div>
        <h1 class="hero-title">Optimasi Rantai Pasok & Infrastruktur Cabai Rawit</h1>
        <p class="hero-sub">
            Simulasi lokasi <b>STA & Cold Storage</b> bagi TPID/Dinas Pertanian serta panduan <b>rute setor panen</b> bagi petani.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- INLINE MOBILE CONTROL DRAWER ---
    expand_drawer = st.session_state["user_role"] != "Petani & Pengepul"
    with st.expander("🎛️ Atur Skenario Simulasi (Ketuk untuk Ubah)", expanded=expand_drawer):
        jenis = st.radio(
            "1. Jenis Fasilitas Pascapanen",
            ["STA", "Cold Storage"],
            horizontal=True,
        )

        st.markdown("<div style='font-size:0.80rem;font-weight:700;margin-bottom:0.25rem;'>2. Pilih Cepat Jumlah Unit</div>", unsafe_allow_html=True)
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("1 Unit", use_container_width=True, type="primary" if st.session_state["preset_k"] == 1 else "secondary"):
            st.session_state["preset_k"] = 1
        if b2.button("2 Unit", use_container_width=True, type="primary" if st.session_state["preset_k"] == 2 else "secondary"):
            st.session_state["preset_k"] = 2
        if b3.button("3 Unit", use_container_width=True, type="primary" if st.session_state["preset_k"] == 3 else "secondary"):
            st.session_state["preset_k"] = 3
        if b4.button("4 Unit", use_container_width=True, type="primary" if st.session_state["preset_k"] == 4 else "secondary"):
            st.session_state["preset_k"] = 4

        mode = st.selectbox(
            "3. Metode Penentuan Lokasi",
            ["Rekomendasi Optimal Sistem", "Prioritas Indeks SAIPI", "Pilih Kecamatan Sendiri"],
            index=0,
        )

        kandidat_semua = [c for c in T_base.index if c in prod_ton.index]
        kandidat = [c for c in lolos if c in T_base.index] or kandidat_semua

        kap, susut, efek = 5000.0, 0.15, 0.50
        radius, porsi_pare, faktor = 30, 1.0, 1.0
        peta_dasar = "Peta Terang (CartoDB Positron)"

        if jenis == "Cold Storage":
            label_skenario_susut = st.selectbox(
                "Skenario Susut Pascapanen",
                list(SKENARIO_SUSUT.keys()),
                index=1,
            )
            susut = SKENARIO_SUSUT[label_skenario_susut]

        atur_lanjut = st.toggle("⚙️ Tampilkan Pengaturan Lanjutan (Radius, Kapasitas, Basemap)", value=False)
        if atur_lanjut:
            radius = st.slider("Radius Layanan Ideal (menit)", 10, 90, 30, 5)
            porsi_pare = st.slider("Pasokan Lewat Pasar Induk Pare (%)", 10, 100, 100, 5) / 100.0
            if jenis == "Cold Storage":
                efek = st.slider("Efektivitas Reduksi Susut (%)", 10, 100, 50, 5) / 100.0
                kap = float(st.number_input("Kapasitas per Unit (ton/tahun)", 500, 50000, 5000, 500))
            peta_dasar = st.selectbox(
                "Tampilan Peta Dasar",
                ["Peta Terang (CartoDB Positron)", "OpenStreetMap (Standar Jalan)", "Citra Satelit (Esri)"],
                index=0,
            )
            faktor = st.slider("Faktor Kecepatan Lalu Lintas", 0.6, 1.4, 1.0, 0.05)

        T = T_base / faktor

        if mode == "Pilih Kecamatan Sendiri":
            default_sel = [rank[jenis][0]] if rank.get(jenis) else kandidat[:1]
            pilih = st.multiselect("Pilih Kecamatan Target", kandidat_semua, default=default_sel)
        else:
            k = st.session_state["preset_k"]
            if mode == "Rekomendasi Optimal Sistem":
                pilih = penempatan_optimal(T, prod_ton, kandidat, k, jenis, radius, susut, efek, kap, porsi_pare)
            else:
                urut = [c for c in rank.get(jenis, []) if c in T.columns]
                pilih = urut[:k]

        if pilih:
            pills_html = "".join(f"<span class='loc-pill'>🌶️ {p}</span>" for p in pilih)
            st.markdown(f"<div style='margin-top:0.35rem;font-size:0.78rem;'><b>Titik Terpilih ({len(pilih)} Unit):</b> {pills_html}</div>", unsafe_allow_html=True)
        else:
            st.warning("Silakan pilih minimal 1 kecamatan.")

    hasil, detail = simulasi(T, prod_ton, pilih, radius, susut, efek, kap, porsi_pare)

    # --- 2x2 MOBILE KPI CARDS ---
    delta_t = -hasil["turun_menit"]
    delta_pct = -hasil["turun_persen"]
    cls_t = "kpi-delta-pos" if hasil["turun_menit"] > 0.1 else "kpi-delta-neutral"

    cov_pct = hasil["cakupan_skenario"] * 100
    cov_diff = (hasil["cakupan_skenario"] - hasil["cakupan_baseline"]) * 100
    cls_c = "kpi-delta-pos" if cov_diff > 0.1 else "kpi-delta-neutral"

    baru_ton = hasil["ton_baru_terjangkau"]
    cls_b = "kpi-delta-pos" if baru_ton > 0 else "kpi-delta-neutral"

    if jenis == "Cold Storage":
        selamat = hasil["diselamatkan_ton"]
        cls_4 = "kpi-delta-pos" if selamat > 0 else "kpi-delta-neutral"
        card4_html = (
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Susut Dicegah ({int(susut*100)}%)</div>'
            f'<div class="kpi-value">{selamat:,.0f} <span style="font-size:0.82rem;font-weight:500;">t/th</span></div>'
            f'<span class="{cls_4}">5–30%: {hasil["diselamatkan_rendah"]:,.0f}–{hasil["diselamatkan_tinggi"]:,.0f} t</span>'
            f'</div>'
        )
    else:
        n_terbantu = int((detail["turun_menit"] > 0.5).sum())
        card4_html = (
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Wilayah Terbantu</div>'
            f'<div class="kpi-value">{n_terbantu}<span style="font-size:0.85rem;font-weight:500;">/{len(detail)} kec</span></div>'
            f'<span class="kpi-delta-neutral">{hasil["total_produksi_ton"]:,.0f} ton/th</span>'
            f'</div>'
        )

    kpi_grid_html = (
        f'<div class="kpi-grid-2x2">'
        f'<div class="kpi-card">'
        f'<div class="kpi-label">Rata Waktu Angkut</div>'
        f'<div class="kpi-value">{hasil["t_rata_skenario"]:.1f} <span style="font-size:0.85rem;font-weight:500;">mnt</span></div>'
        f'<span class="{cls_t}">{delta_t:+.1f} mnt ({delta_pct:+.1f}%)</span>'
        f'</div>'
        f'<div class="kpi-card">'
        f'<div class="kpi-label">Cakupan ≤ {radius} Mnt</div>'
        f'<div class="kpi-value">{cov_pct:.1f}<span style="font-size:0.9rem;">%</span></div>'
        f'<span class="{cls_c}">{cov_diff:+.1f}% jangkauan</span>'
        f'</div>'
        f'<div class="kpi-card">'
        f'<div class="kpi-label">Pasokan Baru Ideal</div>'
        f'<div class="kpi-value">{baru_ton:,.0f} <span style="font-size:0.82rem;font-weight:500;">t/th</span></div>'
        f'<span class="{cls_b}">Masuk radius ≤ {radius}m</span>'
        f'</div>'
        f'{card4_html}'
        f'</div>'
    )
    st.markdown(kpi_grid_html, unsafe_allow_html=True)

    # --- RINGKASAN KEBIJAKAN ---
    if pilih:
        lokasi_str = ", ".join(f"<b>Kec. {p}</b>" for p in pilih)
        n_kec_bantu = int((detail["turun_menit"] > 0.5).sum())
        if jenis == "STA":
            ringkasan_aksi = (
                f"Penempatan <b>{len(pilih)} unit STA</b> di {lokasi_str} memangkas waktu angkut rata-rata sebesar "
                f"<b>{hasil['turun_menit']:.1f} menit ({hasil['turun_persen']:.1f}%)</b>, melayani langsung "
                f"<b>{n_kec_bantu} kecamatan</b>, dengan cakupan ideal (≤ {radius} mnt) <b>{hasil['cakupan_skenario']*100:.1f}%</b>."
            )
        else:
            ringkasan_aksi = (
                f"Pembangunan <b>{len(pilih)} unit Cold Storage</b> di {lokasi_str} menyerap "
                f"<b>{hasil['ditangani_ton']:,.0f} ton/th</b>. Potensi susut yang dicegah mencapai "
                f"<b>{hasil['diselamatkan_rendah']:,.0f}–{hasil['diselamatkan_tinggi']:,.0f} ton/th</b> "
                f"(skenario aktif {int(susut*100)}%: <b>{hasil['diselamatkan_ton']:,.0f} ton/th</b>)."
            )
        st.markdown(
            f'<div class="policy-brief">'
            f'<div class="policy-brief-title">📋 Ringkasan Rekomendasi (TPID & Dinas Pertanian)</div>'
            f'<p class="policy-brief-text">{ringkasan_aksi}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # --- MOBILE SWIPEABLE TABS ---
    tab_names = [
        "🗺️ Peta Zonasi",
        "🚜 Rute Petani",
        "📊 Grafik & Skenario",
        "📑 Data 26 Kec",
    ]
    if st.session_state["user_role"] == "Petani & Pengepul":
        tab_names = [
            "🚜 Rute Petani",
            "🗺️ Peta Zonasi",
            "📊 Grafik & Skenario",
            "📑 Data 26 Kec",
        ]
    elif st.session_state["user_role"] == "Peneliti & Akademisi":
        tab_names = [
            "📊 Grafik & Skenario",
            "📑 Data 26 Kec",
            "🗺️ Peta Zonasi",
            "🚜 Rute Petani",
        ]

    tabs = st.tabs(tab_names)
    tab_map = dict(zip(tab_names, tabs))

    # ========================================================================
    # TAB 1: PETA INTERAKTIF
    # ========================================================================
    with tab_map["🗺️ Peta Zonasi"]:
        st.markdown('<div class="panel-header">Peta Zonasi Layanan & Pengalihan Arus</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">Ketuk lingkaran kecamatan pada peta untuk melihat detail waktu tempuh & titik setor terdekat.</div>', unsafe_allow_html=True)

        from kediri_modul3_waktu_tempuh import PARE_DEFAULT
        lat0, lon0 = titik["lat"].mean(), titik["lon"].mean()

        m = folium.Map(location=[lat0, lon0], zoom_start=10, tiles=None)
        tile_configs = [
            ("Peta Terang (CartoDB Positron)", "CartoDB positron", None, None),
            ("OpenStreetMap (Standar Jalan)", "OpenStreetMap", None, None),
            (
                "Citra Satelit (Esri)",
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                "Esri World Imagery",
                "Citra Satelit (Esri)",
            ),
        ]
        tile_configs.sort(key=lambda item: 0 if item[0] == peta_dasar else 1)
        for label_t, tiles_src, attr_t, name_t in tile_configs:
            if attr_t:
                folium.TileLayer(tiles=tiles_src, attr=attr_t, name=name_t).add_to(m)
            else:
                folium.TileLayer(tiles=tiles_src, name=label_t).add_to(m)

        if gdf is not None:
            gdf_plot = gdf.merge(detail, on="Kecamatan", how="left")
            folium.GeoJson(
                gdf_plot,
                name="Batas 26 Kecamatan Kediri",
                style_function=lambda feat: {
                    "fillColor": "#059669" if (feat["properties"].get("turun_menit") or 0) > 0.5 else "#94A3B8",
                    "color": "#1E293B",
                    "weight": 1.1,
                    "fillOpacity": 0.28 if (feat["properties"].get("turun_menit") or 0) > 0.5 else 0.08,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["Kecamatan", "produksi_total_ton", "t_ke_pare_menit", "t_skenario_menit", "dilayani"],
                    aliases=[
                        "Kecamatan:",
                        "Produksi (ton/th):",
                        "Awal ke Pare (mnt):",
                        "Baru (mnt):",
                        "Titik Setor:",
                    ],
                    localize=True,
                ),
            ).add_to(m)

        fg_flow = folium.FeatureGroup(name="Jalur Pengalihan")
        for _, r in detail.iterrows():
            kec = r["Kecamatan"]
            tujuan = r["dilayani"]
            if kec in titik.index and tujuan != PARE and tujuan in titik.index:
                p_asal = [titik.loc[kec, "lat"], titik.loc[kec, "lon"]]
                p_tuj = [titik.loc[tujuan, "lat"], titik.loc[tujuan, "lon"]]
                folium.PolyLine(
                    [p_asal, p_tuj],
                    color="#059669",
                    weight=2.5,
                    opacity=0.85,
                    dash_array="6, 6",
                ).add_to(fg_flow)
        fg_flow.add_to(m)

        fg_nodes = folium.FeatureGroup(name="Sentra Lahan Cabai")
        pmax = max(detail["produksi_total_ton"].max(), 1)
        for _, r in detail.iterrows():
            kec = r["Kecamatan"]
            if kec not in titik.index:
                continue
            t = titik.loc[kec]
            terbantu = r["turun_menit"] > 0.5
            warna = "#059669" if terbantu else "#334155"
            popup_html = (
                f'<div style="font-family:sans-serif;font-size:12px;min-width:185px;line-height:1.4;">'
                f'<div style="font-weight:700;font-size:13px;color:#064E3B;border-bottom:1px solid #E2E8F0;padding-bottom:3px;margin-bottom:4px;">Kecamatan {kec}</div>'
                f'Produksi: <b>{r["produksi_total_ton"]:,.1f} ton/th</b><br>'
                f'Awal (Pare): {r["t_ke_pare_menit"]:.1f} mnt<br>'
                f'Baru: <b>{r["t_skenario_menit"]:.1f} mnt</b> (<span style="color:#059669;font-weight:bold;">-{r["turun_menit"]:.1f} m</span>)<br>'
                f'Tujuan: <b>{r["dilayani"]}</b>'
                f'</div>'
            )
            folium.CircleMarker(
                [t["lat"], t["lon"]],
                radius=5 + 13 * np.sqrt(r["produksi_total_ton"] / pmax),
                color="#FFFFFF",
                weight=1.3,
                fill=True,
                fill_color=warna,
                fill_opacity=0.88,
                popup=folium.Popup(popup_html, max_width=240),
                tooltip=f"{kec} ({r['produksi_total_ton']:,.0f} t/th)",
            ).add_to(fg_nodes)
        fg_nodes.add_to(m)

        folium.Marker(
            PARE_DEFAULT,
            tooltip="Pasar Induk Pare",
            icon=folium.Icon(color="black", icon="shopping-cart", prefix="fa"),
        ).add_to(m)

        for f in pilih:
            if f in titik.index:
                folium.Marker(
                    [titik.loc[f, "lat"], titik.loc[f, "lon"]],
                    tooltip=f"Rencana {jenis}: {f}",
                    icon=folium.Icon(
                        color="red" if jenis == "Cold Storage" else "green",
                        icon="snowflake-o" if jenis == "Cold Storage" else "leaf",
                        prefix="fa",
                    ),
                ).add_to(m)

        folium.LayerControl(collapsed=True).add_to(m)
        if gdf is not None and not gdf.empty:
            minx, miny, maxx, maxy = gdf.total_bounds
            m.fit_bounds([[miny, minx], [maxy, maxx]])
        st_folium(m, height=370, use_container_width=True, returned_objects=[])

        st.markdown('<div class="panel-header" style="margin-top:0.75rem;">Prioritas Manfaat per Kecamatan</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">Diurutkan dari penghematan waktu tempuh terbesar.</div>', unsafe_allow_html=True)
        df_side = detail.sort_values(["turun_menit", "produksi_total_ton"], ascending=[False, False]).copy()
        st.dataframe(
            df_side[["Kecamatan", "produksi_total_ton", "t_ke_pare_menit", "t_skenario_menit", "turun_menit", "dilayani"]],
            column_config={
                "Kecamatan": st.column_config.TextColumn("Kecamatan"),
                "produksi_total_ton": st.column_config.NumberColumn("Prod.", format="%.0f t"),
                "t_ke_pare_menit": st.column_config.NumberColumn("Awal", format="%.1f m"),
                "t_skenario_menit": st.column_config.NumberColumn("Baru", format="%.1f m"),
                "turun_menit": st.column_config.NumberColumn("Hemat", format="-%.1f m"),
                "dilayani": st.column_config.TextColumn("Tujuan"),
            },
            hide_index=True,
            height=340,
            use_container_width=True,
        )

    # ========================================================================
    # TAB 2: KALKULATOR RUTE PETANI & PENGEPUL
    # ========================================================================
    with tab_map["🚜 Rute Petani"]:
        st.markdown('<div class="panel-header">🚜 Kalkulator Rute Setor Panen</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">Cek titik setor terdekat & penghematan susut panen langsung dari HP Anda.</div>', unsafe_allow_html=True)

        asal_kec = st.selectbox(
            "📍 Kecamatan Asal Kebun / Panen:",
            sorted(detail["Kecamatan"].tolist()),
            index=sorted(detail["Kecamatan"].tolist()).index("Kepung") if "Kepung" in detail["Kecamatan"].tolist() else 0,
        )
        tonase_panen = st.number_input(
            "⚖️ Volume Panen Sekali Angkut (kuintal):",
            min_value=1.0, max_value=500.0, value=10.0, step=1.0,
            help="10 kuintal = 1 ton (1 pikap sayur).",
        )

        baris_asal = detail[detail["Kecamatan"] == asal_kec].iloc[0]
        t_pare_asal = baris_asal["t_ke_pare_menit"]
        t_baru_asal = baris_asal["t_skenario_menit"]
        tujuan_rek = baris_asal["dilayani"]
        hemat_asal = baris_asal["turun_menit"]

        susut_hemat_kg = (tonase_panen * 100) * susut * efek if hemat_asal > 0.5 else 0.0
        status_rute = (
            f"Dialihkan ke <b>{jenis} Kec. {tujuan_rek}</b> (Hemat {hemat_asal:.1f} mnt)"
            if tujuan_rek != PARE
            else "Langsung ke <b>Pasar Induk Pare</b> (Sudah terdekat)"
        )
        farmer_html = (
            f'<div class="farmer-card">'
            f'<div style="font-size:0.70rem;font-weight:800;color:#059669;text-transform:uppercase;letter-spacing:0.04em;">Rekomendasi Rute • Kec. {asal_kec}</div>'
            f'<div style="font-size:1.05rem;font-weight:800;color:#0F172A;margin:0.25rem 0 0.5rem 0;line-height:1.3;">{status_rute}</div>'
            f'<div class="farmer-metrics-3">'
            f'<div class="farmer-metric-box"><div style="font-size:0.65rem;color:#64748B;font-weight:700;">KE PARE</div><div style="font-size:1.02rem;font-weight:800;color:#334155;">{t_pare_asal:.1f} m</div></div>'
            f'<div class="farmer-metric-box"><div style="font-size:0.65rem;color:#64748B;font-weight:700;">TITIK BARU</div><div style="font-size:1.02rem;font-weight:800;color:#059669;">{t_baru_asal:.1f} m</div></div>'
            f'<div class="farmer-metric-box"><div style="font-size:0.65rem;color:#64748B;font-weight:700;">SUSUT HEMAT</div><div style="font-size:1.02rem;font-weight:800;color:#B91C1C;">{susut_hemat_kg:.1f} kg</div></div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(farmer_html, unsafe_allow_html=True)

        opsi_tujuan = [PARE] + pilih
        tabel_opsi = []
        for tj in dict.fromkeys(opsi_tujuan):
            if tj in T.columns:
                wkt = float(T.loc[asal_kec, tj])
                tabel_opsi.append({
                    "Titik Tujuan": "Pasar Induk Pare" if tj == PARE else f"{jenis} — Kec. {tj}",
                    "Waktu (mnt)": round(wkt, 1),
                    "Status": "Ideal (≤30m)" if wkt <= 30 else ("Sedang (31-45m)" if wkt <= 45 else "Jauh (>45m)"),
                })
        st.markdown("<div style='font-size:0.84rem;font-weight:700;margin:0.85rem 0 0.35rem 0;'>Perbandingan Semua Pilihan Titik Setor:</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(tabel_opsi).sort_values("Waktu (mnt)"), hide_index=True, use_container_width=True)

    # ========================================================================
    # TAB 3: ANALISIS SKENARIO & SENSITIVITAS
    # ========================================================================
    with tab_map["📊 Grafik & Skenario"]:
        k_bandingan = max(len(pilih), 1)
        opsi = {
            f"Skenario Aktif ({k_bandingan} Unit)": pilih,
            f"Prioritas SAIPI (Top-{k_bandingan})": [c for c in rank.get(jenis, []) if c in T.columns][:k_bandingan],
            f"Optimal Jaringan ({k_bandingan} Unit)": penempatan_optimal(
                T, prod_ton, kandidat, k_bandingan, jenis, radius, susut, efek, kap, porsi_pare
            ),
        }
        baris = []
        for nama_strat, lok in opsi.items():
            h, _ = simulasi(T, prod_ton, lok, radius, susut, efek, kap, porsi_pare)
            baris.append({
                "Strategi": nama_strat,
                "Lokasi": ", ".join(lok) if lok else "-",
                "Rata (m)": round(h["t_rata_skenario"], 1),
                "Hemat (m)": round(h["turun_menit"], 1),
                f"≤{radius}m (%)": round(h["cakupan_skenario"] * 100, 1),
                "Susut Dicegah 5/15/30% (t/th)": (
                    f"{h['diselamatkan_rendah']:,.0f} / {h['diselamatkan_sedang']:,.0f} / {h['diselamatkan_tinggi']:,.0f}"
                ),
            })
        df_comp = pd.DataFrame(baris)

        st.markdown('<div class="panel-header">Komparasi Strategi Penempatan</div>', unsafe_allow_html=True)
        st.dataframe(df_comp, hide_index=True, use_container_width=True)

        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=df_comp["Strategi"],
            y=df_comp["Hemat (m)"],
            marker_color=["#334155", "#D97706", "#059669"],
            width=0.45,
            text=df_comp["Hemat (m)"].apply(lambda v: f"<b>{v:.1f} m</b>"),
            textposition="outside",
        ))
        fig1.update_layout(
            title="(a) Penghematan Waktu Angkut Antarstrategi",
            yaxis_title="Menit",
            yaxis_range=[0, max(df_comp["Hemat (m)"].max() * 1.28, 5)],
        )
        st.plotly_chart(gaya_plotly_mobile(fig1, 290), use_container_width=True)

        top12 = detail.sort_values("produksi_total_ton", ascending=False).head(12).iloc[::-1]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=top12["Kecamatan"],
            x=top12["t_ke_pare_menit"],
            name="Awal (ke Pare)",
            orientation="h",
            marker_color="#94A3B8",
        ))
        fig2.add_trace(go.Bar(
            y=top12["Kecamatan"],
            x=top12["t_skenario_menit"],
            name="Setelah Intervensi",
            orientation="h",
            marker_color="#059669",
        ))
        fig2.update_layout(
            title="(b) Waktu Tempuh 12 Sentra Utama (Menit)",
            xaxis_title="Menit",
            barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(gaya_plotly_mobile(fig2, 370), use_container_width=True)

        ks_kurva = [0, 1, 2, 3, 4]
        t_kurva, low_k, med_k, high_k = [], [], [], []
        for k_item in ks_kurva:
            lok_k = [] if k_item == 0 else penempatan_optimal(T, prod_ton, kandidat, k_item, jenis, radius, susut, efek, kap, porsi_pare)
            hk, _ = simulasi(T, prod_ton, lok_k, radius, susut, efek, kap, porsi_pare)
            t_kurva.append(hk["t_rata_skenario"])
            low_k.append(hk["diselamatkan_rendah"])
            med_k.append(hk["diselamatkan_sedang"])
            high_k.append(hk["diselamatkan_tinggi"])

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=["Awal", "1 Unit", "2 Unit", "3 Unit", "4 Unit"],
            y=t_kurva,
            mode="lines+markers+text",
            line=dict(color="#064E3B", width=3),
            marker=dict(size=8, color="#DC2626"),
            text=[f"{v:.1f}m" for v in t_kurva],
            textposition="top center",
        ))
        fig3.update_layout(
            title="(c) Kurva Efisiensi Waktu Tempuh (0–4 Unit)",
            yaxis_title="Menit",
            yaxis_range=[0, max(t_kurva) * 1.28],
        )
        st.plotly_chart(gaya_plotly_mobile(fig3, 290), use_container_width=True)

        x_lbl = ["1 Unit", "2 Unit", "3 Unit", "4 Unit"]
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=x_lbl, y=low_k[1:], name="Susut 5%", marker_color="#60A5FA"))
        fig4.add_trace(go.Bar(x=x_lbl, y=med_k[1:], name="Susut 15%", marker_color="#059669"))
        fig4.add_trace(go.Bar(x=x_lbl, y=high_k[1:], name="Susut 30%", marker_color="#D97706"))
        fig4.update_layout(
            title="(d) Potensi Susut Dicegah (ton/tahun)",
            yaxis_title="Ton/Tahun",
            barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(gaya_plotly_mobile(fig4, 310), use_container_width=True)

    # ========================================================================
    # TAB 4: DATA TERPADU & METODOLOGI
    # ========================================================================
    with tab_map["📑 Data 26 Kec"]:
        st.markdown('<div class="panel-header">Data Terpadu 26 Kecamatan Kab. Kediri</div>', unsafe_allow_html=True)
        master = ind.merge(
            detail[["Kecamatan", "t_ke_pare_menit", "t_skenario_menit", "turun_menit", "dilayani"]],
            on="Kecamatan",
            how="left",
        )
        if FILE_MODUL2.exists():
            m2 = pd.read_csv(FILE_MODUL2)
            cols_m2 = [c for c in ["Kecamatan", "spi_ekstrem", "t_mean_c", "rh_mean_pct", "cropland_ha", "jarak_terbangun_m"] if c in m2.columns]
            master = master.merge(m2[cols_m2], on="Kecamatan", how="left")

        st.dataframe(master.round(2), hide_index=True, use_container_width=True, height=340)
        st.download_button(
            label="📥 Unduh Data CSV (.CSV)",
            data=master.to_csv(index=False).encode("utf-8"),
            file_name=f"chili_chain_{jenis.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.divider()
        st.markdown("""
        ### Ringkasan Metodologi & Sumber Data
- **Data Spasial & Produksi:** BPS Kab. Kediri (2021–2025), tutupan lahan pertanian Dynamic World 10 m, Copernicus DEM GLO-30 (30 m), jaringan jalan OpenStreetMap/OSRM ($f = 0,85$), dan SPI-3 CHIRPS.
- **Indeks SAIPI & Optimasi:** Pembobotan hibrida subjektif–entropi Shannon ($\alpha = 0,5$), uji Monte Carlo 2.000 iterasi, dan algoritma *greedy p-median*.
- **Skenario Susut Pascapanen:** Mengacu pada *Lumaela (2024)*, *Iswari & Srimaryati (2014)*, dan *Mulyawanti dkk. (2024)* pada suhu penyimpanan 5–10°C.
        """)


if __name__ == "__main__":
    if "--bangun-matriks" in sys.argv:
        muat_atau_bangun_matriks(paksa=True)
    else:
        main()