#!/usr/bin/env python3
"""
KEDIRI VALUECHAIN - Modul 4 (Figur Publikasi Ilmiah Berbahasa Indonesia, 300 DPI)
Menghasilkan 6 figur multi-panel berstandar jurnal internasional (Elsevier / Scopus style)
dalam Bahasa Indonesia dari keluaran bersih Modul 1, 2, 3, dan 5.

Jalankan:
    python kediri_modul4_publikasi_jurnal.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "publikasi_jurnal_id"
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ----------------------------------------------------------------------------
# KONFIGURASI TIPOGRAFI & PALET STANDAR JURNAL INTERNASIONAL (OKABE-ITO)
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.2,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

PALETTE = {
    "navy": "#1B365D",
    "teal": "#00796B",
    "emerald": "#2E7D32",
    "amber": "#D97706",
    "crimson": "#B91C1C",
    "slate": "#64748B",
    "blue": "#0072B2",
    "orange": "#D55E00",
}


def _norm(nama):
    n = str(nama).upper().replace("KECAMATAN", "").replace("KEC.", "")
    return "".join(ch for ch in n if ch.isalnum())


def tambah_panah_utara(ax, x=0.93, y=0.92, size=0.06):
    ax.annotate(
        "U",
        xy=(x, y),
        xytext=(x, y - size),
        arrowprops=dict(facecolor="#1E293B", edgecolor="#1E293B", width=2.5, headwidth=7, headlength=7),
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        xycoords=ax.transAxes,
    )


def tambah_scalebar(ax, gdf_wgs84, panjang_km=10):
    minx, miny, maxx, maxy = gdf_wgs84.total_bounds
    lat_mid = (miny + maxy) / 2.0
    deg_per_km = 1.0 / (111.32 * np.cos(np.radians(lat_mid)))
    bar_deg = panjang_km * deg_per_km
    x0 = minx + (maxx - minx) * 0.06
    y0 = miny + (maxy - miny) * 0.06
    ax.plot([x0, x0 + bar_deg], [y0, y0], color="#0F172A", lw=2.2, zorder=10)
    ax.plot([x0, x0], [y0 - 0.004, y0 + 0.004], color="#0F172A", lw=1.5, zorder=10)
    ax.plot([x0 + bar_deg, x0 + bar_deg], [y0 - 0.004, y0 + 0.004], color="#0F172A", lw=1.5, zorder=10)
    ax.text(x0 + bar_deg / 2.0, y0 + 0.008, f"{panjang_km} km", ha="center", va="bottom",
            fontsize=7.5, fontweight="bold", color="#0F172A")


def muat_semua_data():
    f_ind = BASE / "hasil_modul1" / "indikator_modul1.csv"
    f_shp = BASE / "KAB_KEDIRI_26_KEC.shp"
    if not f_ind.exists():
        sys.exit("File hasil_modul1/indikator_modul1.csv tidak ditemukan. Jalankan Modul 1 terlebih dahulu.")
    ind = pd.read_csv(f_ind)

    for fn in ["data_modul2.csv", "data_tambahan.csv"]:
        p = BASE / fn
        if p.exists():
            df_extra = pd.read_csv(p)
            cols = [c for c in df_extra.columns if c != "Kecamatan" and c not in ind.columns]
            ind = ind.merge(df_extra[["Kecamatan"] + cols], on="Kecamatan", how="left")

    gdf = gpd.read_file(f_shp)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)
    gdf["_key"] = gdf["KECAMATAN"].map(_norm)
    ind["_key"] = ind["Kecamatan"].map(_norm)
    merged = gdf.merge(ind, on="_key", how="left").drop(columns=["_key"])
    return ind, merged


# ============================================================================
# FIGUR 1: ATLAS SPASIAL 2x2 (PRODUKSI, VOLATILITAS, IKLIM, LOGISTIK)
# ============================================================================
def buat_fig1_spatial_atlas(merged):
    from kediri_modul3_waktu_tempuh import PARE_DEFAULT
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.6))
    pts = merged.to_crs(32749).geometry.representative_point().to_crs(4326)

    panels = [
        (
            axes[0, 0],
            merged["prod_mean_kw"] / 10.0,
            "YlGn",
            "Rata-Rata Produksi (ton/tahun)",
            "(a) Konsentrasi Produksi Cabai Rawit (2021–2025)",
        ),
        (
            axes[0, 1],
            merged["cv_produksi"],
            "OrRd",
            "Koefisien Variasi (CV)",
            "(b) Volatilitas Pasokan Antartahun (CV)",
        ),
        (
            axes[1, 0],
            merged["spi_ekstrem"] * 100 if "spi_ekstrem" in merged.columns else merged["share_pct"],
            "PuBuGn",
            "Frekuensi Anomali SPI-3 Ekstrem (% bulan)" if "spi_ekstrem" in merged.columns else "Pangsa Produksi (%)",
            "(c) Paparan Anomali Iklim Ekstrem (|SPI-3| ≥ 1,5)",
        ),
        (
            axes[1, 1],
            merged["waktu_tempuh_pare_menit"] if "waktu_tempuh_pare_menit" in merged.columns else merged["share_pct"],
            "YlOrBr",
            "Waktu Tempuh Terkoreksi Lereng (menit)",
            "(d) Waktu Tempuh Logistik ke Pasar Induk Pare",
        ),
    ]

    for ax, series, cmap, cbar_label, title in panels:
        merged.plot(
            column=series,
            cmap=cmap,
            linewidth=0.65,
            edgecolor="#334155",
            legend=True,
            legend_kwds={"label": cbar_label, "shrink": 0.72, "pad": 0.02},
            ax=ax,
        )
        ax.scatter(
            [PARE_DEFAULT[1]], [PARE_DEFAULT[0]],
            marker="*", s=165, color="#DC2626", edgecolor="white", linewidth=0.8, zorder=8,
            label="Pasar Induk Pare",
        )
        for idx, row in merged.iterrows():
            val_prod = row.get("prod_mean_kw", 0) or 0
            fs = 6.2 if val_prod >= 10000 else 5.2
            fw = "bold" if val_prod >= 10000 else "normal"
            ax.annotate(
                str(row["Kecamatan"]).title(),
                xy=(pts.loc[idx].x, pts.loc[idx].y),
                ha="center", va="center", fontsize=fs, fontweight=fw, color="#0F172A",
            )
        ax.set_title(title, loc="left", pad=8)
        ax.set_xlabel("Bujur (°BT)", fontsize=8, fontweight="normal")
        ax.set_ylabel("Lintang (°LS)", fontsize=8, fontweight="normal")
        ax.tick_params(labelsize=7.5)
        tambah_panah_utara(ax)
        tambah_scalebar(ax, merged, panjang_km=10)

    axes[1, 1].legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.5)
    fig.suptitle("Sintesis Spasial Produksi, Risiko Agroklimat, dan Friksi Logistik Cabai Rawit Kabupaten Kediri",
                 fontsize=12.2, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig1_Atlas_Spasial_Kediri.png", dpi=300)
    plt.close(fig)
    print("  [OK] Fig1_Atlas_Spasial_Kediri.png")


# ============================================================================
# FIGUR 2: PARETO PRODUKSI & KUADRAN VOLATILITAS PASOKAN
# ============================================================================
def buat_fig2_production_quadrant(ind):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.2))

    # Panel (a): Pareto / Lollipop Top-12 + Pangsa Kumulatif
    top12 = ind.sort_values("prod_mean_kw", ascending=False).head(12).copy()
    top12["prod_ton"] = top12["prod_mean_kw"] / 10.0
    top12["cum_share"] = top12["share_pct"].cumsum()
    y_pos = np.arange(len(top12))[::-1]

    ax1.hlines(y=y_pos, xmin=0, xmax=top12["prod_ton"], color=PALETTE["teal"], lw=2.8)
    ax1.scatter(top12["prod_ton"], y_pos, color=PALETTE["navy"], s=55, zorder=4)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top12["Kecamatan"])
    ax1.set_xlabel("Rata-Rata Produksi Tahunan (ton/tahun)")
    ax1.set_title("(a) 12 Sentra Produksi Terbesar & Pangsa Kumulatif", loc="left", pad=10)
    ax1.grid(axis="x", linestyle="--", alpha=0.35)

    for yp, (_, r) in zip(y_pos, top12.iterrows()):
        ax1.text(r["prod_ton"] + 80, yp, f"{r['share_pct']:.1f}% (Σ {r['cum_share']:.0f}%)",
                 va="center", fontsize=7.8, color="#1E293B")
    ax1.set_xlim(0, top12["prod_ton"].max() * 1.34)

    # Panel (b): Matriks Kuadran Intervensi (Volume vs CV, warna = SPI-3)
    valid = ind[ind["cv_produksi"].notna() & (ind["prod_mean_kw"] > 0)].copy()
    valid["prod_ton"] = valid["prod_mean_kw"] / 10.0
    c_val = valid["spi_ekstrem"] * 100 if "spi_ekstrem" in valid.columns else valid["share_pct"]

    sc = ax2.scatter(
        valid["prod_ton"],
        valid["cv_produksi"],
        s=np.clip(valid["luas_mean_ha"] * 0.45, 35, 380),
        c=c_val,
        cmap="YlOrRd",
        edgecolor="#0F172A",
        linewidth=0.8,
        alpha=0.88,
        zorder=4,
    )
    ax2.set_xscale("log")
    ax2.axvline(1000, color="#475569", linestyle="--", lw=1.0, zorder=2)
    med_cv = valid["cv_produksi"].median()
    ax2.axhline(med_cv, color="#475569", linestyle=":", lw=1.0, zorder=2)

    for _, r in valid[valid["prod_ton"] >= 700].iterrows():
        ax2.annotate(
            r["Kecamatan"],
            (r["prod_ton"], r["cv_produksi"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold" if r["prod_ton"] >= 1000 else "normal",
        )

    ax2.text(0.97, 0.94, "Volume Tinggi + Volatilitas Tinggi\n(Zona Prioritas Cold Storage)",
             transform=ax2.transAxes, ha="right", va="top", fontsize=7.8, fontweight="bold",
             color="#991B1B", bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2", edgecolor="#FECACA"))
    ax2.text(0.97, 0.08, "Volume Tinggi + Pasokan Stabil\n(Zona Konsolidasi STA)",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=7.8, fontweight="bold",
             color="#065F46", bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECFDF5", edgecolor="#A7F3D0"))

    cbar = plt.colorbar(sc, ax=ax2, pad=0.02)
    cbar.set_label("Frekuensi SPI-3 Ekstrem (%)" if "spi_ekstrem" in valid.columns else "Pangsa (%)")
    ax2.set_xlabel("Rata-Rata Produksi Tahunan (ton, skala logaritmik)")
    ax2.set_ylabel("Volatilitas Pasokan (Koefisien Variasi / CV)")
    ax2.set_title("(b) Matriks Intervensi Infrastruktur (Volume vs. Volatilitas)", loc="left", pad=10)
    ax2.grid(True, which="major", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig2_Kuadran_Produksi_Volatilitas.png", dpi=300)
    plt.close(fig)
    print("  [OK] Fig2_Kuadran_Produksi_Volatilitas.png")


# ============================================================================
# FIGUR 3: PEMBOBOTAN HYBRID SAIPI & DEKOMPOSISI SKOR PRIORITAS
# ============================================================================
def buat_fig3_hybrid_ranking(ind):
    from kediri_modul1_analisis import KRITERIA, normalisasi, bobot_hybrid

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.9))

    skema_cs = "COLD_STORAGE"
    kand_cs = ind[ind["layak_cold_storage"]].copy() if "layak_cold_storage" in ind.columns else ind[ind["lolos_ambang"]].copy()
    ada_cs = [k for k in KRITERIA[skema_cs] if k in kand_cs.columns and kand_cs[k].notna().any()]
    X_cs = normalisasi(kand_cs, ada_cs, {k: KRITERIA[skema_cs][k] for k in ada_cs})
    w_hyb_cs, w_ahli_cs, w_ent_cs = bobot_hybrid(X_cs, skema_cs)

    label_map = {
        "prod_mean_kw": "Volume\nProduksi",
        "cv_produksi": "Volatilitas\nPasokan (CV)",
        "spi_ekstrem": "Risiko Iklim\n(SPI-3)",
        "jarak_terbangun_m": "Kedekatan\nUtilitas Listrik",
        "luas_mean_ha": "Luas Panen\nRata-Rata",
        "waktu_tempuh_pare_menit": "Waktu Tempuh\nke Pare",
    }

    x = np.arange(len(ada_cs))
    width = 0.26
    axes[0].bar(x - width, w_ahli_cs.values, width, label="Bobot Ahli / AHP", color=PALETTE["navy"])
    axes[0].bar(x, w_ent_cs.values, width, label="Bobot Shannon Entropy", color=PALETTE["amber"])
    axes[0].bar(x + width, w_hyb_cs.values, width, label="Bobot Hybrid (α = 0,5)", color=PALETTE["teal"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([label_map.get(c, c) for c in ada_cs], fontsize=8)
    axes[0].set_ylabel("Bobot Ternormalisasi")
    axes[0].set_title("(a) Kalibrasi Bobot Hybrid (Cold Storage)", loc="left", pad=10)
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    configs = [
        (axes[1], "STA", "layak_sta", "(b) Dekomposisi Skor Prioritas STA"),
        (axes[2], "COLD_STORAGE", "layak_cold_storage", "(c) Dekomposisi Skor Prioritas Cold Storage"),
    ]
    colors_stack = [PALETTE["navy"], PALETTE["teal"], PALETTE["amber"], PALETTE["crimson"]]

    for ax, skema, col_layak, title in configs:
        kand = ind[ind[col_layak]].copy() if col_layak in ind.columns else ind[ind["lolos_ambang"]].copy()
        ada = [k for k in KRITERIA[skema] if k in kand.columns and kand[k].notna().any()]
        X = normalisasi(kand, ada, {k: KRITERIA[skema][k] for k in ada})
        w_hyb, _, _ = bobot_hybrid(X, skema)
        contrib = X.multiply(w_hyb, axis=1)
        contrib["Total"] = contrib.sum(axis=1)
        contrib["Kecamatan"] = kand["Kecamatan"].values
        contrib = contrib.sort_values("Total", ascending=True).tail(8)

        left = np.zeros(len(contrib))
        for i, col in enumerate(ada):
            vals = contrib[col].to_numpy()
            ax.barh(contrib["Kecamatan"], vals, left=left, color=colors_stack[i % len(colors_stack)],
                    edgecolor="white", linewidth=0.5, label=label_map.get(col, col).replace("\n", " "))
            left += vals

        for yp, tot in enumerate(contrib["Total"]):
            ax.text(tot + 0.012, yp, f"{tot:.3f}", va="center", fontsize=8, fontweight="bold")

        ax.set_xlim(0, contrib["Total"].max() * 1.19)
        ax.set_xlabel("Skor Komposit SAIPI Hybrid")
        ax.set_title(title, loc="left", pad=10)
        ax.legend(frameon=False, loc="lower right", fontsize=7.5)
        ax.grid(axis="x", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig3_Bobot_Hybrid_Peringkat_SAIPI.png", dpi=300)
    plt.close(fig)
    print("  [OK] Fig3_Bobot_Hybrid_Peringkat_SAIPI.png")


# ============================================================================
# FIGUR 4: UJI SENSITIVITAS & KETAHANAN MONTE CARLO (2.000 SIMULASI)
# ============================================================================
def buat_fig4_montecarlo_robustness(ind):
    from kediri_modul1_analisis import KRITERIA, normalisasi, bobot_hybrid, N_SIMULASI, PERTURBASI, SEED

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9))
    rng = np.random.default_rng(SEED)

    for ax, skema, col_layak, title in [
        (axes[0], "STA", "layak_sta", "(a) Stabilitas Peringkat STA (2.000 Simulasi Monte Carlo)"),
        (axes[1], "COLD_STORAGE", "layak_cold_storage", "(b) Stabilitas Peringkat Cold Storage (2.000 Simulasi Monte Carlo)"),
    ]:
        kand = ind[ind[col_layak]].copy() if col_layak in ind.columns else ind[ind["lolos_ambang"]].copy()
        ada = [k for k in KRITERIA[skema] if k in kand.columns and kand[k].notna().any()]
        X = normalisasi(kand, ada, {k: KRITERIA[skema][k] for k in ada})
        w, _, _ = bobot_hybrid(X, skema)

        kand["base_score"] = X.to_numpy() @ w.to_numpy()
        kand = kand.sort_values("base_score", ascending=True)
        Xv = X.loc[kand.index].to_numpy()
        names = kand["Kecamatan"].tolist()

        ranks_mat = np.zeros((N_SIMULASI, len(names)))
        for i in range(N_SIMULASI):
            wp = w.to_numpy() * rng.uniform(1 - PERTURBASI, 1 + PERTURBASI, len(w))
            wp = wp / wp.sum()
            s = Xv @ wp
            ranks_mat[i, :] = pd.Series(s).rank(ascending=False, method="min").to_numpy()

        top5_prob = (ranks_mat <= 5).mean(axis=0) * 100

        ax.boxplot(
            [ranks_mat[:, j] for j in range(len(names))],
            orientation="horizontal",
            tick_labels=names,
            patch_artist=True,
            widths=0.55,
            showfliers=False,
            boxprops=dict(facecolor="#CCFBF1", edgecolor=PALETTE["teal"], linewidth=1.1),
            medianprops=dict(color=PALETTE["crimson"], linewidth=1.8),
            whiskerprops=dict(color=PALETTE["slate"]),
            capprops=dict(color=PALETTE["slate"]),
        )
        ax.axvline(5.5, color="#DC2626", linestyle="--", lw=1.0, label="Batas Prioritas 5 Besar")
        ax.invert_xaxis()
        ax.set_xlim(len(names) + 0.9, 0.1)
        ax.set_xlabel("Distribusi Peringkat Simulasi (1 = Prioritas Tertinggi)")
        ax.set_title(title, loc="left", pad=10)
        ax.grid(axis="x", linestyle="--", alpha=0.35)

        for j, prob in enumerate(top5_prob):
            ax.text(0.35, j + 1, f"P(Top-5)={prob:.0f}%", va="center", ha="right",
                    fontsize=7.6, fontweight="bold" if prob >= 80 else "normal",
                    color="#065F46" if prob >= 80 else "#475569")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig4_Sensitivitas_MonteCarlo.png", dpi=300)
    plt.close(fig)
    print("  [OK] Fig4_Sensitivitas_MonteCarlo.png")


# ============================================================================
# FIGUR 5: FRIKSI TOPOGRAFI & PENALTI KEMIRINGAN LERENG
# ============================================================================
def buat_fig5_logistics_slope(ind):
    if "waktu_tempuh_pare_menit" not in ind.columns or "waktu_tempuh_datar_menit" not in ind.columns:
        print("  [SKIP] Data waktu tempuh Modul 3 belum tersedia untuk Fig 5.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.2))

    d15 = ind.sort_values("waktu_tempuh_pare_menit", ascending=True).tail(15).copy()
    y = np.arange(len(d15))

    ax1.hlines(y=y, xmin=d15["waktu_tempuh_datar_menit"], xmax=d15["waktu_tempuh_pare_menit"],
               color="#94A3B8", lw=2.2, zorder=2)
    ax1.scatter(d15["waktu_tempuh_datar_menit"], y, color=PALETTE["blue"], s=50, label="Waktu Rute Datar (Tanpa Lereng)", zorder=3)
    ax1.scatter(d15["waktu_tempuh_pare_menit"], y, color=PALETTE["orange"], s=55, label="Waktu Riil Terkoreksi Lereng (DEM 30m)", zorder=4)

    for yi, (_, r) in enumerate(d15.iterrows()):
        diff = r["waktu_tempuh_pare_menit"] - r["waktu_tempuh_datar_menit"]
        ax1.text(r["waktu_tempuh_pare_menit"] + 0.8, yi, f"{r['waktu_tempuh_pare_menit']:.1f} mnt (+{diff:.1f})",
                 va="center", fontsize=7.5, color="#1E293B")

    ax1.set_yticks(y)
    ax1.set_yticklabels(d15["Kecamatan"])
    ax1.set_xlim(0, d15["waktu_tempuh_pare_menit"].max() * 1.26)
    ax1.set_xlabel("Waktu Tempuh ke Pasar Induk Pare (menit)")
    ax1.set_title("(a) Dampak Penalti Kemiringan Lereng terhadap Waktu Angkut", loc="left", pad=10)
    ax1.legend(frameon=False, loc="lower right")
    ax1.grid(axis="x", linestyle="--", alpha=0.35)

    sc = ax2.scatter(
        ind["jarak_rute_km"],
        ind["rasio_tanjakan"],
        s=np.clip(ind["prod_mean_kw"] / 25.0, 35, 350),
        c=ind["waktu_tempuh_pare_menit"],
        cmap="viridis",
        edgecolor="#0F172A",
        linewidth=0.75,
        alpha=0.88,
    )
    for _, r in ind[(ind["rasio_tanjakan"] >= 1.04) | (ind["prod_mean_kw"] >= 15000)].iterrows():
        ax2.annotate(
            r["Kecamatan"],
            (r["jarak_rute_km"], r["rasio_tanjakan"]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=7.8,
            fontweight="bold" if r["rasio_tanjakan"] >= 1.05 else "normal",
        )
    cbar = plt.colorbar(sc, ax=ax2, pad=0.02)
    cbar.set_label("Total Waktu Tempuh (menit)")
    ax2.set_xlabel("Jarak Jaringan Jalan OSM ke Pasar Induk Pare (km)")
    ax2.set_ylabel("Faktor Penalti Tanjakan (t_lereng / t_datar)")
    ax2.set_title("(b) Jarak Jaringan Jalan vs. Friksi Lereng Gunung Kelud & Wilis", loc="left", pad=10)
    ax2.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig5_Friksi_Topografi_Logistik.png", dpi=300)
    plt.close(fig)
    print("  [OK] Fig5_Friksi_Topografi_Logistik.png")


# ============================================================================
# FIGUR 6: SIMULASI SKENARIO KEBIJAKAN (1-4 UNIT & 3 SKENARIO SUSUT)
# ============================================================================
def buat_fig6_policy_scenarios(ind):
    f_mat = BASE / "hasil_modul5" / "matriks_waktu.csv"
    if not f_mat.exists():
        print("  [SKIP] Matriks Modul 5 belum tersedia untuk Fig 6.")
        return

    from kediri_modul5_simulator import simulasi, penempatan_optimal

    T = pd.read_csv(f_mat, index_col=0)
    prod_ton = ind.set_index("Kecamatan")["prod_mean_kw"].fillna(0.0) / 10.0
    kand_sta = ind.loc[ind["layak_sta"], "Kecamatan"].tolist() if "layak_sta" in ind.columns else ind.loc[ind["lolos_ambang"], "Kecamatan"].tolist()
    kand_cs = ind.loc[ind["layak_cold_storage"], "Kecamatan"].tolist() if "layak_cold_storage" in ind.columns else kand_sta

    ks = [0, 1, 2, 3, 4]
    t_sta, cov_sta = [], []
    loss_low, loss_med, loss_high = [], [], []

    for k in ks:
        if k == 0:
            lok_sta, lok_cs = [], []
        else:
            lok_sta = penempatan_optimal(T, prod_ton, kand_sta, k, "STA", radius=30, susut=0.15, efek=0.50, kapasitas=5000, porsi_pare=1.0)
            lok_cs = penempatan_optimal(T, prod_ton, kand_cs, k, "Cold Storage", radius=30, susut=0.15, efek=0.50, kapasitas=5000, porsi_pare=1.0)

        h_sta, _ = simulasi(T, prod_ton, lok_sta, radius=30, susut=0.15, efek=0.50, kapasitas=5000, porsi_pare=1.0)
        h_cs, _ = simulasi(T, prod_ton, lok_cs, radius=30, susut=0.15, efek=0.50, kapasitas=5000, porsi_pare=1.0)

        t_sta.append(h_sta["t_rata_skenario"])
        cov_sta.append(h_sta["cakupan_skenario"] * 100)
        loss_low.append(h_cs["diselamatkan_rendah"])
        loss_med.append(h_cs["diselamatkan_sedang"])
        loss_high.append(h_cs["diselamatkan_tinggi"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.9))

    ax1_r = ax1.twinx()
    ax1_r.spines["top"].set_visible(False)
    ax1_r.spines["right"].set_visible(True)

    l1 = ax1.plot(ks, t_sta, marker="o", lw=2.4, color=PALETTE["navy"], label="Rata-Rata Waktu Angkut Tertimbang (menit)")
    l2 = ax1_r.plot(ks, cov_sta, marker="s", lw=2.4, linestyle="--", color=PALETTE["teal"], label="Cakupan Pasokan Radius ≤ 30 Menit (%)")

    for k, tv, cv in zip(ks, t_sta, cov_sta):
        ax1.annotate(f"{tv:.1f} mnt", (k, tv), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, fontweight="bold", color=PALETTE["navy"])
        ax1_r.annotate(f"{cv:.1f}%", (k, cv), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=8, fontweight="bold", color=PALETTE["teal"])

    ax1.set_xticks(ks)
    ax1.set_xticklabels(["Kondisi Awal\n(Pare Saja)", "1 Unit", "2 Unit", "3 Unit", "4 Unit"])
    ax1.set_ylabel("Waktu Angkut Rata-Rata Tertimbang (menit)", color=PALETTE["navy"])
    ax1_r.set_ylabel("Cakupan Produksi Radius ≤ 30 Menit (%)", color=PALETTE["teal"])
    ax1.set_ylim(0, max(t_sta) * 1.24)
    ax1_r.set_ylim(0, 114)
    ax1.set_title("(a) Efisiensi Marjinal Logistik berdasarkan Jumlah Unit STA", loc="left", pad=10)
    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines], frameon=False, loc="center right", fontsize=7.8)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    x_cs = np.arange(1, 5)
    w = 0.25
    ax2.bar(x_cs - w, loss_low[1:], w, label="Skenario Susut Rendah (5%)", color="#93C5FD", edgecolor="#1E3A8A", linewidth=0.6)
    ax2.bar(x_cs, loss_med[1:], w, label="Skenario Susut Sedang (15%)", color=PALETTE["teal"], edgecolor="#064E3B", linewidth=0.6)
    ax2.bar(x_cs + w, loss_high[1:], w, label="Skenario Susut Tinggi (30%)", color=PALETTE["amber"], edgecolor="#78350F", linewidth=0.6)

    for idx, k_val in enumerate(x_cs):
        ax2.text(k_val, loss_med[idx + 1] + 40, f"{loss_med[idx + 1]:,.0f} t", ha="center", va="bottom", fontsize=7.8, fontweight="bold")

    ax2.set_xticks(x_cs)
    ax2.set_xticklabels(["1 Unit", "2 Unit", "3 Unit", "4 Unit"])
    ax2.set_ylabel("Potensi Susut Pascapanen Dicegah (ton/tahun)")
    ax2.set_ylim(0, max(loss_high[1:]) * 1.22)
    ax2.set_title("(b) Dampak Cold Storage pada Tiga Skenario Susut Literatur", loc="left", pad=10)
    ax2.legend(frameon=False, loc="upper left")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig6_Skenario_Kebijakan_Pascapanen.png", dpi=300)
    plt.close(fig)
    print("  [OK] Fig6_Skenario_Kebijakan_Pascapanen.png")


def main():
    print("Membuat 6 Figur Publikasi Berbahasa Indonesia (300 DPI) di folder publikasi_jurnal_id/...")
    ind, merged = muat_semua_data()
    buat_fig1_spatial_atlas(merged)
    buat_fig2_production_quadrant(ind)
    buat_fig3_hybrid_ranking(ind)
    buat_fig4_montecarlo_robustness(ind)
    buat_fig5_logistics_slope(ind)
    buat_fig6_policy_scenarios(ind)
    print(f"\nSelesai! Seluruh grafik 300 DPI berbahasa Indonesia tersimpan di: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()