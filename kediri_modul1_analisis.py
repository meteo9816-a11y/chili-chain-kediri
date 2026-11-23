#!/usr/bin/env python3
"""
KEDIRI VALUECHAIN - Modul 1 (Production Intelligence) + SAIPI Hybrid + Integrasi SHP

Pembaruan metodologis:
1. Pembobotan Hybrid: gabungan bobot ahli/AHP (w_ahli) dan Shannon Entropy (w_entropy)
   untuk meredam bias subjektif maupun distorsi pencilan (Shen & Liao, 2022).
2. Syarat Kelayakan (Eligibility Filter): untuk Cold Storage, jarak ke kawasan terbangun
   (proksi utilitas listrik) dan paparan SPI-3 ekstrem digunakan sebagai syarat kelayakan
   (Upadhyay & Bhattacharya, 2019).

Jalankan:
    python kediri_modul1_analisis.py [path_csv] [folder_output] [path_shp]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- util lokasi file & konsol ----------------------------------------------
BASE = Path(__file__).resolve().parent
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
    sys.exit(f"CSV data cabai tidak ditemukan di {BASE}.")


def cari_shp():
    kandidat = [
        BASE / "KAB_KEDIRI_26_KEC.shp",
        Path(r"D:\LOMBA_BI\KAB_KEDIRI_26_KEC.shp"),
    ]
    for p in kandidat:
        if p.exists():
            return str(p)
    sisa = sorted(BASE.glob("*.shp"))
    return str(sisa[0]) if sisa else None


# ----------------------------------------------------------------------------
# KONFIGURASI METODOLOGI & SYARAT KELAYAKAN
# ----------------------------------------------------------------------------
TAHUN = [2021, 2022, 2023, 2024, 2025]
BARIS_TOTAL = "Kabupaten Kediri"

# Ambang kelayakan (Eligibility Thresholds)
AMBANG_PRODUKSI_KW = 10_000          # minimum rata-rata produksi (kw/tahun)
MAKS_JARAK_TERBANGUN_M = 2500.0      # syarat jarak maks ke kawasan terbangun (proksi akses listrik 3-fasa)
MIN_SPI_EKSTREM = 0.01               # syarat minimal paparan anomali iklim |SPI-3| >= 1,5

# Validasi kualitas data
MIN_TAHUN_VALID = 3
AMBANG_PRODUKTIVITAS = 200
AMBANG_CV = 0.9

# Konfigurasi Pembobotan Hybrid (Shen & Liao, 2022):
# w_hybrid = ALPHA_HYBRID * w_ahli + (1 - ALPHA_HYBRID) * w_entropy
ALPHA_HYBRID = 0.50
N_SIMULASI = 2000
PERTURBASI = 0.20
SEED = 42

# Kriteria & Bobot Ahli (AHP / Pakar Agribisnis-Logistik)
KRITERIA = {
    "STA": {
        "prod_mean_kw": "benefit",
        "luas_mean_ha": "benefit",
        "waktu_tempuh_pare_menit": "benefit",   # makin jauh/lama ke Pare, makin butuh titik konsolidasi
    },
    "COLD_STORAGE": {
        "prod_mean_kw": "benefit",
        "cv_produksi": "benefit",               # fluktuasi pasokan musiman
        "spi_ekstrem": "benefit",               # kerentanan anomali iklim (SPI-3)
        "jarak_terbangun_m": "cost",            # kedekatan ke kawasan terbangun (utilitas)
    },
}

BOBOT_AHLI = {
    "STA": {
        "prod_mean_kw": 0.45,
        "luas_mean_ha": 0.20,
        "waktu_tempuh_pare_menit": 0.35,
    },
    "COLD_STORAGE": {
        "prod_mean_kw": 0.40,
        "cv_produksi": 0.25,
        "spi_ekstrem": 0.20,
        "jarak_terbangun_m": 0.15,
    },
}


# ----------------------------------------------------------------------------
# 1. LOAD & BERSIHKAN
# ----------------------------------------------------------------------------
def muat_data(path):
    df = pd.read_csv(path, dtype={"Kode_Kecamatan": str})
    print(f"Membaca data: {path}")
    perlu = [f"Produksi_{y}_kw" for y in TAHUN] + [f"Luas_Panen_{y}_ha" for y in TAHUN]
    hilang = [c for c in perlu if c not in df.columns]
    if hilang:
        sys.exit(f"CSV tidak lengkap, kolom hilang: {hilang}.")
    total = df[df["Kecamatan"] == BARIS_TOTAL]
    kec = df[df["Kecamatan"] != BARIS_TOTAL].copy().reset_index(drop=True)

    print("=" * 70)
    print("VALIDASI DATA")
    print("=" * 70)
    print(f"Jumlah kecamatan : {len(kec)} (harus 26)")
    if not total.empty:
        for y in TAHUN:
            for jenis, kol in [("Produksi", f"Produksi_{y}_kw"), ("Luas", f"Luas_Panen_{y}_ha")]:
                if kol in df.columns:
                    s, t = kec[kol].sum(), total[kol].iloc[0]
                    ok = "OK" if abs(s - t) <= max(2, 0.001 * t) else "BEDA"
                    print(f"  {jenis:8s} {y}: jumlah kecamatan={s:,.1f} vs total={t:,.1f} [{ok}]")
    na = kec.isna().sum()
    na = na[na > 0]
    if len(na):
        print("Sel kosong (diperlakukan sebagai TIDAK TERCATAT, bukan nol):")
        for kol, n in na.items():
            nama = ", ".join(kec.loc[kec[kol].isna(), "Kecamatan"])
            print(f"  {kol}: {n} -> {nama}")
    return kec


# ----------------------------------------------------------------------------
# 2. INDIKATOR MODUL 1 & SYARAT KELAYAKAN
# ----------------------------------------------------------------------------
def hitung_indikator(kec):
    P = kec[[f"Produksi_{y}_kw" for y in TAHUN]].to_numpy(float)
    L = kec[[f"Luas_Panen_{y}_ha" for y in TAHUN]].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        Y = np.where(L > 0, P / L, np.nan)

    out = pd.DataFrame({"Kecamatan": kec["Kecamatan"]})
    out["n_tahun_valid"] = np.sum(~np.isnan(P), axis=1)
    out["prod_mean_kw"] = np.nanmean(P, axis=1)
    out["prod_median_kw"] = np.nanmedian(P, axis=1)
    out["prod_2025_kw"] = P[:, -1]
    out["luas_mean_ha"] = np.nanmean(L, axis=1)
    out["produktivitas_median"] = np.nanmedian(Y, axis=1)
    out["produktivitas_max"] = np.nanmax(Y, axis=1)

    sd = np.nanstd(P, axis=1, ddof=1)
    cv = sd / out["prod_mean_kw"].to_numpy()
    cv[out["n_tahun_valid"].to_numpy() < MIN_TAHUN_VALID] = np.nan
    out["cv_produksi"] = cv
    out["max_per_median"] = np.nanmax(P, axis=1) / out["prod_median_kw"].replace(0, np.nan)

    tren = []
    x = np.array(TAHUN, float)
    for row, m in zip(P, out["prod_mean_kw"]):
        ok = ~np.isnan(row)
        tren.append(np.polyfit(x[ok], row[ok], 1)[0] / m * 100 if ok.sum() >= 3 else np.nan)
    out["tren_pct_per_tahun"] = tren

    tot = out["prod_mean_kw"].sum()
    out["share_pct"] = out["prod_mean_kw"] / tot * 100

    flag = []
    for i in range(len(out)):
        f = []
        if np.nanmax(Y[i]) > AMBANG_PRODUKTIVITAS:
            f.append(f"produktivitas>{AMBANG_PRODUKTIVITAS}")
        if out.loc[i, "cv_produksi"] > AMBANG_CV:
            f.append(f"CV>{AMBANG_CV}")
        if out.loc[i, "max_per_median"] > 5:
            f.append("lonjakan>5x median")
        if out.loc[i, "n_tahun_valid"] < len(TAHUN):
            f.append("data tidak lengkap")
        flag.append("; ".join(f))
    out["flag_kualitas"] = flag
    out["lolos_ambang"] = out["prod_mean_kw"] >= AMBANG_PRODUKSI_KW
    return out


def terapkan_syarat_kelayakan(ind):
    """Evaluasi syarat kelayakan spesifik untuk STA dan Cold Storage (termasuk SPI & jarak terbangun)."""
    ind["layak_sta"] = ind["prod_mean_kw"] >= AMBANG_PRODUKSI_KW

    layak_cs = ind["prod_mean_kw"] >= AMBANG_PRODUKSI_KW
    alasan_cs = []
    for _, r in ind.iterrows():
        alasan = []
        if r["prod_mean_kw"] < AMBANG_PRODUKSI_KW:
            alasan.append(f"produksi<{AMBANG_PRODUKSI_KW}kw")
        if "jarak_terbangun_m" in ind.columns and pd.notna(r.get("jarak_terbangun_m")):
            if r["jarak_terbangun_m"] > MAKS_JARAK_TERBANGUN_M:
                alasan.append(f"jarak_utilitas>{MAKS_JARAK_TERBANGUN_M:.0f}m")
        if "spi_ekstrem" in ind.columns and pd.notna(r.get("spi_ekstrem")):
            if r["spi_ekstrem"] < MIN_SPI_EKSTREM:
                alasan.append("tanpa_paparan_spi_ekstrem")
        alasan_cs.append("; ".join(alasan))

    ind["alasan_kelayakan_cs"] = alasan_cs
    ind["layak_cold_storage"] = layak_cs & (ind["alasan_kelayakan_cs"] == "")
    return ind


# ----------------------------------------------------------------------------
# 3. SAIPI HYBRID: normalisasi + hybrid (Entropy + Ahli) + sensitivitas
# ----------------------------------------------------------------------------
def normalisasi(df, kolom, arah):
    X = pd.DataFrame(index=df.index)
    for k in kolom:
        v = df[k].astype(float)
        lo, hi = v.min(), v.max()
        z = (v - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=v.index)
        X[k] = z if arah[k] == "benefit" else 1 - z
    return X.fillna(0.0)


def bobot_entropy(X):
    P = X.to_numpy(float) + 1e-12
    P = P / P.sum(axis=0)
    n = P.shape[0]
    e = -(P * np.log(P)).sum(axis=0) / np.log(n)
    d = 1 - e
    if d.sum() <= 0:
        return pd.Series(1.0 / len(X.columns), index=X.columns)
    return pd.Series(d / d.sum(), index=X.columns)


def bobot_hybrid(X, skema):
    """Kombinasi linear bobot subjektif terstruktur (AHP/Ahli) dan objektif (Shannon Entropy)."""
    w_ent = bobot_entropy(X)
    raw_ahli = BOBOT_AHLI.get(skema, {})
    w_ahli = pd.Series([raw_ahli.get(c, 1.0) for c in X.columns], index=X.columns, dtype=float)
    w_ahli = w_ahli / w_ahli.sum()
    w_comb = ALPHA_HYBRID * w_ahli + (1.0 - ALPHA_HYBRID) * w_ent
    return w_comb / w_comb.sum(), w_ahli, w_ent


def hitung_saipi(ind, skema, kriteria):
    kol_layak = "layak_sta" if skema == "STA" else "layak_cold_storage"
    kand = ind[ind[kol_layak]].copy() if kol_layak in ind.columns else ind[ind["lolos_ambang"]].copy()

    ada = [k for k in kriteria if k in kand.columns and kand[k].notna().any()]
    hilang = [k for k in kriteria if k not in ada]
    if hilang:
        print(f"  [{skema}] kriteria belum tersedia, dilewati: {', '.join(hilang)}")
    if len(ada) < 2:
        print(f"  [{skema}] kurang dari 2 kriteria, ranking tidak dihitung.")
        return None, None, None

    arah = {k: kriteria[k] for k in ada}
    X = normalisasi(kand, ada, arah)
    w, w_ahli, w_ent = bobot_hybrid(X, skema)

    kand["skor"] = X.to_numpy() @ w.to_numpy()
    kand["rank"] = kand["skor"].rank(ascending=False, method="min").astype(int)
    kand = kand.sort_values("rank")

    # Uji sensitivitas Monte Carlo (perturbasi +/-20% pada bobot hybrid)
    rng = np.random.default_rng(SEED)
    n = len(kand)
    top5 = pd.Series(0, index=kand["Kecamatan"])
    korelasi = []
    base_rank = kand["skor"].rank(ascending=False).to_numpy()
    Xv = X.loc[kand.index].to_numpy()
    for _ in range(N_SIMULASI):
        wp = w.to_numpy() * rng.uniform(1 - PERTURBASI, 1 + PERTURBASI, len(w))
        wp = wp / wp.sum()
        s = Xv @ wp
        r = pd.Series(s).rank(ascending=False).to_numpy()
        korelasi.append(pd.Series(base_rank).corr(pd.Series(r), method="spearman"))
        top5.loc[kand["Kecamatan"].to_numpy()[np.argsort(-s)[: min(5, n)]]] += 1

    sens = pd.DataFrame({
        "Kecamatan": top5.index,
        "persen_masuk_top5": top5.to_numpy() / N_SIMULASI * 100,
    }).sort_values("persen_masuk_top5", ascending=False)

    print(f"  [{skema}] kandidat lolos syarat kelayakan: {len(kand)} kecamatan")
    print(f"  [{skema}] bobot Ahli   : " + ", ".join(f"{k}={v:.2f}" for k, v in w_ahli.items()))
    print(f"  [{skema}] bobot Entropy: " + ", ".join(f"{k}={v:.2f}" for k, v in w_ent.items()))
    print(f"  [{skema}] bobot HYBRID : " + ", ".join(f"{k}={v:.2f}" for k, v in w.items()))
    print(f"  [{skema}] rata-rata Spearman setelah perturbasi: {np.nanmean(korelasi):.3f}")
    return kand, w, sens


# ----------------------------------------------------------------------------
# 4. INTEGRASI SPASIAL (SHP) & PETA
# ----------------------------------------------------------------------------
def gabung_dan_simpan_shp(ind, hasil, path_shp, folder):
    if not path_shp or not Path(path_shp).exists():
        print(f"\n[PERINGATAN] File SHP tidak ditemukan ({path_shp}), ekspor spasial dilewati.")
        return None

    print(f"\nMembaca batas kecamatan SHP: {path_shp}")
    gdf = gpd.read_file(path_shp)
    gdf["_key"] = gdf["KECAMATAN"].astype(str).str.upper().str.strip()

    ind_sp = ind.copy()
    ind_sp["_key"] = ind_sp["Kecamatan"].astype(str).str.upper().str.strip()

    for skema, (kand, _, _) in hasil.items():
        if kand is not None:
            sub = kand[["Kecamatan", "skor", "rank"]].copy()
            sub["_key"] = sub["Kecamatan"].astype(str).str.upper().str.strip()
            sub = sub.rename(columns={
                "skor": f"skor_{skema.lower()}",
                "rank": f"rank_{skema.lower()}",
            })
            ind_sp = ind_sp.merge(sub[["_key", f"skor_{skema.lower()}", f"rank_{skema.lower()}"]], on="_key", how="left")

    merged = gdf.merge(ind_sp, on="_key", how="left").drop(columns=["_key"])
    gagal_join = merged[merged["prod_mean_kw"].isna()]["KECAMATAN"].tolist()
    if gagal_join:
        print(f"  [PERINGATAN] Kecamatan SHP tidak cocok dengan CSV: {gagal_join}")
    else:
        print("  Join SHP <-> CSV sukses 100% (26/26 kecamatan cocok).")

    merged.to_file(folder / "indikator_modul1_spasial.geojson", driver="GeoJSON")
    merged.to_file(folder / "indikator_modul1_spasial.shp")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    merged.plot(
        column=merged["prod_mean_kw"] / 100,
        cmap="YlGn",
        linewidth=0.7,
        edgecolor="black",
        legend=True,
        legend_kwds={"label": "Rata-rata Produksi (ton)", "shrink": 0.75},
        missing_kwds={"color": "lightgrey", "label": "No data"},
        ax=axes[0],
    )
    axes[0].set_title("Sebaran Rata-rata Produksi Cabai Rawit (2021-2025)")
    axes[0].set_axis_off()

    merged.plot(
        column="cv_produksi",
        cmap="OrRd",
        linewidth=0.7,
        edgecolor="black",
        legend=True,
        legend_kwds={"label": "Koefisien Variasi (CV)", "shrink": 0.75},
        missing_kwds={"color": "lightgrey", "label": "No data"},
        ax=axes[1],
    )
    axes[1].set_title("Stabilitas Pasokan (CV Produksi)")
    axes[1].set_axis_off()

    pts = merged.to_crs(32749).geometry.representative_point().to_crs(merged.crs)
    for idx, row in merged.iterrows():
        for ax in axes:
            ax.annotate(
                row["KECAMATAN"].title(),
                xy=(pts.loc[idx].x, pts.loc[idx].y),
                ha="center",
                va="center",
                fontsize=6.5,
                color="#111111",
            )

    fig.tight_layout()
    fig.savefig(folder / "04_peta_spasial_modul1.png", dpi=180)
    plt.close(fig)
    return merged


# ----------------------------------------------------------------------------
# 5. GRAFIK NON-SPASIAL
# ----------------------------------------------------------------------------
def grafik(ind, hasil, folder):
    d = ind.sort_values("prod_mean_kw", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(d["Kecamatan"][::-1], d["prod_mean_kw"][::-1] / 100, color="#2E7D32")
    ax.set_xlabel("Rata-rata produksi 2021-2025 (ton)")
    ax.set_title("12 Kecamatan Penghasil Cabai Rawit Terbesar")
    for i, (v, s) in enumerate(zip(d["prod_mean_kw"][::-1] / 100, d["share_pct"][::-1])):
        ax.text(v, i, f" {s:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "01_top_produksi.png", dpi=150)
    plt.close(fig)

    d = ind[ind["cv_produksi"].notna()]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(d["prod_mean_kw"] / 100, d["cv_produksi"], s=40, color="#C62828", alpha=0.7)
    ax.set_xscale("log")
    for _, r in d[d["prod_mean_kw"] >= 10_000].iterrows():
        ax.annotate(r["Kecamatan"], (r["prod_mean_kw"] / 100, r["cv_produksi"]), fontsize=8,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Rata-rata produksi (ton, skala log)")
    ax.set_ylabel("Koefisien variasi produksi")
    ax.set_title("Volume vs Stabilitas Pasokan\n(kanan-atas = kandidat prioritas Cold Storage)")
    ax.axhline(AMBANG_CV, ls="--", color="gray", lw=0.8)
    fig.tight_layout()
    fig.savefig(folder / "02_volume_vs_stabilitas.png", dpi=150)
    plt.close(fig)

    for skema, (kand, w, sens) in hasil.items():
        if kand is None:
            continue
        d = kand.head(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(d["Kecamatan"][::-1], d["skor"][::-1], color="#1565C0")
        ax.set_xlabel("Skor SAIPI Hybrid (Ahli + Entropy)")
        ax.set_title(f"Peringkat Prioritas {skema} (Model Hybrid)")
        fig.tight_layout()
        fig.savefig(folder / f"03_ranking_{skema.lower()}.png", dpi=150)
        plt.close(fig)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else cari_csv()
    folder = Path(sys.argv[2]) if len(sys.argv) > 2 else BASE / "hasil_modul1"
    path_shp = sys.argv[3] if len(sys.argv) > 3 else cari_shp()
    folder.mkdir(parents=True, exist_ok=True)

    kec = muat_data(path)
    ind = hitung_indikator(kec)

    for nama_file, label in [("data_tambahan.csv", "Modul 3"), ("data_modul2.csv", "Modul 2")]:
        f = BASE / nama_file
        if f.exists():
            tamb = pd.read_csv(f)
            baru = [c for c in tamb.columns if c != "Kecamatan" and c not in ind.columns]
            ind = ind.merge(tamb[["Kecamatan"] + baru], on="Kecamatan", how="left")
            print(f"\nMemuat {nama_file} ({label}): {', '.join(baru)}")

    ind = terapkan_syarat_kelayakan(ind)

    print("\n" + "=" * 70)
    print("INDIKATOR MODUL 1 (10 kecamatan terbesar)")
    print("=" * 70)
    cols = ["Kecamatan", "prod_mean_kw", "share_pct", "cv_produksi", "produktivitas_median", "tren_pct_per_tahun"]
    print(ind.sort_values("prod_mean_kw", ascending=False)[cols].head(10).round(2).to_string(index=False))
    print(f"\n3 kecamatan teratas = {ind.nlargest(3, 'prod_mean_kw')['share_pct'].sum():.1f}% produksi kabupaten")

    print("\nKecamatan dengan flag kualitas data:")
    fl = ind[ind["flag_kualitas"] != ""][["Kecamatan", "flag_kualitas"]]
    print(fl.to_string(index=False))

    tidak_layak_cs = ind[(ind["prod_mean_kw"] >= AMBANG_PRODUKSI_KW) & (~ind["layak_cold_storage"])]
    if len(tidak_layak_cs):
        print("\nSentra produksi yang gugur syarat kelayakan Cold Storage:")
        print(tidak_layak_cs[["Kecamatan", "alasan_kelayakan_cs"]].to_string(index=False))

    print("\n" + "=" * 70)
    print(f"SAIPI HYBRID (Alpha Ahli={ALPHA_HYBRID:.2f}, Entropy={1-ALPHA_HYBRID:.2f})")
    print("=" * 70)
    hasil = {}
    for skema, kr in KRITERIA.items():
        hasil[skema] = hitung_saipi(ind, skema, kr)
        kand, w, sens = hasil[skema]
        if kand is not None:
            print(kand[["rank", "Kecamatan", "skor"]].head(8).round(3).to_string(index=False))
            print("  Persen simulasi masuk top-5:")
            print(sens.head(8).round(1).to_string(index=False))
            print()

    print("=" * 70)
    print("UJI ROBUSTNESS: tanpa kecamatan ber-flag CV/produktivitas/lonjakan")
    print("=" * 70)
    bersih = ind[~ind["flag_kualitas"].str.contains("CV|produktivitas|lonjakan", regex=True)].copy()
    for skema, kr in KRITERIA.items():
        kand, _, _ = hitung_saipi(bersih, skema, kr)
        if kand is not None:
            print("  Top-5:", ", ".join(kand["Kecamatan"].head(5)))

    ind.round(3).to_csv(folder / "indikator_modul1.csv", index=False)
    for skema, (kand, w, sens) in hasil.items():
        if kand is not None:
            kand.round(4).to_csv(folder / f"ranking_{skema.lower()}.csv", index=False)
            sens.round(2).to_csv(folder / f"sensitivitas_{skema.lower()}.csv", index=False)
    grafik(ind, hasil, folder)
    gabung_dan_simpan_shp(ind, hasil, path_shp, folder)
    print(f"\nSelesai. File hasil ada di folder: {folder.resolve()}")


if __name__ == "__main__":
    main()