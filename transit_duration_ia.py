from pathlib import Path
import io
import itertools
import json
import sys

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

#  constants
G = 6.67430e-11          # m^3 kg^-1 s^-2 
R_SUN = 6.957e8          # m
M_SUN = 1.98847e30       # kg
R_EARTH = 6.371e6        # m
DAY, HOUR = 86400.0, 3600.0

RNG = np.random.default_rng(42)    
N_BOOT = 5000
N_SIM = 400                         # Monte Carlo simulations

OUT = Path("ia_output")
OUT.mkdir(exist_ok=True)

COLS = [
    "kepid", "kepoi_name", "kepler_name", "koi_disposition",
    "koi_period", "koi_period_err1", "koi_period_err2",
    "koi_duration", "koi_duration_err1", "koi_duration_err2",
    "koi_prad",
    "koi_srad", "koi_srad_err1", "koi_srad_err2",
    "koi_slogg", "koi_slogg_err1", "koi_slogg_err2",
]
TAP_URL = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
           + requests.utils.quote(f"select {','.join(COLS)} from cumulative")
           + "&format=csv")

# quality cuts
CUTS = (0.10, 0.20, 0.15)


# ---------------------------------------------------------------- data
def load_data():
    if len(sys.argv) > 1:
        df = pd.read_csv(sys.argv[1])
        print(f"Loaded {len(df)} KOIs from {sys.argv[1]}")
        return df[COLS], f"saved file {sys.argv[1]}"
    r = requests.get(TAP_URL, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), comment="#")
    print(f"Loaded {len(df)} KOIs from  NASA Exoplanet Archive")
    return df[COLS], "NASA Exoplanet Archive TAP"


def clean(df, cuts=CUTS):
    tT, tR, tg = cuts
    log = [("All KOIs in table", len(df))]
    d = df[df.koi_disposition == "CONFIRMED"].copy()
    log.append(("Confirmed planets", len(d)))

    d = d.dropna(subset=[c for c in COLS if c != "kepler_name"])
    log.append(("No missing values in required columns", len(d)))

    # symmetric uncertainties = mean of upper and |lower| error
    d["sP"] = (d.koi_period_err1 + d.koi_period_err2.abs()) / 2
    d["sT"] = (d.koi_duration_err1 + d.koi_duration_err2.abs()) / 2
    d["sR"] = (d.koi_srad_err1 + d.koi_srad_err2.abs()) / 2
    d["slogg"] = (d.koi_slogg_err1 + d.koi_slogg_err2.abs()) / 2

    d = d[(d.sT > 0) & (d.sR > 0) & (d.slogg > 0)]
    log.append(("Non-zero reported uncertainties", len(d)))

    d = d[(d.sT / d.koi_duration < tT) & (d.sR / d.koi_srad < tR) & (d.slogg < tg)]
    log.append((f"sigma_T/T<{tT}, sigma_R/R<{tR}, sigma_logg<{tg} dex", len(d)))

    d = d[d.koi_slogg >= 4.0]
    log.append(("Main-sequence hosts (log g >= 4.0)", len(d)))

    d = derive(d)
    d = d[d.a_over_R >= 5].copy()     # small-angle error
    log.append(("a/R* >= 5 (small-angle error < 1%)", len(d)))
    return d, log


# ---------------------------------------------------------------- physics
def derive(d):
    d = d.copy()
    P = d.koi_period * DAY                           # s
    R = d.koi_srad * R_SUN                           # m
    g = 10 ** d.koi_slogg * 1e-2                     # cgs cm/s^2 -> m/s^2
    M = g * R**2 / G                                 # kg, from g = GM/R^2
    k = d.koi_prad * R_EARTH / R                     # Rp/R*

    d["M_star_Msun"] = M / M_SUN
    d["k"] = k
    d["a_over_R"] = (G * M * P**2 / (4 * np.pi**2)) ** (1 / 3) / R

    # X = (1+k) R* (4P / (pi G M*))^(1/3)  ==  (1+k) (4 P R* / (pi g))^(1/3)
    d["X_h"] = (1 + k) * (4 * P * R / (np.pi * g)) ** (1 / 3) / HOUR
    d["T_h"] = d.koi_duration

    # fractional uncertainty in X
    frac = (1 / 3) * np.sqrt((d.sP / d.koi_period) ** 2 +
                             (d.sR / d.koi_srad) ** 2 +
                             (np.log(10) * d.slogg) ** 2)
    d["sX_h"] = d.X_h * frac
    d["r"] = d.T_h / d.X_h
    d["s_r"] = d.r * np.sqrt((d.sT / d.T_h) ** 2 + frac ** 2)
    return d


def small_angle_error(d):
    x = (1 + d.k) / d.a_over_R
    return np.arcsin(np.clip(x, 0, 1)) / x


# ---------------------------------------------------------------- fits
def ols(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    (m, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    s2 = np.sum((y - (m * x + c)) ** 2) / (len(x) - 2)
    cov = s2 * np.linalg.inv(A.T @ A)
    r2 = 1 - np.sum((y - m * x - c) ** 2) / np.sum((y - y.mean()) ** 2)
    return m, np.sqrt(cov[0, 0]), c, np.sqrt(cov[1, 1]), r2


def through_origin(x, y):
    return np.sum(x * y) / np.sum(x * x)


def bootstrap(func, *arrays, n=N_BOOT):
    size = len(arrays[0])
    out = np.empty(n)
    for i in range(n):
        idx = RNG.integers(0, size, size)
        out[i] = func(*[a[idx] for a in arrays])
    return out.std(ddof=1)


def sun_like(d):
    return d[d.koi_srad.between(0.9, 1.1) & d.M_star_Msun.between(0.8, 1.2)]


# ---------------------------------------------------------------- run
def main():
    raw, source = load_data()
    raw.to_csv(OUT / "0_raw_selected_columns.csv", index=False)
    n_koi = raw[raw.koi_disposition != "FALSE POSITIVE"].groupby("kepid").size()
    d, log = clean(raw)
    d["n_koi"] = d.kepid.map(n_koi)
    d.to_csv(OUT / "1_processed_sample.csv", index=False)

    res = {"data_source": source, "selection": log,
           "small_angle_max_error_kept": float(small_angle_error(d).max() - 1),
           "median_frac_unc": dict(P=float((d.sP / d.koi_period).median()),
                                   T=float((d.sT / d.T_h).median()),
                                   X=float((d.sX_h / d.X_h).median()))}

    # Graph 1 ln T vs ln P 
    sun = sun_like(d)
    lnP, lnT = np.log(sun.koi_period.values), np.log(sun.T_h.values)
    m1, sm1, c1, sc1, r2_1 = ols(lnP, lnT)
    sb1 = bootstrap(lambda a, b: ols(a, b)[0], lnP, lnT)
    c_fixed = lnT - lnP / 3
    mean_lnXP = float(np.mean(np.log(sun.X_h.values) - lnP / 3))
    quad = np.polyfit(lnP, lnT, 2, cov=True)
    res["graph1"] = dict(
        n=len(sun), gradient=m1, gradient_se=sm1, gradient_boot_se=sb1,
        intercept=c1, intercept_se=sc1, r2=r2_1,
        intercept_fixed_third=float(c_fixed.mean()),
        intercept_fixed_third_se=float(c_fixed.std(ddof=1) / np.sqrt(len(sun))),
        mean_ln_X_over_P13=mean_lnXP,
        intercept_predicted=mean_lnXP + np.log(2) - 1,
        quadratic_coef=float(quad[0][0]), quadratic_se=float(np.sqrt(quad[1][0, 0])))

    # exponent for other stellar sizes 
    res["gradient_by_stellar_radius"] = {}
    for lo, hi in [(0.5, 0.7), (0.7, 0.9), (0.9, 1.1), (1.1, 1.5)]:
        t = d[d.koi_srad.between(lo, hi, inclusive="left")]
        mm = t.M_star_Msun.median()
        t = t[(t.M_star_Msun > 0.8 * mm) & (t.M_star_Msun < 1.2 * mm)]
        x, y = np.log(t.koi_period.values), np.log(t.T_h.values)
        res["gradient_by_stellar_radius"][f"{lo}-{hi}"] = dict(
            n=len(t), median_mass=float(mm), gradient=ols(x, y)[0],
            gradient_boot_se=bootstrap(lambda a, b: ols(a, b)[0], x, y, n=2000))

    # Graph 2 T vs X
    X, T = d.X_h.values, d.T_h.values
    m2, sm2, c2, sc2, r2_2 = ols(X, T)
    m0 = through_origin(X, T)
    sm0 = bootstrap(through_origin, X, T)
    res["graph2"] = dict(
        n=len(d), gradient_free=m2, gradient_free_se=sm2, intercept=c2,
        intercept_se=sc2, r2=r2_2, gradient_origin=m0,
        gradient_origin_boot_se=sm0, mean_r=d.r.mean(),
        mean_r_boot_se=bootstrap(np.mean, d.r.values), median_r=d.r.median(),
        frac_r_above_1=float((d.r > 1).mean()), pi_over_4=np.pi / 4)

    # Monte Carlo isotropic orbits and each planet's own measurement noise
    R_obs, fr = d.r.values, (d.s_r / d.r).values
    u = RNG.uniform(0, 1, (N_SIM, len(d)))
    r_sim = np.sqrt(1 - u**2) * (1 + fr * RNG.normal(size=u.shape))
    flat = r_sim.ravel()
    count = lambda arr, lo, hi: ((arr >= lo) & (arr < hi)).sum()
    per_sim = lambda lo, hi: np.mean([count(s, lo, hi) for s in r_sim])
    ks = stats.ks_2samp(R_obs, flat)          # data vs pooled simulation
    res["monte_carlo_isotropic"] = dict(
        mean_r_sim=float(flat.mean()), frac_sim_above_1=float((flat > 1).mean()),
        ks_statistic=float(ks.statistic), ks_p_value=float(ks.pvalue))

    edges = np.linspace(0, 1.3, 27)
    counts = np.histogram(R_obs, bins=edges)[0]
    sim_counts = np.histogram(flat, bins=edges)[0] / N_SIM
    pd.DataFrame({"r_low": edges[:-1], "r_high": edges[1:], "observed": counts,
                  "expected_isotropic_with_noise": sim_counts}).to_csv(
        OUT / "3_r_histogram.csv", index=False)

    # Graph 4 completeness (observed / predicted) by impact parameter
    # r = sqrt(1-u^2)
    ub = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
    comp = []
    for lo, hi in zip(ub[:-1], ub[1:]):
        r_lo, r_hi = np.sqrt(1 - hi**2), np.sqrt(1 - lo**2)
        o, p = count(R_obs, r_lo, r_hi), per_sim(r_lo, r_hi)
        comp.append(dict(u_low=lo, u_high=hi, r_low=r_lo, r_high=r_hi,
                         observed=int(o), predicted=p, ratio=o / p,
                         ratio_err=np.sqrt(o) / p))
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / "4_completeness.csv", index=False)
    res["completeness"] = comp.to_dict("records")
    res["tails"] = {name: dict(observed=int(count(R_obs, lo, hi)),
                               predicted=per_sim(lo, hi))
                    for name, lo, hi in [("r<0.5", -np.inf, 0.5), ("r>=1", 1, np.inf),
                                         ("r>1.1", 1.1, np.inf), ("r>1.3", 1.3, np.inf)]}

    # how much of the gap to pi/4 each explains
    N = len(R_obs)
    miss = per_sim(-np.inf, 0.6) - (R_obs < 0.6).sum()      # u > 0.8 
    extra = (R_obs >= 1).sum() - per_sim(1, np.inf)
    mean_grazing = flat[flat < 0.6].mean()
    m_restored = (R_obs.sum() + miss * mean_grazing) / (N + miss)
    m_both = ((R_obs[R_obs < 1].sum() + per_sim(1, np.inf) * flat[flat >= 1].mean()
               + miss * mean_grazing) / (N + miss - extra))
    res["decomposition"] = dict(
        missing_grazing=float(miss), extra_long=float(extra),
        mean_observed=float(R_obs.mean()),
        mean_grazing_restored=float(m_restored),
        mean_grazing_restored_and_tail_removed=float(m_both),
        mean_isotropic=float(flat.mean()))

    # checks
    res["mean_r_by_logg"] = {}
    for lo, hi in [(4.0, 4.3), (4.3, 4.5), (4.5, 5.3)]:
        s = d.r[(d.koi_slogg >= lo) & (d.koi_slogg < hi)]
        res["mean_r_by_logg"][f"{lo}-{hi}"] = dict(n=len(s), mean_r=s.mean(), se=s.sem())
    res["multiplicity"] = {
        name: dict(n=len(s), mean_r=s.r.mean(), se=s.r.sem(),
                   frac_r_above_1p1=float((s.r > 1.1).mean()))
        for name, s in [("single", d[d.n_koi == 1]), ("multi", d[d.n_koi > 1])]}
    small = d[d.k < 0.05]
    res["small_planets_k_lt_0.05"] = dict(
        n=len(small), mean_r=small.r.mean(),
        mean_r_boot_se=bootstrap(np.mean, small.r.values))
    terc = pd.qcut(d.koi_period, 3, labels=["short", "middle", "long"])
    res["mean_r_by_period_tercile"] = {
        t: dict(n=int((terc == t).sum()), mean_r=float(d.r[terc == t].mean()),
                P_range_days=[float(d.koi_period[terc == t].min()),
                              float(d.koi_period[terc == t].max())])
        for t in ["short", "middle", "long"]}

    # sensitivity of graph 2 gradient to the quality cuts
    grid = []
    for cuts in itertools.product([0.05, 0.10, 0.20], [0.10, 0.20, 0.30],
                                  [0.10, 0.15, 0.20]):
        s, _ = clean(raw, cuts)
        grid.append(dict(sT=cuts[0], sR=cuts[1], slogg=cuts[2], n=len(s),
                         gradient=through_origin(s.X_h.values, s.T_h.values),
                         mean_r=s.r.mean()))
    grid = pd.DataFrame(grid)
    grid.to_csv(OUT / "5_cut_sensitivity.csv", index=False)
    res["cut_sensitivity"] = dict(
        n_combinations=len(grid), n_range=[int(grid.n.min()), int(grid.n.max())],
        gradient_min=grid.gradient.min(), gradient_max=grid.gradient.max(),
        gradient_sd=grid.gradient.std(ddof=1))

    # example rows for the raw-data table
    ex = d.sort_values("koi_period").iloc[np.linspace(0, len(d) - 1, 8).astype(int)]
    ex[["kepoi_name", "kepler_name", "koi_period", "sP", "koi_duration", "sT",
        "koi_srad", "sR", "koi_slogg", "slogg", "koi_prad", "X_h", "sX_h", "r"]
       ].to_csv(OUT / "2_example_rows.csv", index=False)

    make_figures(d, sun, lnP, lnT, m1, sm1, c1, m0, sm0, edges, counts,
                 sim_counts, comp)
    with open(OUT / "results.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))


# ---------------------------------------------------------------- figures
def make_figures(d, sun, lnP, lnT, m1, sm1, c1, m0, sm0, edges, counts,
                 sim_counts, comp):
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150})
    X, T = d.X_h.values, d.T_h.values

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(lnP, lnT, yerr=(sun.sT / sun.T_h).values, fmt="o", ms=2.5,
                alpha=0.5, color="#4C72B0", ecolor="#9fb3d1", elinewidth=0.6,
                label="Sun-like hosts (error bar = ΔT₁₄/T₁₄)")
    xs = np.linspace(lnP.min(), lnP.max(), 50)
    ax.plot(xs, m1 * xs + c1, color="#DD8452", lw=2,
            label=f"best fit: gradient = {m1:.3f} ± {sm1:.3f}")
    ax.plot(xs, xs / 3 + (c1 + (m1 - 1 / 3) * lnP.mean()), "k--", lw=1.2,
            label="gradient 1/3 (theory)")
    ax.set_xlabel("ln(P / days)")
    ax.set_ylabel("ln(T₁₄ / hours)")
    ax.set_title(f"Graph 1: ln T₁₄ against ln P (n = {len(sun)})")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "graph1_lnT_lnP.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(X, T, xerr=d.sX_h, yerr=d.sT, fmt="o", ms=2.5, alpha=0.35,
                color="#4C72B0", ecolor="#9fb3d1", elinewidth=0.6,
                label="Kepler planets")
    xs = np.linspace(0, X.max() * 1.02, 50)
    ax.plot(xs, xs, "k-", lw=1.2, label="gradient 1 (central transit, b = 0)")
    ax.plot(xs, np.pi / 4 * xs, "k--", lw=1.2, label="gradient π/4 (random orientation)")
    ax.plot(xs, m0 * xs, color="#DD8452", lw=2,
            label=f"best fit through origin: {m0:.3f} ± {sm0:.3f}")
    ax.set_xlabel("X = (1+k) R* (4P / πGM*)^(1/3)  /  hours")
    ax.set_ylabel("T₁₄  /  hours")
    ax.set_title(f"Graph 2: transit duration against X (n = {len(d)})")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "graph2_T_vs_X.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    lo, hi = edges[:-1], edges[1:]
    ax.bar(lo, counts, width=hi - lo, align="edge", color="#4C72B0", alpha=0.7,
           edgecolor="white", label="observed")
    ax.step(np.append(lo, hi[-1]), np.append(sim_counts, sim_counts[-1]),
            where="post", color="k", lw=1.5,
            label="predicted: isotropic orbits + measurement noise")
    ax.axvline(np.pi / 4, color="k", ls="--", lw=1, label="π/4 (predicted mean)")
    ax.axvline(d.r.mean(), color="#DD8452", lw=2,
               label=f"observed mean {d.r.mean():.3f}")
    ax.set_xlabel("r = T₁₄ / X")
    ax.set_ylabel("number of planets")
    ax.set_title("Graph 3: distribution of r")
    ax.set_ylim(0, max(counts.max(), sim_counts.max()) * 1.35)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "graph3_r_hist.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    mid = (comp.u_low + comp.u_high) / 2
    ax.errorbar(mid, comp.ratio, yerr=comp.ratio_err,
                xerr=(comp.u_high - comp.u_low) / 2, fmt="o", color="#4C72B0",
                capsize=3, label="observed / predicted")
    ax.axhline(1, color="k", ls="--", lw=1, label="no selection bias")
    ax.set_xlim(0.15, 1.02)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("impact parameter  b / (1 + k)")
    ax.set_ylabel("fraction of predicted planets observed")
    ax.set_title("Graph 4: detection completeness against impact parameter")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "graph4_completeness.png"); plt.close(fig)


if __name__ == "__main__":
    main()
