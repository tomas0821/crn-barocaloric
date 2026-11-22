"""Step 3: Monte-Carlo uncertainty bands for the CrN barocaloric response.

The parameter sets stay PAIRED BY SOURCE: bands are computed separately for the Chaussy
and Browne sets, never pooled. For each sample the anomaly shape is the digitized Chaussy C_p excess
(baseline B1), rescaled to the sampled ΔS_t. It is also broadened by a Gaussian T_N distribution
(stoichiometric spread) before the reversible response is computed with code/baro.py.

Sampled inputs (uniform priors; ranges and justification are in results/step3_montecarlo.txt).
The wide slope prior (adopted after the step-4 validation showed that the T_t(P) slope dominates the error)
adds slope curvature and widens the initial slope; the narrow prior is kept for comparison. The headline
bands use the wide prior.
Narrow-prior inputs:
  Chaussy: ΔS_t 6.30-7.00 J/mol/K (digitized-baseline spread up to the calorimetric 29 J/g);
           a 8.5-9.5 K/GPa; h 2-5 K (classic samples; own h unknown); σ_TN 0-1.5 K; α_V ±10 %
  Browne:  ΔS_t 8.20-8.90 J/mol/K (spin-only 8.58, Clapeyron 8.65; ±4 %);
           a 7.35-7.47 K/GPa; h 1.7-2.0 K (Browne 1971 Fig. 1, own specimen); σ_TN 0-1.5 K; α_V ±10 %
A regression check is printed first: the deterministic Chaussy h = 4 K case must reproduce step 2.
Light: N = 1500 per set on coarse grids. Run it capped (code/README.md).
"""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baro import Baro, first_crossing

ROOT = Path(__file__).resolve().parents[1]
M = 66.003e-3                       # kg/mol
V_kg = 6.02214076e23 * 17.84e-30 / M  # m^3/kg
A0, A1 = 1.95e-5, 4.39e-9
CAL = 4.184
RNG = np.random.default_rng(20261121)
N = 1500
PG = np.round(np.arange(0.0, 2.5001, 0.02), 3)
TG = np.round(np.arange(250.0, 340.0001, 0.1), 3)
P_DT = (0.5, 0.6, 0.75, 1.0, 1.5, 2.0)


def chaussy_shape():
    g = np.loadtxt(ROOT / "data/digitized/chaussy1971_cp_grid.csv", delimiter=",")
    T_C, C_cal = g[:, 0], g[:, 2]
    c1, c2 = np.interp(-30, T_C, C_cal), np.interp(40, T_C, C_cal)
    base = c1 + (c2 - c1) * (T_C + 30) / 70
    win = (T_C >= -30) & (T_C <= 40)
    Cex = np.where(win, np.clip(C_cal - base, 0, None), 0.0) * CAL / M      # J/kg/K
    Cbg = np.where(win, base, C_cal) * CAL / M
    TK = T_C + 273.15
    # Uniform 0.1 K grid on 200-400 K (extend the background linearly, the excess with zeros).
    T = np.round(np.arange(200.0, 400.0001, 0.1), 3)
    pbg = np.polyfit(TK[TK > TK.max() - 15], Cbg[TK > TK.max() - 15], 1)
    Cbg_u = np.where(T <= TK.max(), np.interp(T, TK, Cbg), np.polyval(pbg, T))
    Cex_u = np.interp(T, TK, Cex, left=0.0, right=0.0)
    return T, Cex_u, Cbg_u, TK[np.argmax(C_cal)]


def make(T, Cex, Cbg, Tpk_shift, dS_molK, a, h, sigma, aV_scale, b=0.0, dalpha=0.0):
    """dalpha: extra volumetric thermal expansion of the high-T (PM) phase over the AFM phase [1/K]. By the
    Maxwell relation it makes ΔS_t decline along the boundary: dΔS_t/dP = -v·Δα (R3 sensitivity)."""
    Cx = np.interp(T - Tpk_shift, T, Cex, left=0.0, right=0.0)
    c = V_kg * dalpha * 1e9 / (dS_molK / M)       # fractional decline of ΔS_t per GPa
    return Baro(T, Cx, Cbg, a=a, h=h, v=V_kg, alphaV=lambda TT: aV_scale * (A0 + A1 * TT),
                dS_target=dS_molK / M, sigma_broad=sigma, b=b, c=c)


def sample_slope(a_meas, prior):
    """narrow: a in a narrow box, no curvature. wide (after step 4): initial slope a0 within ±10 %
    of the measured value, curvature b in [-0.8, +1.2] K/GPa^2, rejecting samples whose 0-1.8 GPa average
    slope a0 + 1.8 b falls outside the measured envelope [6.5, 11] K/GPa."""
    if prior["kind"] == "narrow":
        return RNG.uniform(*prior["a"]), 0.0
    while True:
        a0 = RNG.uniform(0.9 * a_meas, 1.1 * a_meas); b = RNG.uniform(-0.8, 1.2)
        if 6.5 <= a0 + 1.8 * b <= 11.0:
            return a0, b


def main():
    T, Cex, Cbg, Tpk = chaussy_shape()
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    # Regression check against step 2 (Chaussy, h = 4 K, no broadening).
    b = make(T, Cex, Cbg, 0.0, 29 * 66.003 / 281.65, 9.0, 4.0, 0.0, 1.0)
    f, dSr, dS1, dT = b.summary(TG, PG, P_DT)
    out("Regression vs step 2 (Chaussy, h=4): onset/P50 = "
        f"{first_crossing(PG, f, 0.01):.2f}/{first_crossing(PG, f, 0.5):.2f} GPa (step 2: 0.46/0.94); "
        f"|dS_rev|@1 GPa = {dSr[np.argmin(abs(PG-1))]:.0f} J/kg/K (59); dT_rev@1 GPa = {dT[1.0]:.1f} K (4.5)")
    out("")

    base_sets = {
        "Chaussy": dict(dS=(6.30, 7.00), a_meas=9.0, a=(8.5, 9.5), shift=0.0, h=(2.0, 5.0)),   # h: classic-sample range (1.9-5 K); own h unknown
        "Browne":  dict(dS=(8.20, 8.90), a_meas=7.41, a=(7.35, 7.47), shift=287.0 - Tpk, h=(1.7, 2.0)),   # h: Browne 1971 Fig. 1 (1.86 K at 0, 1.82 K at 4 kbar)
    }
    sets = {}
    for nm, bs in base_sets.items():
        sets[f"{nm}, wide slope prior (with curvature)"] = dict(bs, prior=dict(kind="wide"), c=0.0)
        sets[f"{nm}, narrow slope prior"] = dict(bs, prior=dict(kind="narrow", a=bs["a"]), c=0.0)
    for nm, bs in base_sets.items():
        for da in (0.5e-5, 1.0e-5):
            sets[f"{nm}, wide prior + PM-phase Δα = {da:.1e} 1/K (sensitivity)"] = dict(bs, prior=dict(kind="wide"), c=da)
    # Hysteresis sensitivity for the Chaussy set (own h unknown): the narrower 3-5 K range without Browne 1971.
    sets["Chaussy, wide slope prior, h 3-5 K (sensitivity)"] = dict(base_sets["Chaussy"], h=(3.0, 5.0), prior=dict(kind="wide"), c=0.0)
    bands = {}
    for name, s in sets.items():
        F, DSR, DTR, ON, P50 = [], [], [], [], []
        nS = N if s["c"] == 0.0 else N // 3   # s["c"] holds Δα [1/K] for the sensitivity sets
        for k in range(nS):
            dS = RNG.uniform(*s["dS"]); a, bcurv = sample_slope(s["a_meas"], s["prior"]); h = RNG.uniform(*s["h"])
            sig = RNG.uniform(0.0, 1.5); aV = RNG.uniform(0.9, 1.1)
            bb = make(T, Cex, Cbg, s["shift"], dS, a, h, sig, aV, b=bcurv, dalpha=s["c"])
            f, dSr, dS1, dT = bb.summary(TG, PG, P_DT)
            F.append(f); DSR.append(dSr); DTR.append([dT[p] for p in P_DT])
            ON.append(first_crossing(PG, f, 0.01)); P50.append(first_crossing(PG, f, 0.5))
        F, DSR, DTR = map(np.array, (F, DSR, DTR)); ON, P50 = np.array(ON), np.array(P50)
        q = lambda x, ax=0: np.nanpercentile(x, [2.5, 16, 50, 84, 97.5], axis=ax)
        bands[name] = dict(F=q(F), DSR=q(DSR), DTR=q(DTR), ON=q(ON), P50=q(P50))
        pr = (f"a {s['prior']['a']} K/GPa, no curvature" if s["prior"]["kind"] == "narrow"
              else f"a0 = {s['a_meas']} ±10 %, b in [-0.8, 1.2] K/GPa², avg slope 0-1.8 GPa in [6.5, 11]")
        out(f"== {name} (N = {nS}; ΔS_t {s['dS']} J/mol/K, {pr}, h {s['h'][0]:g}-{s['h'][1]:g} K, σ_TN 0-1.5 K, α_V ±10 %, Δα_PM = {s['c']:.1e} 1/K -> ΔS_t decline {100*V_kg*s['c']*1e9/(np.mean(s['dS'])/M)/(1):.1f} %/GPa)")
        out(f"   onset  [2.5,16,50,84,97.5]% = " + ", ".join(f"{x:.2f}" for x in bands[name]['ON']) + " GPa")
        out(f"   P50    [2.5,16,50,84,97.5]% = " + ", ".join(f"{x:.2f}" for x in bands[name]['P50']) + " GPa")
        for p in (0.25, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0):
            i = np.argmin(abs(PG - p)); qq = bands[name]["DSR"][:, i]
            j = P_DT.index(p) if p in P_DT else None
            dtt = bands[name]["DTR"][:, j] if j is not None else None
            out(f"   P = {p:4.2f} GPa: |ΔS_rev| median {qq[2]:5.1f} J/kg/K (68 %: {qq[1]:.1f}-{qq[3]:.1f}; 95 %: {qq[0]:.1f}-{qq[4]:.1f})"
                + (f"; ΔT_rev median {dtt[2]:.1f} K (68 %: {dtt[1]:.1f}-{dtt[3]:.1f})" if dtt is not None else ""))
        out("")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/step3_montecarlo.txt").write_text("\n".join(lines) + "\n")

    import figstyle as fs
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(fs.FULL, 2.5))
    for nm, col in (("Chaussy", "tab:blue"), ("Browne", "tab:orange")):
        name = f"{nm}, wide slope prior (with curvature)"
        B = bands[name]["DSR"]
        axs[0].fill_between(PG, B[0], B[4], color=col, alpha=0.15, lw=0)
        axs[0].fill_between(PG, B[1], B[3], color=col, alpha=0.35, lw=0)
        axs[0].plot(PG, B[2], color=col, label=f"{nm} set")
        axs[0].plot(PG, bands[f"{nm}, narrow slope prior"]["DSR"][2], ":", color=col, lw=0.9)
        D = bands[name]["DTR"]; Pd = np.array(P_DT)
        axs[1].errorbar(Pd + (0.01 if nm == "Browne" else -0.01), D[2], yerr=[D[2] - D[1], D[3] - D[2]],
                        fmt="o", ms=3, color=col, capsize=2, lw=0.9, label=f"{nm} set")
    for k, ax in enumerate(axs):
        ax.axvline(0.6, color="gray", ls=":", lw=0.8); ax.set_xlabel("$P$ (GPa)"); fs.panel(ax, "ab"[k])
    axs[0].set_ylabel(r"peak reversible $|\Delta S|$ (J kg$^{-1}$ K$^{-1}$)"); axs[1].set_ylabel(r"$\Delta T_\mathrm{rev}$ (K)")
    axs[0].legend(loc="upper left"); axs[1].legend(loc="upper left")
    fig.tight_layout(); fs.save(fig, ROOT / "figures/step3_mc_bands"); plt.close(fig)


if __name__ == "__main__":
    main()
