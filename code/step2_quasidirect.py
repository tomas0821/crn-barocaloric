"""Step 2: quasi-direct barocaloric core for CrN.

Method: the standard quasi-direct construction of the barocaloric literature (e.g. Lloveras et al.,
Nat. Commun. 10, 1803 (2019); Stern-Taulats et al., PRB 89, 214105 (2014)).
- Entropy from the digitized C_p (Chaussy 1971 Fig. 3, measured on WARMING):
      S_h(T, 0) = S_bg(T) + S_ex(T),
  where S_ex is the cumulative excess entropy of the anomaly over baseline B1 (0 below the anomaly,
  ΔS_tr above it) and S_bg is the background (lattice + electronic) entropy.
- The cooling branch is the warming branch shifted down by the hysteresis h.
- Pressure shifts the anomaly by a·P (dT_N/dP = a > 0, conventional barocaloric) and adds the lattice
  term −V_m·α_V(T)·P (Maxwell relation):
      S_h(T, P) = S_bg(T) − V_m α_V P + S_ex(T − aP)
      S_c(T, P) = S_bg(T) − V_m α_V P + S_ex(T − aP + h)
- REVERSIBLE (cyclic) response pairs the heating branch at 0 with the cooling branch at P:
      ΔS_rev(T, P) = min(0, S_c(T, P) − S_h(T, 0))   [plus the always-reversible lattice term]
      ΔT_rev(T, P): solve S_c(T + ΔT, P) = S_h(T, 0)
  FIRST-APPLICATION / single-branch (an upper bound, not cyclic): S_h(T, P) − S_h(T, 0).
- Reversible fraction f(P) = max_T |ΔS_rev,transition| / ΔS_tr. Onset, P50 and P90 are the pressures
  where f first exceeds 0.01, 0.5 and 0.9.

Parameter sets are PAIRED BY SOURCE. They are never mixed:
  "Chaussy": ΔS_tr = 6.80 J/mol/K (29 J/g calorimetric), a = 9.0 K/GPa, T_peak as digitized (281.95 K)
  "Browne":  ΔS_tr = 8.58 J/mol/K (spin-only, σ_c = 0.93), a = 7.41 K/GPa, T_peak shifted to 287 K
  "slope-11": the Chaussy entropy with a = 11 K/GPa (sensitivity bound from the T_N ≈ 240 K family)
The anomaly SHAPE (width) is Chaussy's for every set; only its integral is rescaled.

Light by design (grids of about 10^6 floats); run it capped, as described in code/README.md.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAL = 4.184
NA = 6.02214076e23
VM = NA * 17.84e-30                    # m^3/mol
A0, A1 = 1.95e-5, 4.39e-9              # alpha_V = A0 + A1*T (Yan 2020)
T0C = 273.15

SETS = {
    "Chaussy":  dict(dS=29 * 66.003 / 281.65, a=9.0,  Tpk=None),
    "Browne":   dict(dS=0.130 * 66.003,       a=7.41, Tpk=287.0),
    "slope-11": dict(dS=29 * 66.003 / 281.65, a=11.0, Tpk=None),
}
H_LIST = [2.0, 3.0, 4.0, 5.0, 10.0]
P = np.round(np.arange(0.0, 3.0001, 0.005), 4)         # GPa
TG = np.round(np.arange(220.0, 360.0001, 0.02), 4)      # K


def load_anomaly():
    g = np.loadtxt(ROOT / "data/digitized/chaussy1971_cp_grid.csv", delimiter=",")
    T_C, C_cal = g[:, 0], g[:, 2]
    TK = T_C + T0C
    c1, c2 = np.interp(-30, T_C, C_cal), np.interp(40, T_C, C_cal)
    base = c1 + (c2 - c1) * (T_C + 30) / 70                 # baseline B1 (step 1b)
    win = (T_C >= -30) & (T_C <= 40)
    Cex = np.where(win, np.clip(C_cal - base, 0, None), 0.0) * CAL   # J/mol/K
    Cbg = np.where(win, base, C_cal) * CAL
    Tpk = TK[np.argmax(C_cal)]
    return TK, Cex, Cbg, Tpk


def cumint(y, x):
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def build(params):
    TK, Cex, Cbg, Tpk_dig = load_anomaly()
    Sex = cumint(Cex / TK, TK)                                # J/mol/K, 0 -> ΔS_dig
    dS_dig = Sex[-1]
    scale = params["dS"] / dS_dig
    shiftT = (params["Tpk"] - Tpk_dig) if params["Tpk"] else 0.0
    # Background: extend linearly beyond the digitized range (needed for shifted arguments).
    pbg = np.polyfit(TK[TK > TK.max() - 15], Cbg[TK > TK.max() - 15], 1)
    TKe = np.r_[TK, np.arange(TK.max() + 0.1, 400, 0.1)]
    Cbge = np.r_[Cbg, np.polyval(pbg, TKe[len(TK):])]
    Sbg = cumint(Cbge / TKe, TKe)
    S_ex = lambda T: scale * np.interp(T - shiftT, TK, Sex, left=0.0, right=dS_dig)
    S_bg = lambda T: np.interp(T, TKe, Sbg)
    Cp_bg_at = lambda T: np.interp(T, TKe, Cbge)
    return S_ex, S_bg, Cp_bg_at, dict(scale=scale, dS_dig=dS_dig, Tpk=Tpk_dig + shiftT)


def analyse(name, params, h):
    S_ex, S_bg, _, info = build(params)
    a = params["a"]
    alphaV = A0 + A1 * TG
    Sh0 = S_bg(TG) + S_ex(TG)
    dS_tr = params["dS"]
    f = np.zeros(P.size); dSrev_max = np.zeros(P.size); dTrev_max = np.zeros(P.size); dS1_max = np.zeros(P.size)
    dSrev_curves, dTrev_curves = {}, {}
    for i, p in enumerate(P):
        lat = -VM * alphaV * p * 1e9                       # J/mol/K (always reversible)
        Sc = S_bg(TG) + lat + S_ex(TG - a * p + h)
        Shp = S_bg(TG) + lat + S_ex(TG - a * p)
        trans = np.minimum(0.0, (Sc - lat) - Sh0)          # transition part only
        f[i] = -trans.min() / dS_tr
        dSrev = trans + lat
        dSrev_max[i] = -dSrev.min()
        dS1_max[i] = -(Shp - Sh0).min()
        # ΔT_rev: adiabatic compression from the heating-branch state at (T, 0). The AFM fraction can only
        # GROW on compression (x_eff = max(x_h(T,0), x_c(T',P))), so the transition entropy after
        # compression is min(S_ex(T), S_ex(T' - aP + h)). Solve
        #     S_bg(T') + lat(T') + min(S_ex(T), S_ex(T'-aP+h)) = S_bg(T) + S_ex(T)
        # for T' by vectorized bisection (the left side increases monotonically with T').
        if i % 4 == 0 or np.isclose(p, [0.25, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0]).any():
            Sex0 = S_ex(TG); target = Sh0
            lo, hi = TG - 2.0, TG + 60.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                g = S_bg(mid) - VM * (A0 + A1 * mid) * p * 1e9 + np.minimum(Sex0, S_ex(mid - a * p + h)) - target
                hi = np.where(g > 0, mid, hi); lo = np.where(g > 0, lo, mid)
            dT = 0.5 * (lo + hi) - TG
            ok = (TG > 240) & (TG < 330)
            dTrev_max[i] = dT[ok].max()
        else:
            dTrev_max[i] = np.nan
        if np.isclose(p, [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]).any():
            dSrev_curves[p] = dSrev.copy(); dTrev_curves[p] = np.where(ok, dT, np.nan)
    good = np.isfinite(dTrev_max)
    dTrev_max = np.interp(P, P[good], dTrev_max[good])
    def first(th):
        j = np.where(f >= th)[0]
        return P[j[0]] if j.size else np.nan
    return dict(f=f, dSrev_max=dSrev_max, dTrev_max=dTrev_max, dS1_max=dS1_max, onset=first(0.01),
                P50=first(0.5), P90=first(0.9), curves=(dSrev_curves, dTrev_curves), info=info)


def main():
    lines = []
    def out(s=""):
        print(s); lines.append(s)
    out("Step 2 — quasi-direct barocaloric core for CrN (reversible = heating@0 vs cooling@P)")
    out(f"V_m = {VM*1e6:.3f} cm^3/mol; alpha_V(280 K) = {A0+A1*280:.3e} 1/K -> lattice term {VM*(A0+A1*280)*1e9:.3f} J/mol/K per GPa")
    W_eff = 7.3; W_fwhm = 5.3
    out(f"{'set':9s} {'h(K)':>5s} {'a':>5s} {'onset':>6s} {'P50':>6s} {'P90':>6s} | closed form (h+fW)/a, W={W_fwhm}-{W_eff} K: P50, P90 | "
        f"{'|dS_rev| @0.5/0.6/1/2 GPa (J/kg/K)':>34s} | {'dT_rev @0.5/0.6/1/2 GPa (K)':>28s} | aP-h @1 GPa")
    results = {}
    for name, prm in SETS.items():
        for h in H_LIST:
            r = analyse(name, prm, h); results[(name, h)] = r
            a = prm["a"]
            cf50 = f"{(h+0.5*W_fwhm)/a:.2f}-{(h+0.5*W_eff)/a:.2f}"; cf90 = f"{(h+0.9*W_fwhm)/a:.2f}-{(h+0.9*W_eff)/a:.2f}"
            pick = lambda arr, p: arr[np.argmin(abs(P - p))]
            ds = "/".join(f"{pick(r['dSrev_max'], p)/0.066003:.0f}" for p in (0.5, 0.6, 1.0, 2.0))
            dt = "/".join(f"{pick(r['dTrev_max'], p):.1f}" for p in (0.5, 0.6, 1.0, 2.0))
            out(f"{name:9s} {h:5.0f} {a:5.2f} {r['onset']:6.2f} {r['P50']:6.2f} {r['P90']:6.2f} | {cf50:>11s} {cf90:>11s}"
                f"{'':27s} | {ds:>34s} | {dt:>26s} | {a*1-h:5.1f}")
    out("")
    out("Columns: onset/P50/P90 in GPa from the full curves; the closed form brackets use W = FWHM..effective width.")
    out("|dS_rev| includes the lattice term; dT_rev is the max over T of the reversible adiabatic temperature change.")
    i0 = results[("Chaussy", 3.0)]["info"]
    out(f"Chaussy set: digitized ΔS = {i0['dS_dig']:.2f} J/mol/K rescaled x{i0['scale']:.3f} to 6.80; T_peak = {i0['Tpk']:.2f} K")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/step2_quasidirect.txt").write_text("\n".join(lines) + "\n")

    import figstyle as fs
    import matplotlib.pyplot as plt
    # Fig A (repository only): reversible fraction vs P for each set and h.
    fig, axs = plt.subplots(1, 3, figsize=(fs.FULL, 2.3), sharey=True)
    for k, (ax, name) in enumerate(zip(axs, SETS)):
        for h, col in zip(H_LIST, plt.cm.viridis(np.linspace(0, 0.9, len(H_LIST)))):
            ax.plot(P, results[(name, h)]["f"], color=col, label=f"$h$ = {h:g} K")
        ax.axvline(0.6, color="gray", ls=":", lw=0.8)
        ax.text(0.97, 0.05, f"{name}, $a$ = {SETS[name]['a']} K GPa$^{{-1}}$", transform=ax.transAxes, ha="right", fontsize=7)
        ax.set_xlabel("$P$ (GPa)"); ax.set_xlim(0, 3); fs.panel(ax, "abc"[k], x=-0.12)
    axs[0].set_ylabel(r"reversible fraction of $\Delta S_\mathrm{t}$"); axs[0].legend(loc="upper left", fontsize=6.5)
    fig.tight_layout(); fs.save(fig, ROOT / "figures/step2_reversible_fraction"); plt.close(fig)
    # Fig B (manuscript): ΔS_rev(T) and ΔT_rev(T) for central inputs (Chaussy set, h = 4 K).
    r = results[("Chaussy", 4.0)]; dSc, dTc = r["curves"]
    fig, axs = plt.subplots(1, 2, figsize=(fs.FULL, 2.4))
    for p, col in zip(sorted(dSc), plt.cm.plasma(np.linspace(0, 0.85, len(dSc)))):
        axs[0].plot(TG, dSc[p] / 0.066003, color=col, label=f"{p:.2f} GPa")
        axs[1].plot(TG, dTc[p], color=col, label=f"{p:.2f} GPa")
    axs[0].set_xlim(270, 310); axs[1].set_xlim(270, 310)
    axs[0].set_ylabel(r"$\Delta S_\mathrm{rev}$ (J kg$^{-1}$ K$^{-1}$)"); axs[1].set_ylabel(r"$\Delta T_\mathrm{rev}$ (K)")
    for k, ax in enumerate(axs):
        ax.set_xlabel("$T$ (K)"); fs.panel(ax, "ab"[k])
    axs[0].legend(loc="lower left", fontsize=6.5)
    fig.tight_layout(); fs.save(fig, ROOT / "figures/step2_dS_dT_chaussy_h4"); plt.close(fig)


if __name__ == "__main__":
    main()
