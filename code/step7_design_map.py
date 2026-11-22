"""Step 7: the cross-material design-rule map.

Source table: data/benchmarks/barocaloric_benchmarks.csv, compiled and verified 2026-11-21. Every value
carries a DOI, an anchor, and a flag: stated / figure-read / derived / compiled. The NUMERIC TABLE below is
that CSV with each multi-valued cell turned into central (low-high) values. Every choice is commented;
nothing is filled in from memory.

Physics: Clausius–Clapeyron, |dT/dP| = ΔV/ΔS, so a volumetric transition entropy satisfies
    ΔS_vol = (ΔV/V) / |dT/dP|.
On a log-log plot of ΔS_vol against |dT/dP|, lines of constant ΔV/V are universal. A material whose
entropy is large but whose volume change is small (CrN) sits far below the line its entropy would need
for an efficient barocaloric effect. The CrN uniaxial point uses the shear transformation strain
(1.25-1.6 %) in place of ΔV/V: the same Clausius–Clapeyron form for the conjugate stress.
Panel (b): the reversibility-onset estimate h/|dT/dP| against ΔS_t. A single consistent definition is
used for all materials, so it is a screening metric, not a measured onset.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# name: (dS_J_kgK c, lo, hi), (|a| K/GPa c, lo, hi), (h_K c, lo, hi or None), rho_g_cm3, sign of a, note
B = {
    "Mn3GaN":                 ((22.3, 22.3, 22.3), (65, 65, 65), None, 7.0, -1, "h n.v."),
    "Mn3NiN":                 ((43, 43, 43), (13.5, 13, 14), (8, 8, 8), 6.75, -1, "h = 8 K primary (Boldrin 2021 table lists 2 K from another sample)"),
    "Fe49Rh51":               ((12.5, 11.5, 13.5), (59, 54, 64), (10, 9.6, 10), 9.9, +1, "a: heating 54 / cooling 64"),
    "MnCoGeB0.03":            ((47, 47, 47), (100, 100, 100), (18, 13, 18), 8.0, -1, "h: 13-18 K across samples"),
    "Ni-Mn-In":               ((27, 24, 30), (18, 18, 18), None, 8.20, +1, "ΔS ±3; h n.v."),
    "NPG":                    ((372, 361, 383), (103, 93, 113), (20, 15, 25), 1.064, +1, "heat/cool ΔS and a; h 15-25 K (sources disagree)"),
    "(MnNiSi)0.62(FeCoGe)0.38": ((62, 62, 62), (75, 75, 75), (15, 15, 15), 7.3, -1, "x = 0.38 (Samanta 2018)"),
    "NdCu3Fe4O12":            ((84.2, 84.2, 84.2), (36.1, 36.1, 36.1), None, 6.16, -1, "h n.v."),
    "Fe3(bntrz)6(tcnset)6 (SCO)": ((80, 80, 80), (250, 250, 250), (2, 2, 2), 1.43, +1, "spin crossover"),
    "(NH4)2SO4":              ((65, 65, 65), (51, 45, 57), (6, 3, 6), 1.8, -1, "first-order step only; h 3-6 K (sources disagree)"),
}
# Ni0.85Fe0.15S excluded: ΔS_t n.v. (only the ESI was available).
RHO_CRN = 4 * 66.003 / (6.02214076e23 * (4.148e-8) ** 3)       # g/cm^3 (x-ray, a = 4.148 Å)
CRN = {
    "CrN hydrostatic (Chaussy set)":  ((103, 95, 106), (9.0, 8.1, 9.9), (4, 3, 5), RHO_CRN, +1, "calorimetric ΔS; a ±10 %; classic bulk h 3-5 K"),
    "CrN hydrostatic (Browne set)":   ((130, 124, 135), (7.41, 6.7, 8.15), (4, 3, 5), RHO_CRN, +1, "spin-only ΔS (not calorimetric)"),
    "CrN uniaxial <110> (CC estimate)": ((103, 95, 106), (22.5, 19.7, 25.3), (4, 3, 5), RHO_CRN, +1, "single-domain/textured; shear strain 1.25-1.6 %"),
}

def main():
    lines = []
    def out(x=""):
        print(x); lines.append(x)
    out(f"Step 7 — design-rule table (CrN x-ray density {RHO_CRN:.3f} g/cm³)")
    out(f"{'material':34s} {'ΔS_t J/kg/K':>12s} {'ΔS_vol J/cm³/K':>15s} {'|dT/dP| K/GPa':>14s} {'ΔV/V (CC) %':>12s} {'h K':>6s} {'h/|dT/dP| GPa':>15s}")
    rows = {}
    for name, (dS, a, h, rho, sgn, note) in {**B, **CRN}.items():
        dSv = dS[0] * rho * 1e-3
        dVV = dSv * a[0] / 1e3 * 100                        # %, Clausius–Clapeyron
        pon = (h[0] / a[0], h[1] / a[2], h[2] / a[1]) if h else None
        rows[name] = (dS, a, h, rho, dSv, dVV, pon)
        out(f"{name:34s} {dS[0]:12.1f} {dSv:15.3f} {a[0]:14.1f} {dVV:12.2f} {('%g' % h[0]) if h else 'n.v.':>6s} "
            f"{(f'{pon[0]:.2f} ({pon[1]:.2f}-{pon[2]:.2f})') if pon else 'n.v.':>15s}")
    out("")
    out("CC consistency check: implied ΔV/V for CrN ≈ 0.57-0.6 % (measured 0.56-0.62 %); Mn3NiN ≈ 0.39 % (measured ~0.4 %).")
    out("Reading: per volume, CrN's transition entropy (0.63-0.80 J/cm³/K) is the largest of the magnetically ordered")
    out("materials listed and above NPG (≈ 0.40). Its small volume change (0.6 %) puts it at |dT/dP| ≈ 7-9 K/GPa, so its")
    out("reversibility onset under hydrostatic pressure (≈ 0.3-0.7 GPa for h = 3-5 K) is among the highest. The shear")
    out("channel (uniaxial <110>, CC estimate) moves CrN to ≈ 20-25 K/GPa and an onset of ≈ 0.12-0.25 GPa.")
    (ROOT / "results/step7_benchmark_table.txt").write_text("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SHORT = {"Mn3GaN": "Mn$_3$GaN", "Mn3NiN": "Mn$_3$NiN", "Fe49Rh51": "FeRh", "MnCoGeB0.03": "MnCoGeB",
             "Ni-Mn-In": "Ni–Mn–In", "NPG": "NPG", "(MnNiSi)0.62(FeCoGe)0.38": "MnNiSi–FeCoGe",
             "NdCu3Fe4O12": "NdCu$_3$Fe$_4$O$_{12}$", "Fe3(bntrz)6(tcnset)6 (SCO)": "spin crossover",
             "(NH4)2SO4": "(NH$_4$)$_2$SO$_4$", "CrN hydrostatic (Chaussy set)": "CrN, P (Chaussy)",
             "CrN hydrostatic (Browne set)": "CrN, P (Browne)", "CrN uniaxial <110> (CC estimate)": "CrN, σ∥⟨110⟩ (est.)"}
    OFF_A = {"Mn3GaN": (4, -11), "Mn3NiN": (4, -11), "Fe49Rh51": (4, -11), "MnCoGeB0.03": (4, 4), "Ni-Mn-In": (4, -11),
             "NPG": (4, -11), "(MnNiSi)0.62(FeCoGe)0.38": (-40, 7), "NdCu3Fe4O12": (-30, 7), "Fe3(bntrz)6(tcnset)6 (SCO)": (-30, 7),
             "(NH4)2SO4": (4, -11), "CrN hydrostatic (Chaussy set)": (6, -12), "CrN hydrostatic (Browne set)": (-20, 9),
             "CrN uniaxial <110> (CC estimate)": (6, 5)}
    OFF_B = {"Mn3NiN": (6, -3), "Fe49Rh51": (6, -10), "MnCoGeB0.03": (6, 2), "NPG": (6, 2), "(MnNiSi)0.62(FeCoGe)0.38": (6, 2),
             "Fe3(bntrz)6(tcnset)6 (SCO)": (6, 2), "(NH4)2SO4": (-55, 4), "CrN hydrostatic (Chaussy set)": (6, -12),
             "CrN hydrostatic (Browne set)": (6, 5), "CrN uniaxial <110> (CC estimate)": (-95, 6)}
    def style(name):
        crn = name.startswith("CrN")
        col = "tab:red" if "uniaxial" in name else ("tab:blue" if crn else "0.3")
        return crn, col
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=130)
    ax = axs[0]
    XL, YL = (2, 400), (0.08, 1.5)
    aa = np.logspace(np.log10(XL[0]), np.log10(XL[1]), 60)
    for dvv in (0.001, 0.003, 0.01, 0.03, 0.1):
        ax.plot(aa, dvv * 1e3 / aa, color="0.82", lw=0.8, zorder=0)
        xl = dvv * 1e3 / (0.88 * YL[1])
        if xl < XL[0] * 1.15:
            xl = XL[0] * 1.15
        yl = dvv * 1e3 / xl
        if YL[0] < yl < YL[1]:
            ax.text(xl, yl * 1.05, f"ΔV/V = {dvv*100:g} %", fontsize=6.5, color="0.5", clip_on=True)
    for name, (dS, a, h, rho, dSv, dVV, pon) in rows.items():
        crn, col = style(name)
        xerr = [[a[0] - a[1]], [a[2] - a[0]]]; yerr = [[(dS[0] - dS[1]) * rho * 1e-3], [(dS[2] - dS[0]) * rho * 1e-3]]
        ax.errorbar(a[0], dSv, xerr=xerr, yerr=yerr, fmt="*" if crn else "o", ms=12 if crn else 5, color=col, capsize=2,
                    lw=0.8, mfc="white" if "uniaxial" in name else col, mec=col, zorder=3)
        ax.annotate(SHORT[name], (a[0], dSv), fontsize=6.8, xytext=OFF_A.get(name, (4, 4)), textcoords="offset points",
                    color=col, annotation_clip=True)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*XL); ax.set_ylim(*YL)
    ax.set_xlabel("|dT$_t$/dP|  (or dT$_t$/dσ for the uniaxial estimate)  (K GPa$^{-1}$)")
    ax.set_ylabel("transition entropy per volume ΔS$_t$ (J cm$^{-3}$ K$^{-1}$)")
    ax.set_title("(a) Clausius–Clapeyron map: ΔS$_{vol}$ = (ΔV/V) / |dT/dP|", fontsize=9)
    ax = axs[1]
    for name, (dS, a, h, rho, dSv, dVV, pon) in rows.items():
        if pon is None:
            continue
        crn, col = style(name)
        ax.errorbar(pon[0], dS[0], xerr=[[pon[0] - pon[1]], [pon[2] - pon[0]]], yerr=[[dS[0] - dS[1]], [dS[2] - dS[0]]],
                    fmt="*" if crn else "o", ms=12 if crn else 5, color=col, capsize=2, lw=0.8,
                    mfc="white" if "uniaxial" in name else col, mec=col, zorder=3)
        ax.annotate(SHORT[name], (pon[0], dS[0]), fontsize=6.8, xytext=OFF_B.get(name, (6, 2)), textcoords="offset points",
                    color=col, annotation_clip=True)
    ax.axvline(0.6, color="gray", ls=":", lw=1)
    ax.text(0.63, 9.5, "≈ 0.6 GPa: limit of\nbarocaloric calorimetry", fontsize=6.5, color="gray", clip_on=True)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.005, 2); ax.set_ylim(8, 600)
    ax.set_xlabel("reversibility-onset estimate h / |dT$_t$/dP| (GPa)"); ax.set_ylabel("transition entropy ΔS$_t$ (J kg$^{-1}$ K$^{-1}$)")
    ax.set_title("(b) practical view: entropy vs pressure needed to start reversible cycling", fontsize=9)
    fig.tight_layout(); fig.savefig(ROOT / "figures/step7_design_map.png"); plt.close(fig)

if __name__ == "__main__":
    main()
