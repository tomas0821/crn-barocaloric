"""Step 1b: validate the digitized Chaussy (1971) C_p curve against the paper's own numbers.

Chaussy et al. report a latent heat of 29 J/g, "determined from the area of the anomaly", against
28 J/g from Clapeyron, with the C_p peak at 8.5 degC (warming). If the digitization is faithful,
integrating the excess over a background must land near 29 J/g. The background is not specified in
the paper, so several reasonable baselines are compared; the spread is a systematic uncertainty that
goes into the Monte-Carlo inputs.

Also derived, all as inputs for step 2:
- the transition entropy ΔS = ∫ excess/T dT, both from the whole anomaly and from its sharp part;
- the transition width W (FWHM of the excess, and the effective width L / peak excess);
- the background C_p near T_N (lattice + electronic), needed for ΔT_ad.

Input:  data/digitized/chaussy1971_fig3_cp.csv (from step1_digitize_chaussy.py)
Output: results/step1_cp_validation.txt, figures/step1_cp_baselines.png,
        data/digitized/chaussy1971_cp_grid.csv (uniform 0.1 K grid; rising branch below T_peak,
        falling branch above)
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
M_CRN = 66.003          # g/mol
CAL = 4.184             # J/cal
T0C = 273.15

def load():
    rows = [l.strip().split(",") for l in open(ROOT / "data/digitized/chaussy1971_fig3_cp.csv") if l[0].isdigit() or l[0] == "-"]
    T = np.array([float(r[0]) for r in rows]); C = np.array([float(r[2]) for r in rows]); b = np.array([r[4] for r in rows])
    return T, C, b

def branch_curve(T, C, sel):
    """Average duplicate T samples within 0.15 K bins; return sorted (T, C)."""
    t, c = T[sel], C[sel]
    o = np.argsort(t); t, c = t[o], c[o]
    bins = np.round(t / 0.15).astype(int)
    ub, inv = np.unique(bins, return_inverse=True)
    tm = np.bincount(inv, t) / np.bincount(inv); cm = np.bincount(inv, c) / np.bincount(inv)
    return tm, cm

def main():
    T, C, b = load()
    ipk = np.argmax(C); Tpk = T[ipk]
    rise_sel = np.isin(b, ["low", "spike-base", "rise"]) & (T <= Tpk)
    fall_sel = np.isin(b, ["fall", "high"]) | ((b == "spike-base") & (T > Tpk))
    fall_sel &= T >= Tpk
    tr, cr = branch_curve(T, C, rise_sel)
    tf, cf = branch_curve(T, C, fall_sel)
    grid = np.round(np.arange(-185.0, 40.0 + 1e-9, 0.1), 3)
    Cg = np.where(grid <= Tpk, np.interp(grid, tr, cr), np.interp(grid, tf, cf))
    np.savetxt(ROOT / "data/digitized/chaussy1971_cp_grid.csv", np.c_[grid, grid + T0C, Cg, Cg * CAL],
               delimiter=",", fmt="%.3f",
               header="T_C,T_K,Cp_cal_per_mol_K,Cp_J_per_mol_K  (uniform grid from step1_validate_cp.py; Chaussy 1971 Fig. 3)")

    TK = grid + T0C
    # Baselines (cal/mol/K).
    lo = (grid >= -185) & (grid <= -30); hi = (grid >= 30) & (grid <= 40)
    bl = {}
    # B1: straight line between the curve at -30 and +40 degC.
    c1, c2 = np.interp(-30, grid, Cg), np.interp(40, grid, Cg)
    bl["B1 line(-30..+40 C)"] = c1 + (c2 - c1) * (grid + 30) / 70
    # B2: quadratic fitted to both wings (-185..-30 and +30..+40 degC).
    sel = lo | hi; p = np.polyfit(grid[sel], Cg[sel], 2); bl["B2 quad fit both wings"] = np.polyval(p, grid)
    # B3: low-T quadratic extrapolated up to T_peak, then a line to the +40 degC value (a phase-switching background).
    p3 = np.polyfit(grid[lo], Cg[lo], 2); b3 = np.polyval(p3, grid)
    cpk3 = np.polyval(p3, Tpk); m = grid > Tpk
    b3[m] = cpk3 + (c2 - cpk3) * (grid[m] - Tpk) / (40 - Tpk); bl["B3 low-T quad -> step to high-T"] = b3
    # B4: narrow window, line between -10 and +25 degC (sharp anomaly only).
    c3, c4 = np.interp(-10, grid, Cg), np.interp(25, grid, Cg)
    b4 = np.full_like(grid, np.nan); w = (grid >= -10) & (grid <= 25)
    b4[w] = c3 + (c4 - c3) * (grid[w] + 10) / 35; bl["B4 line(-10..+25 C), sharp part only"] = b4

    lines = []
    def out(s=""):
        print(s); lines.append(s)
    out("Step 1b — validation of the digitized Chaussy (1971) Fig. 3 C_p curve")
    out(f"peak: C_p = {Cg.max():.1f} cal/mol/K = {Cg.max()*CAL:.0f} J/mol/K at {grid[Cg.argmax()]:.2f} degC (paper: 8.5 degC)")
    out(f"C_p at +40 degC (PM, above anomaly): {c2:.2f} cal/mol/K = {c2*CAL:.1f} J/mol/K (Dulong-Petit 6R = {6*8.314462/CAL:.2f} cal/mol/K)")
    out("")
    out(f"{'baseline':40s} {'window':>14s} {'L (cal/mol)':>11s} {'L (J/g)':>8s} {'dS (J/mol/K)':>13s} {'dS (J/kg/K)':>12s}")
    res = {}
    for name, base in bl.items():
        ex = Cg - base
        win = np.isfinite(base) & (grid >= (-10 if name.startswith("B4") else -30)) & (grid <= (25 if name.startswith("B4") else 40))
        exw = np.clip(ex[win], 0, None)
        L = np.trapezoid(exw, grid[win])                      # cal/mol
        dS = np.trapezoid(exw * CAL / TK[win], grid[win])     # J/mol/K
        res[name] = (L, dS)
        wtxt = "-10..+25 C" if name.startswith("B4") else "-30..+40 C"
        out(f"{name:40s} {wtxt:>14s} {L:11.0f} {L*CAL/M_CRN:8.1f} {dS:13.2f} {dS/M_CRN*1e3:12.1f}")
    out("")
    # Shortest temperature intervals holding 50 % and 90 % of each background's ΔS_t, and the closed-form
    # pressures they imply for the rigid shift: P_x ≈ (h + W_x)/a (h = 4 K, a = 9.0 K/GPa). Shows which
    # reversibility pressures depend on the background (P90 does, P50 barely).
    out("shortest intervals holding 50 % / 90 % of ΔS_t, and closed-form P50 / P90 = (h + W)/a at h = 4 K, a = 9.0 K/GPa:")
    for name, base in bl.items():
        win = np.isfinite(base) & (grid >= (-10 if name.startswith("B4") else -30)) & (grid <= (25 if name.startswith("B4") else 40))
        g, y = grid[win], np.clip((Cg - base)[win], 0, None) * CAL / TK[win]
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(g))]); cum /= cum[-1]
        def shortest(fr):
            best = np.inf
            for i in range(len(g)):
                j = np.searchsorted(cum, cum[i] + fr)
                if j < len(g):
                    best = min(best, g[j] - g[i])
            return best
        w50, w90 = shortest(0.5), shortest(0.9)
        out(f"  {name:40s} W50 = {w50:5.1f} K, W90 = {w90:5.1f} K -> P50 ≈ {(4 + w50)/9.0:.2f} GPa, P90 ≈ {(4 + w90)/9.0:.2f} GPa")
    out("")
    out("paper: L = 29 J/g (anomaly area) = 457 cal/mol; Clapeyron 28 J/g; dS = 29/281.65 J/g/K = 6.80 J/mol/K = 103 J/kg/K")
    # Width.
    base = bl["B1 line(-30..+40 C)"]; ex = Cg - base; pk = ex.max()
    above = grid[ex >= pk / 2]
    L1 = res["B1 line(-30..+40 C)"][0]
    out(f"excess peak (B1) = {pk:.1f} cal/mol/K; FWHM = {above.max()-above.min():.2f} K "
        f"({above.min():.2f} .. {above.max():.2f} degC); effective width L/peak = {L1/pk:.2f} K")
    # Fraction of latent heat inside the sharp part (|T - Tpk| < 5 K).
    sharp = (grid > Tpk - 5) & (grid < Tpk + 5)
    Ls = np.trapezoid(np.clip(ex[sharp], 0, None), grid[sharp])
    out(f"fraction of B1 anomaly area within +-5 K of the peak: {Ls/L1:.2f}")
    bgT = np.interp(Tpk, grid, base)
    out(f"background C_p at T_peak (B1): {bgT:.2f} cal/mol/K = {bgT*CAL:.1f} J/mol/K  -> lattice+electronic input for dT_ad")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/step1_cp_validation.txt").write_text("\n".join(lines) + "\n")

    import figstyle as fs
    import matplotlib.pyplot as plt
    nice = {"B1": "B1: line, 243–313 K", "B2": "B2: quadratic, both wings",
            "B3": "B3: low-T quadratic, step", "B4": "B4: line, 263–298 K (sharp peak)"}
    fig, axs = plt.subplots(1, 2, figsize=(fs.FULL, 2.5))
    for ax, xl in zip(axs, [(T0C - 190, T0C + 45), (T0C - 25, T0C + 40)]):
        ax.plot(TK, Cg * CAL, "k-", lw=1.0, label="digitized $C_p$")
        for (name, base), col in zip(bl.items(), ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
            ax.plot(TK, base * CAL, "--", lw=0.9, color=col, label=nice[name[:2]])
        ax.set_xlim(*xl); ax.set_xlabel("$T$ (K)"); ax.set_ylabel(r"$C_p$ (J mol$^{-1}$ K$^{-1}$)")
    axs[1].set_ylim(0, 335); axs[0].legend(loc="upper left")
    fs.panel(axs[0], "a"); fs.panel(axs[1], "b")
    fig.tight_layout(); fs.save(fig, ROOT / "figures/step1_cp_baselines"); plt.close(fig)

if __name__ == "__main__":
    main()
