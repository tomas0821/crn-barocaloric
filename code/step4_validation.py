"""Step 4: BLIND validation of the quasi-direct pipeline (code/baro.py) on FeRh and Mn3NiN.
This is the go/no-go test of the method.

The rule of the test: predictions use ONLY ambient-pressure data (transition shape, ΔS_t, hysteresis)
plus the measured transition-temperature slopes dT_t/dP. The slopes are the same class of input the
CrN analysis uses. Everything measured under pressure (the ΔS(T,p) and ΔT(T,p) curves) is HELD OUT
and used only for comparison. The FeRh profile derived from the 2.5-kbar curves
(ferh_ambient_S_profile_from_fig4.csv) is deliberately NOT used: it would leak held-out information.

PRE-REGISTERED CRITERIA (fixed 2026-11-21, before the first run):
  FeRh (Stern-Taulats et al., PRB 89, 214105 (2014); vector-extracted Figs. 1, 4):
    C1  single-branch curves: at every pressure, predicted peak |ΔS| within ±2.5 J/kg/K (20 % of ΔS_t)
        of measured, and both half-maximum edges within ±1.5 K.
    C2  reversible (overlap construction, as in the paper): predicted peak |ΔS_rev| within
        ±2.5 J/kg/K at every pressure where the measured value is ≥ 50 % of ΔS_t; the pressure at
        which the reversible fraction crosses 0.5 is within ±0.03 GPa.
    C3  reversibility window at nominal 2.5 kbar (paper: 319-325 K): FWHM edges within ±1.5 K.
    PASS = C1 and C2 and C3.
  Mn3NiN (Boldrin et al., PRX 8, 041035 (2018); Figs. 1, 3; inverse BCE, cooling-only data):
    C4  for P ≤ 0.28 GPa: predicted peak |ΔS| and peak ΔT on pressure removal within ±20 % of
        measured. Above 0.28 GPa, the ratio pred/meas is REPORTED as a measure of how much the
        rigid-shift assumption (constant ΔS_t along the boundary) fails. Measured ΔS_t falls
        from 43 to ~34 J/kg/K by 0.56 GPa (their Fig. 3(b)). It is not a pass/fail item.
Run it capped (code/README.md).
"""
from pathlib import Path
import sys
import csv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baro import Baro

ROOT = Path(__file__).resolve().parents[1]
VD = ROOT / "data/validation"
lines = []
def out(s=""):
    print(s); lines.append(s)

def rows(fn):
    return [r for r in csv.reader(l for l in open(VD / fn) if not l.startswith("#"))][1:]

def half_edges(T, y):
    y = np.abs(y); m = y.max()
    if m <= 0:
        return np.nan, np.nan
    idx = np.where(y >= 0.5 * m)[0]
    return T[idx[0]], T[idx[-1]]

# ------------------------------------------------------------------ FeRh
def ferh():
    out("=== FeRh (Fe49Rh51), conventional BCE; blind inputs: Fig. 1 ambient heating thermogram, ΔS_t, h, slopes ===")
    r = rows("ferh_ambient_dqdT_fig1.csv")
    Th = np.array([float(x[1]) for x in r if x[0] == "heating"]); Ch = np.array([float(x[3]) for x in r if x[0] == "heating"])
    o = np.argsort(Th); Th, Ch = Th[o], Ch[o]
    T = np.round(np.arange(280.0, 360.0001, 0.02), 3)
    Cex = np.clip(np.interp(T, Th, Ch, left=0.0, right=0.0), 0, None)
    Tpk_h = T[np.argmax(Cex)]
    dSt, h, a_c, a_h = 12.5, 9.6, 64.0, 54.0
    b = Baro(T, Cex, np.full_like(T, 450.0), a=60.0, h=h, v=1 / 9360, alphaV=0.0, dS_target=dSt, a_heat=a_h, a_cool=a_c)
    out(f"inputs: ΔS_t = {dSt} J/kg/K, h = {h} K, a_cool = {a_c}, a_heat = {a_h} K/GPa; heating peak of the ambient shape at {Tpk_h:.2f} K (paper Th 319.4); lattice term off (authors neglect it)")
    # Measured curves.
    P_act = {"cooling": [0.312, 0.609, 1.118, 1.615, 1.896, 2.000, 2.405, 2.445],
             "heating": [0.323, 0.637, 1.161, 1.660, 1.913, 2.018, 2.443, 2.530]}
    labels = ["0.3", "0.6", "1.1", "1.6", "1.9", "2.0", "2.4", "2.5"]
    m = rows("ferh_fig4_dS_vs_T_by_pressure.csv")
    meas = {}
    for br in ("cooling", "heating"):
        for lab in labels:
            tt = np.array([float(x[2]) for x in m if x[0] == br and x[1] == lab]); dd = np.array([float(x[3]) for x in m if x[0] == br and x[1] == lab])
            oo = np.argsort(tt); meas[(br, lab)] = (tt[oo], dd[oo])
    Tc = np.round(np.arange(295.0, 340.0001, 0.02), 3)
    c1_ok, c2_ok = True, True
    out(f"{'p(kbar)':>7s} | {'cool peak meas/pred':>19s} {'edges meas':>15s} {'edges pred':>15s} | {'heat peak meas/pred':>19s} {'edges meas':>15s} {'edges pred':>15s} | {'rev meas/pred(overlap)/pred(cross)':>34s}")
    rev_meas, rev_pred, Pmean = [], [], []
    curves = {}
    for i, lab in enumerate(labels):
        pc, ph = P_act["cooling"][i] / 10, P_act["heating"][i] / 10
        tm_c, dm_c = meas[("cooling", lab)]; tm_h, dm_h = meas[("heating", lab)]
        pr_c = b.dS_branch(Tc, pc, "cool"); pr_h = b.dS_branch(Tc, ph, "heat")
        mc = np.interp(Tc, tm_c, dm_c, left=0, right=0); mh = np.interp(Tc, tm_h, dm_h, left=0, right=0)
        em_c, ep_c = half_edges(Tc, mc), half_edges(Tc, pr_c); em_h, ep_h = half_edges(Tc, mh), half_edges(Tc, pr_h)
        pk = lambda y: np.abs(y).max()
        okc = abs(pk(mc) - pk(pr_c)) <= 2.5 and abs(em_c[0] - ep_c[0]) <= 1.5 and abs(em_c[1] - ep_c[1]) <= 1.5
        okh = abs(pk(mh) - pk(pr_h)) <= 2.5 and abs(em_h[0] - ep_h[0]) <= 1.5 and abs(em_h[1] - ep_h[1]) <= 1.5
        c1_ok &= okc and okh
        rm = np.minimum(np.abs(mc), np.abs(mh)); rp = np.minimum(np.abs(pr_c), np.abs(pr_h))
        rx = np.abs(np.minimum(0.0, b.S_ex(Tc - a_c * pc + h) - b.S_ex(Tc)))
        rev_meas.append(rm.max()); rev_pred.append(rp.max()); Pmean.append(0.5 * (pc + ph))
        if rm.max() >= 0.5 * dSt:
            c2_ok &= abs(rm.max() - rp.max()) <= 2.5
        curves[lab] = (mc, pr_c, mh, pr_h, rm, rp)
        out(f"{lab:>7s} | {pk(mc):8.2f}/{pk(pr_c):<8.2f}{'' if okc else '*':1s} {em_c[0]:6.1f}-{em_c[1]:6.1f}  {ep_c[0]:6.1f}-{ep_c[1]:6.1f} | "
            f"{pk(mh):8.2f}/{pk(pr_h):<8.2f}{'' if okh else '*':1s} {em_h[0]:6.1f}-{em_h[1]:6.1f}  {ep_h[0]:6.1f}-{ep_h[1]:6.1f} | "
            f"{rm.max():10.2f}/{rp.max():<8.2f}/{rx.max():<8.2f}")
    # C2 crossing pressure.
    Pm = np.array(Pmean); rm_ = np.array(rev_meas) / dSt
    Pf = np.round(np.arange(0, 0.3, 0.001), 4)
    fp = np.array([np.minimum(np.abs(b.dS_branch(Tc, p, "cool")), np.abs(b.dS_branch(Tc, p, "heat"))).max() for p in Pf]) / dSt
    cross = lambda P, f: float(np.interp(0.5, f, P)) if (f >= 0.5).any() else np.nan
    Pc_m, Pc_p = cross(Pm, np.maximum.accumulate(rm_)), cross(Pf, np.maximum.accumulate(fp))
    c2x = abs(Pc_m - Pc_p) <= 0.03
    c2_ok &= c2x
    # C3 window at 2.5 kbar.
    mc, prc, mh, prh, rm, rp = curves["2.5"]
    wm, wp = half_edges(Tc, rm), half_edges(Tc, rp)
    c3_ok = abs(wm[0] - wp[0]) <= 1.5 and abs(wm[1] - wp[1]) <= 1.5
    out(f"reversible fraction crosses 0.5 at: measured {Pc_m:.3f} GPa, predicted {Pc_p:.3f} GPa")
    out(f"reversibility window (FWHM) at 2.5 kbar: measured {wm[0]:.1f}-{wm[1]:.1f} K (paper text: 319-325 K), predicted {wp[0]:.1f}-{wp[1]:.1f} K")
    out(f"C1 single-branch curves: {'PASS' if c1_ok else 'FAIL'}   (* marks a pressure that misses)")
    out(f"C2 reversible magnitude and crossing: {'PASS' if c2_ok else 'FAIL'}   (crossing within ±0.03 GPa: {c2x})")
    out(f"C3 reversibility window: {'PASS' if c3_ok else 'FAIL'}")
    verdict = c1_ok and c2_ok and c3_ok
    out(f"FeRh blind validation: {'PASS' if verdict else 'FAIL'}")
    out("")
    return Tc, curves, labels, verdict, (Pm, np.array(rev_meas), Pf, fp * dSt)

# ------------------------------------------------------------------ Mn3NiN
def mn3nin():
    out("=== Mn3NiN, inverse BCE (dT/dP < 0); cooling-only data; blind inputs: Fig. 1(g) ambient dQ/dT, ΔS_t(0), slope ===")
    r = rows("mnn_fig1_ambient_dQdT_S_V.csv")
    T0 = np.array([float(x[1]) for x in r if x[0] == "dQdT"]); Q0 = np.array([float(x[2]) for x in r if x[0] == "dQdT"])
    o = np.argsort(T0); T0, Q0 = T0[o], Q0[o]
    Tpk = T0[np.argmax(Q0)]
    wing = np.abs(T0 - Tpk) > 8.0
    pb = np.polyfit(T0[wing], Q0[wing], 1)
    T = np.round(np.arange(200.0, 300.0001, 0.02), 3)
    Cbg = np.polyval(pb, T)
    Cex = np.clip(np.interp(T, T0, Q0 - np.polyval(pb, T0), left=0.0, right=0.0), 0, None)
    raw = np.trapezoid(Cex / T, T)
    a = -13.5
    res = {}
    for tag, dSt in (("ΔS_t = 43.1 (paper ambient, incl. tail) [primary]", 43.1), (f"ΔS_t = {raw:.1f} (own linear-baseline integral)", raw)):
        b = Baro(T, Cex, Cbg, a=a, h=0.0, v=1 / 6860, alphaV=0.0, dS_target=dSt)
        res[tag] = b
    out(f"inputs: a = {a} K/GPa, ambient cooling peak at {Tpk:.2f} K, background C_p ≈ {np.polyval(pb, Tpk):.0f} J/kg/K, own excess integral {raw:.1f} J/kg/K")
    m = rows("mnn_fig3de_dS_dT_vs_T.csv")
    Ps = sorted({float(x[1]) for x in m})
    Tg = np.round(np.arange(215.0, 275.0001, 0.02), 3)
    c4 = True
    out(f"{'P(GPa)':>6s} | {'|ΔS| meas':>9s} " + " ".join(f"{'pred '+str(i+1):>8s}" for i in range(len(res))) + f" | {'ΔT meas':>7s} " + " ".join(f"{'pred '+str(i+1):>8s}" for i in range(len(res))) + " | ratio(primary) ΔS, ΔT")
    curves = {}
    for p in Ps:
        dSm = max(abs(float(x[3])) for x in m if x[0] == "dS" and float(x[1]) == p)
        dTm = max(float(x[3]) for x in m if x[0] == "dT" and float(x[1]) == p)
        preds_S, preds_T = [], []
        for tag, b in res.items():
            # Removal p -> 0 on the cooling branch (h = 0): ΔS = S(T,0) - S(T,p).
            dS = b.S_ex(Tg) - b.S_ex(Tg - a * p)
            preds_S.append(np.abs(dS).max())
            # ΔT on removal: S_tot(T', 0) = S_tot(T, p); with the lattice term negligible, solve by bisection.
            target = b.S_bg(Tg) + b.S_ex(Tg - a * p)
            lo, hi = Tg - 1.0, Tg + 30.0
            for _ in range(40):
                mid = 0.5 * (lo + hi); g = b.S_bg(mid) + b.S_ex(mid) - target
                hi = np.where(g > 0, mid, hi); lo = np.where(g > 0, lo, mid)
            preds_T.append((0.5 * (lo + hi) - Tg).max())
        rS, rT = preds_S[0] / dSm, preds_T[0] / dTm
        if p <= 0.28 + 1e-9:
            c4 &= abs(rS - 1) <= 0.2 and abs(rT - 1) <= 0.2
        out(f"{p:6.2f} | {dSm:9.2f} " + " ".join(f"{x:8.2f}" for x in preds_S) + f" | {dTm:7.2f} " + " ".join(f"{x:8.2f}" for x in preds_T)
            + f" | {rS:5.2f}, {rT:5.2f}{'' if p > 0.28 else ('' if (abs(rS-1)<=0.2 and abs(rT-1)<=0.2) else '  *')}")
        curves[p] = (dSm, preds_S[0], dTm, preds_T[0])
    out(f"C4 Mn3NiN (P ≤ 0.28 GPa, ±20 %): {'PASS' if c4 else 'FAIL'}")
    st = rows("mnn_fig3b_dSt_vs_P.csv")
    out("measured ΔS_t(P) on cooling (Fig. 3(b)): " + ", ".join(f"{float(x[0]):.2f} GPa: {float(x[1]):.1f}" for x in st[:: max(1, len(st)//6)]))
    out("")
    return curves, c4

def main():
    Tc, curves, labels, v_ferh, rev = ferh()
    mcurves, c4 = mn3nin()
    out(f"GO/NO-GO: FeRh {'PASS' if v_ferh else 'FAIL'}; Mn3NiN C4 {'PASS' if c4 else 'FAIL'}")
    (ROOT / "results/step4_validation.txt").write_text("\n".join(lines) + "\n")

    import figstyle as fs
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(fs.FULL, 2.45))
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(labels)))
    for lab, col in zip(labels, cols):
        mc, pc, mh, ph, rm, rp = curves[lab]
        axs[0].plot(Tc, mc, color=col, lw=1.0); axs[0].plot(Tc, pc, "--", color=col, lw=0.8)
        axs[1].plot(Tc, mh, color=col, lw=1.0, label=f"{float(lab)/10:.2f}"); axs[1].plot(Tc, ph, "--", color=col, lw=0.8)
    for k, ax in enumerate(axs[:2]):
        ax.set_xlim(300, 335); ax.set_xlabel("$T$ (K)"); fs.panel(ax, "ab"[k], x=-0.22)
    axs[0].set_ylabel(r"$\Delta S$ (J kg$^{-1}$ K$^{-1}$)")
    axs[1].legend(title="$P$ (GPa)", ncol=2, loc="lower left", bbox_to_anchor=(-0.03, -0.03), fontsize=6.5, title_fontsize=6.5, handlelength=0.9, columnspacing=0.5, handletextpad=0.4)
    Pm, rmeas, Pf, fpred = rev
    axs[2].plot(Pm, rmeas, "ko", ms=3, label="FeRh meas.")
    axs[2].plot(Pf, fpred, "k--", lw=0.9, label="FeRh pred.")
    ps = sorted(mcurves); axs[2].plot(ps, [mcurves[p][0] for p in ps], "s", ms=3, color="tab:red", label="Mn$_3$NiN meas.")
    axs[2].plot(ps, [mcurves[p][1] for p in ps], "--", lw=0.9, color="tab:red", label="Mn$_3$NiN pred.")
    axs[2].set_xlabel("$P$ (GPa)"); axs[2].set_ylabel(r"peak $|\Delta S|$ (J kg$^{-1}$ K$^{-1}$)"); axs[2].legend(loc="center right", bbox_to_anchor=(1.0, 0.45), fontsize=6.5, handlelength=1.2, borderaxespad=0.2)
    fs.panel(axs[2], "c", x=-0.22)
    fig.tight_layout(w_pad=0.6); fs.save(fig, ROOT / "figures/step4_validation"); plt.close(fig)

if __name__ == "__main__":
    main()
