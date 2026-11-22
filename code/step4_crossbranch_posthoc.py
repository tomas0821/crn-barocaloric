"""Step 4, post hoc: score the cross-branch construction (Eq. 3, the one used for every CrN number) on FeRh.

This check was added after the blind validation and is labelled post hoc. The pre-specified criteria C2/C3
(code/step4_validation.py) score the overlap min(|ΔS_cool|, |ΔS_heat|) of the two single-branch curves, so
they test the single-branch shifts. The CrN results use the cross-branch pairing instead: heating at P = 0
against cooling at P,
    ΔS_x(T, P) = S_c(T, P) - S_h(T, 0)   (transition part; negative = entropy released on compression).

Measured cross-branch target, built from the deposited data only:
    S_c(T, P) - S_h(T, 0) = [S_c(T, P) - S_c(T, 0)]  +  [S_c(T, 0) - S_h(T, 0)]
                             Fig. 4(a) cooling curve      ambient hysteresis gap, built two ways:
  A: from the Fig. 1 thermograms, each branch integrated with its own sign within ±5 K of its peak and
     normalized (the cooling trace carries a residual baseline far from its peak; integrating |dq/dT| over
     the whole trace would count it as transition area). Windows of ±3-10 K give the same result.
  B: from the ambient profiles derived from the Fig. 4 curves at 0.25 GPa (assumes that sample is fully AFM).
Stern-Taulats et al. (2014, p. 12) define the reversibility region the same way: bounded by the ambient
reverse transition and the forward transition under pressure.

Prediction: the same blind inputs as step4_validation.py (ambient heating shape, ΔS_t = 12.5 J/kg/K,
h = 9.6 K, a_cool = 64 K/GPa, lattice term off), with S_c(T, P) = S_ex(T + h - a_cool P).
Reported: the pressure at which max_T |ΔS_x| reaches ΔS_t/2, the magnitudes, and the window at 2.5 kbar.
Also reported: the temperatures at which the two ambient constructions reach x_FM = 0.5 (the registration
of the two figures), and, for CrN, the overlap vs cross-branch reversible fraction. The pressure grid is
sparse (8 cooling curves).
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
    idx = np.where(y >= 0.5 * m)[0]
    return (T[idx[0]], T[idx[-1]]) if m > 0 else (np.nan, np.nan)

dSt, h, a_c, a_h = 12.5, 9.6, 64.0, 54.0
r = rows("ferh_ambient_dqdT_fig1.csv")
br = {k: (np.array([float(x[1]) for x in r if x[0] == k]), np.array([float(x[3]) for x in r if x[0] == k])) for k in ("heating", "cooling")}
T = np.round(np.arange(280.0, 360.0001, 0.02), 3)

def frac_fm(branch, win=5.0):
    """FM (high-T phase) fraction along an ambient branch: cumulative ∫(sign·dq/dT)/T dT from low T, taken
    within ±win K of the branch's peak (heating endothermic +, cooling exothermic -), clipped at 0, normalized."""
    sgn = 1.0 if branch == "heating" else -1.0
    t, c = br[branch]; o = np.argsort(t); t, c = t[o], sgn * c[o]
    pk = t[np.argmax(c)]
    y = np.interp(T, t, c, left=0.0, right=0.0)
    y = np.where(np.abs(T - pk) <= win, np.clip(y, 0.0, None), 0.0) / T
    F = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(T))])
    return F / F[-1]

Fh, Fc = frac_fm("heating"), frac_fm("cooling")
gap0 = dSt * (Fc - Fh)                     # S_c(T,0) - S_h(T,0) >= 0 inside the ambient hysteresis loop

th, ch = br["heating"]; o = np.argsort(th)
Cex = np.clip(np.interp(T, th[o], ch[o], left=0.0, right=0.0), 0, None)
b = Baro(T, Cex, np.full_like(T, 450.0), a=60.0, h=h, v=1 / 9360, alphaV=0.0, dS_target=dSt, a_heat=a_h, a_cool=a_c)

P_cool = [0.312, 0.609, 1.118, 1.615, 1.896, 2.000, 2.405, 2.445]           # kbar (actual, Fig. 3a/5a)
labels = ["0.3", "0.6", "1.1", "1.6", "1.9", "2.0", "2.4", "2.5"]
m = rows("ferh_fig4_dS_vs_T_by_pressure.csv")
Tc = np.round(np.arange(300.0, 335.0001, 0.02), 3)
gap_c = np.interp(Tc, T, gap0)

rp = rows("ferh_ambient_S_profile_from_fig4.csv")
def prof(branch, Tmax):
    """x_FM along an ambient branch from the Fig. 4-derived profile (same temperature scale as Fig. 4)."""
    t = np.array([float(x[1]) for x in rp if x[0] == branch]); xf = np.array([float(x[3]) for x in rp if x[0] == branch])
    o = np.argsort(t); y = np.interp(Tc, t[o], xf[o]); y[Tc > Tmax] = 1.0
    return np.clip(y, 0.0, 1.0)
gaps = {
    "A: ambient gap from Fig. 1 thermograms (signed, ±5 K)": gap_c,
    "B: ambient gap from Fig. 4-derived profiles": dSt * (prof("cooling", 318.0) - prof("heating", 327.0)),
}
Pf = np.round(np.arange(0.0, 0.3, 0.001), 4)
fp = np.array([np.abs(np.minimum(0.0, b.S_ex(Tc + h - a_c * p) - b.S_ex(Tc))).max() for p in Pf])
cross = lambda P, f: float(np.interp(0.5 * dSt, np.maximum.accumulate(f), P)) if (np.asarray(f) >= 0.5 * dSt).any() else np.nan
Pp_half = cross(Pf, fp)

out("Step 4, post hoc: cross-branch construction (Eq. 3) scored on FeRh. NOT part of the pre-specified criteria.")
out(f"FeRh h/FWHM = {h:.1f}/{np.diff(half_edges(T, Cex))[0]:.2f} K (CrN: 3-5/5.3 K)")
for gname, gap in gaps.items():
    out(f"== target {gname} (max gap {gap.max():.2f} J/kg/K)")
    out(f"{'p (GPa)':>8s} | {'max|ΔS_x| measured':>19s} {'predicted Eq.(3)':>17s} {'diff':>6s}")
    Pm, Xm, Xp, cur = [], [], [], {}
    for lab, pk in zip(labels, P_cool):
        p = pk / 10
        tt = np.array([float(x[2]) for x in m if x[0] == "cooling" and x[1] == lab]); dd = np.array([float(x[3]) for x in m if x[0] == "cooling" and x[1] == lab])
        oo = np.argsort(tt)
        dcool = np.interp(Tc, tt[oo], dd[oo], left=0.0, right=0.0)          # S_c(T,P) - S_c(T,0), <= 0
        xm = np.abs(np.minimum(0.0, dcool + gap))
        xp = np.abs(np.minimum(0.0, b.S_ex(Tc + h - a_c * p) - b.S_ex(Tc)))
        Pm.append(p); Xm.append(xm.max()); Xp.append(xp.max()); cur[lab] = (xm, xp)
        out(f"{p:8.3f} | {xm.max():19.2f} {xp.max():17.2f} {xp.max() - xm.max():+6.2f}")
    Pm, Xm = np.array(Pm), np.array(Xm)
    Pm_half = cross(Pm, Xm)
    wm, wp = half_edges(Tc, cur["2.5"][0]), half_edges(Tc, cur["2.5"][1])
    phys = "physical" if Xm.max() <= dSt * 1.05 and Xm[0] < 1.0 else "UNPHYSICAL (exceeds ΔS_t or nonzero at 0.03 GPa)"
    out(f"   target is {phys}")
    out(f"   half-reversibility pressure: measured {Pm_half:.3f} GPa, Eq.(3) {Pp_half:.3f} GPa (diff {Pp_half - Pm_half:+.3f}; C2 tolerance ±0.03)")
    out(f"   window at 2.5 kbar: measured {wm[0]:.1f}-{wm[1]:.1f} K, Eq.(3) {wp[0]:.1f}-{wp[1]:.1f} K (published 319-325 K; C3 tolerance ±1.5 K)")
    d = np.array(Xp) - Xm; big = Xm >= 0.5 * dSt
    out(f"   magnitudes (pred - meas): {d[big].min():+.2f} to {d[big].max():+.2f} J/kg/K where meas >= ΔS_t/2 (C2 rule); "
        f"at 0.162 GPa {d[3]:+.2f} (meas {Xm[3]:.2f}, pred {Xp[3]:.2f})")
    out("")
# Window sensitivity of target A and the registration of the two ambient constructions.
out("== target A, sensitivity to the integration window (half-reversibility pressure, measured)")
for wv in (3.0, 4.0, 6.0, 8.0, 10.0):
    gw = np.interp(Tc, T, dSt * (frac_fm("cooling", wv) - frac_fm("heating", wv)))
    Xw = []
    for lab in labels:
        tt = np.array([float(x[2]) for x in m if x[0] == "cooling" and x[1] == lab]); dd = np.array([float(x[3]) for x in m if x[0] == "cooling" and x[1] == lab])
        oo = np.argsort(tt); Xw.append(np.abs(np.minimum(0.0, np.interp(Tc, tt[oo], dd[oo], left=0.0, right=0.0) + gw)).max())
    out(f"   ±{wv:4.1f} K: {cross(np.array(P_cool) / 10, np.array(Xw)):.3f} GPa")
x50 = lambda TT, xx: float(np.interp(0.5, xx, TT))
out("== registration: temperature at x_FM = 0.5 (heating / cooling)")
out(f"   Fig. 1 thermograms (±5 K): {x50(T, frac_fm('heating')):.2f} / {x50(T, frac_fm('cooling')):.2f} K")
out(f"   Fig. 4-derived profiles  : {x50(Tc, prof('heating', 327.0)):.2f} / {x50(Tc, prof('cooling', 318.0)):.2f} K")
out("")
(ROOT / "results/step4_crossbranch_posthoc.txt").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------- CrN: overlap vs cross-branch reversibility
# Why the CrN numbers use the cross-branch construction: when h is below the transition width, the overlap of
# the two single-branch curves becomes non-zero at vanishing pressure (a spurious onset).
import step3_montecarlo as s3
T3, Cex3, Cbg3, Tpk3 = s3.chaussy_shape()
TG, PG = s3.TG, s3.PG
out("== CrN (Chaussy set, central inputs a = 9.0 K/GPa, no curvature): reversible fraction, transition part only")
out(f"{'h (K)':>6s} | {'overlap onset/P50 (GPa)':>24s} | {'cross-branch onset/P50 (GPa)':>29s}")
from baro import first_crossing
for hh in (2.0, 3.0, 4.0, 5.0):
    bb = s3.make(T3, Cex3, Cbg3, 0.0, 6.80, 9.0, hh, 0.0, 1.0)
    fo, fx = [], []
    for p in PG:
        sh = bb.S_ex(TG - 9.0 * p) - bb.S_ex(TG); sc = bb.S_ex(TG - 9.0 * p + hh) - bb.S_ex(TG + hh)
        fo.append(np.minimum(np.abs(sh), np.abs(sc)).max() / bb.dS)
        fx.append(np.abs(np.minimum(0.0, bb.S_ex(TG - 9.0 * p + hh) - bb.S_ex(TG))).max() / bb.dS)
    fo, fx = np.array(fo), np.array(fx)
    out(f"{hh:6.1f} | {first_crossing(PG, fo, 0.01):10.2f} / {first_crossing(PG, fo, 0.5):<11.2f} | "
        f"{first_crossing(PG, fx, 0.01):13.2f} / {first_crossing(PG, fx, 0.5):<13.2f}")
(ROOT / "results/step4_crossbranch_posthoc.txt").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------- Figure S2 (FeRh cross-branch, target B)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7})
def target_points(gap):
    out_ = []
    for lab in labels:
        tt = np.array([float(x[2]) for x in m if x[0] == "cooling" and x[1] == lab]); dd = np.array([float(x[3]) for x in m if x[0] == "cooling" and x[1] == lab])
        oo = np.argsort(tt); out_.append(np.abs(np.minimum(0.0, np.interp(Tc, tt[oo], dd[oo], left=0.0, right=0.0) + gap)).max())
    return np.array(out_)
Pm_ = np.array(P_cool) / 10
fpo = np.array([np.minimum(np.abs(b.S_ex(Tc - a_c * p) - b.S_ex(Tc)), np.abs(b.S_ex(Tc - a_h * p) - b.S_ex(Tc))).max() for p in Pf])
fig, ax = plt.subplots(figsize=(3.35, 2.5))
ax.plot(Pf, fp, "-", color="tab:blue", lw=1.2, label="predicted, cross-branch (eq. 3)")
ax.plot(Pf, fpo, "--", color="tab:orange", lw=1.0, label="predicted, overlap")
ax.plot(Pm_, target_points(gaps["A: ambient gap from Fig. 1 thermograms (signed, ±5 K)"]), "o", color="k", ms=3.5, label="measured, target A")
ax.plot(Pm_ + 0.003, target_points(gaps["B: ambient gap from Fig. 4-derived profiles"]), "s", mfc="none", color="0.4", ms=3.5, label="measured, target B")
ax.axhline(0.5 * dSt, color="gray", ls=":", lw=0.8)
ax.set_xlabel("pressure (GPa)"); ax.set_ylabel(r"peak $|\Delta S_\mathrm{x}|$ (J kg$^{-1}$ K$^{-1}$)")
ax.set_xlim(0, 0.27); ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(ROOT / f"figures/stepS2_ferh_crossbranch.{ext}", dpi=300)
plt.close(fig)
