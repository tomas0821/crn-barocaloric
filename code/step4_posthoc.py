"""Step 4 POST-HOC diagnostics. Not part of the pre-registered test; these were run after its result was known.

The pre-registered blind validation (step4_validation.py) failed two sub-criteria:
  C1 (FeRh): the lowest-pressure HEATING curve (0.0323 GPa): measured peak |ΔS| 0.60 vs predicted 5.72 J/kg/K.
  C4 (Mn3NiN): |ΔS| under-predicted by 27 % and 22 % at 0.11 and 0.14 GPa.
Diagnosis, written down after the fact:
  D1 The paper's own Fig. 3(a) heating peak at 0.323 kbar is 319.47 K, essentially unshifted from
     ambient (319.37-320.32 K). Their linear fit (54 K/GPa) predicts +1.7 K. The measured near-zero
     ΔS is consistent with that unshifted transition, so the miss comes from the slope input, not
     from the reconstruction.
  D2 Mn3NiN T_N(p) is curved: the initial slope is ≈ -18 K/GPa against the -13.5 average. With the
     initial slope, every P ≤ 0.28 GPa lands within ±8 %.
Lesson for CrN: the dominant error source is the T_N(P) slope and its curvature, not the
quasi-direct construction. The CrN slope prior must therefore be widened.
"""
from pathlib import Path
import sys, csv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baro import Baro

ROOT = Path(__file__).resolve().parents[1]
VD = ROOT / "data/validation"
rows = lambda fn: [r for r in csv.reader(l for l in open(VD / fn) if not l.startswith("#"))][1:]
lines = []
def out(s=""):
    print(s); lines.append(s)

out("Step 4 POST-HOC diagnostics (not part of the pre-registered test)")
t3 = rows("ferh_fig3a_Tt_vs_P.csv")
hp = [(float(x[1]), float(x[2])) for x in t3 if x[0] == "heating"]
out("D1 FeRh heating T_peak vs P (Fig. 3a): " + "; ".join(f"{p:.3f} kbar -> {t:.2f} K" for p, t in hp[:4]))
out("   expected shift at 0.323 kbar from the linear 54 K/GPa: +1.74 K; observed vs ambient (319.37 K): "
    f"{hp[2][1]-hp[0][1]:+.2f} K")

r = rows("mnn_fig1_ambient_dQdT_S_V.csv")
T0 = np.array([float(x[1]) for x in r if x[0] == "dQdT"]); Q0 = np.array([float(x[2]) for x in r if x[0] == "dQdT"])
o = np.argsort(T0); T0, Q0 = T0[o], Q0[o]
Tpk = T0[np.argmax(Q0)]; wing = np.abs(T0 - Tpk) > 8; pb = np.polyfit(T0[wing], Q0[wing], 1)
T = np.round(np.arange(200, 300.0001, 0.02), 3)
Cex = np.clip(np.interp(T, T0, Q0 - np.polyval(pb, T0), left=0, right=0), 0, None)
m = rows("mnn_fig3de_dS_dT_vs_T.csv"); Tg = np.round(np.arange(215, 275.0001, 0.02), 3)
out("D2 Mn3NiN pred/meas peak |ΔS| on removal (ΔS_t = 43.1):")
for a in (-13.5, -18.0):
    b = Baro(T, Cex, np.polyval(pb, T), a=a, h=0, v=1 / 6860, alphaV=0, dS_target=43.1)
    s = []
    for p in (0.11, 0.14, 0.19, 0.24, 0.28):
        dSm = max(abs(float(x[3])) for x in m if x[0] == "dS" and float(x[1]) == p)
        s.append(f"{p}: {np.abs(b.S_ex(Tg) - b.S_ex(Tg - a * p)).max() / dSm:.2f}")
    out(f"   a = {a:6.1f} K/GPa -> " + "  ".join(s))
(ROOT / "results/step4_posthoc.txt").write_text("\n".join(lines) + "\n")
