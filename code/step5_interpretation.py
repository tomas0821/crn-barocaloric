"""Step 5: the interpretive mean-field section.

(1) The η split. Why does hydrostatic pressure couple weakly? In the S = 3/2 Bean–Rodbell-type free
    energy, F/R = -c·T0(1+βω)σ² - Q·σ⁴ + (V_m/R)[ω²/2K + Pω] - T·S(σ)/R, the effective coupling that
    makes the transition first order is η_tot = Q/(T0·d) (d = Brillouin quartic coefficient). Its volume
    (A1g) part is η_vol = K·c²·T0·β²·(R/V_m)/(2d). β is calibrated SELF-CONSISTENTLY:
        |β| = a·Δs(σ_j) / (K·c·T0·σ_j²),  with T0 = T_N/g(η)
    so the first-order model reproduces the measured dT_N/dP (the second-order recipe β = aB/T_N is wrong
    here). The calculation is restricted to the two PAIRED source sets.
(2) Entropy decomposition, with every number traced to an input file or a step-1/2 result.
Lightweight (scalar root-finding only). Run it capped (code/README.md).
"""
from pathlib import Path
import numpy as np
from scipy.optimize import brentq, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
S = 1.5; ms = np.arange(-S, S + 1)
R = 8.314462618; NA = 6.02214076e23; M = 66.003e-3
Vm = 17.84e-30 * NA; RV = R / Vm
c = 3 * S / (2 * (S + 1))
lines = []
def out(x=""):
    print(x); lines.append(x)

def sig_of_y(y):
    w = np.exp(y * ms / S); return (w * ms).sum() / (S * w.sum())
def Ds(sig):
    """Spin-entropy loss (units of R) at reduced moment sig, S = 3/2 Brillouin."""
    if sig <= 0: return 0.0
    y = brentq(lambda y: sig_of_y(y) - sig, 1e-12, 200)
    w = np.exp(y * ms / S); return np.log(2 * S + 1) - (np.log(w.sum()) - y * sig)
xs = np.array([0.01, 0.02, 0.03])
d = np.polyfit(xs**2, [(Ds(x) - c * x**2) / x**4 for x in xs], 1)[1]
def fmin(T, T0, Q):
    f = lambda s: -c * T0 * s**2 - Q * s**4 + T * Ds(s)
    grid = np.linspace(0.02, 0.999, 120); v = [f(s) for s in grid]; i = int(np.argmin(v))
    r = minimize_scalar(f, bounds=(grid[max(i-1, 0)], grid[min(i+1, len(grid)-1)]), method="bounded", options={"xatol": 1e-10})
    return r.fun, r.x
def transition(eta, T0=1.0):
    Q = eta * d * T0
    if eta <= 1: return T0, 0.0
    Tt = brentq(lambda T: fmin(T, T0, Q)[0], T0 * 1.0000001, T0 * 20, xtol=1e-10)
    return Tt, fmin(Tt * (1 - 1e-9), T0, Q)[1]
def eta_for_sig(sj):
    return brentq(lambda e: transition(e)[1] - sj, 1.05, 60, xtol=1e-6)

out("Step 5 — interpretive mean field (paired source sets only)")
out(f"S = 3/2: c = {c:.3f}, d = {d:.4f}, Bean–Rodbell prefactor c²/(2d) = {c*c/(2*d):.3f}")
out("")
out("(1) η split: total effective first-order coupling vs its hydrostatic (volume) part")
sets = {"Chaussy (ΔS 6.80, a 9.0, T_N 281.65)": (6.80, 9.0, 281.65), "Browne (ΔS 8.58, a 7.41, T_N 287)": (8.58, 7.41, 287.0)}
out(f"{'set':38s} {'σ_j':>6s} {'η_tot':>6s} {'T0(K)':>7s} {'|β|':>6s} {'η_vol (B 257-262)':>18s} {'η_vol/η_tot':>12s}")
for name, (dS, a, TN) in sets.items():
    sj = brentq(lambda s: R * Ds(s) - dS, 0.3, 0.999)
    eta = eta_for_sig(sj); g, _ = transition(eta); T0 = TN / g
    ev = []
    for B in (257e9, 262e9):
        K = 1 / B; b = a * 1e-9 * Ds(sj) / (K * c * T0 * sj**2)
        ev.append((K * c**2 * T0 * b**2 * RV / (2 * d), b))
    out(f"{name:38s} {sj:6.3f} {eta:6.2f} {T0:7.1f} {ev[0][1]:6.1f} {ev[0][0]:8.3f}-{ev[1][0]:<8.3f} {ev[0][0]/eta:6.3f}-{ev[1][0]/eta:.3f}")
out("Chaussy's α (θ_N(σ) = θ_N(0)(1+ασ²)) maps to η = c·α/(2d): " + ", ".join(f"α = {al} -> η = {c*al/(2*d):.2f}" for al in (2.3, 3.4)))
# σ_j scan: η_tot depends on the chosen jump; η_vol (calibrated to dT_N/dP, Chaussy set) stays < 1 throughout.
# With β calibrated, η_vol equals B·V_m·(ΔV/V)²/(2 d R T0 σ_j⁴), the elastic energy of the volume jump.
out("σ_j scan (Chaussy a = 9.0, T_N 281.65; B = 257 GPa; σ_j = 0.856 calibrated, 0.894-0.949 strain-implied):")
for sj in (0.856, 0.894, 0.93, 0.949):
    eta = eta_for_sig(sj); g, _ = transition(eta); T0 = 281.65 / g
    K = 1 / 257e9; b = 9.0e-9 * Ds(sj) / (K * c * T0 * sj**2)
    ev = K * c**2 * T0 * b**2 * RV / (2 * d)
    out(f"  σ_j = {sj:.3f}: η_tot = {eta:5.2f}, η_vol = {ev:.3f}, share {100*ev/eta:4.1f} %")
out("Reading: the volume coupling alone is too weak to make the transition first order (η_vol < 1 for every")
out("σ_j); its share of η_tot (≈2-7 %) depends on σ_j. With β calibrated to dT_N/dP, η_vol restates the")
out("Clausius-Clapeyron relation; the mean field does not identify the remaining (non-volume) coupling.")
out("")
out("(2) Entropy decomposition (J mol⁻¹ K⁻¹; ×15.15 for J kg⁻¹ K⁻¹)")
Rln4 = R * np.log(4)
rows = [
    ("full spin entropy R ln 4 (S = 3/2)", Rln4, "upper bound"),
    ("Brillouin jump at σ_j(strain) = 0.89-0.95", f"{R*Ds(0.894):.2f}-{R*Ds(0.949):.2f}", "Chaussy strains 10-20 % below their 22 K values (strain ∝ σ², Chaussy 1971)"),
    ("Brillouin jump at Browne σ_c = 0.93", R * Ds(0.93), "Browne 1971 (spin-only; NOT calorimetric)"),
    ("calorimetric (Chaussy latent heat 29 J/g)", 29 * 66.003 / 281.65, "measured"),
    ("digitized whole anomaly, -30..+40 °C", "6.30-6.55", "results/step1_cp_validation.txt"),
    ("digitized sharp part only, -10..+25 °C", 5.45, "results/step1_cp_validation.txt"),
]
for name, val, src in rows:
    v = f"{val:.2f}" if isinstance(val, float) else val
    out(f"  {name:44s} {v:>11s}   [{src}]")
out(f"  -> calorimetric/R ln 4 = {29*66.003/281.65/Rln4:.2f}; strain-implied Brillouin exceeds calorimetry by "
    f"{R*Ds(0.894)-6.80:.1f}-{R*Ds(0.949)-6.80:.1f} J/mol/K ({(R*Ds(0.894)/6.80-1)*100:.0f}-{(R*Ds(0.949)/6.80-1)*100:.0f} %)")
out("  -> ≈ 16 % of the digitized anomaly sits in broad tails (pre-transition rise and post-transition tail)")
out("  lattice share: Browne 1971 states the lattice ΔS is < 1 % of the transition entropy; a Grüneisen")
out("  estimate (γ ≈ 1.7) gives ~0.5 J/mol/K (≈ 7 %). Reported as a range, unresolved.")
out("Reading (hedged): the calorimetric jump is ~59 % of R ln 4 and 13-36 % below the strain-implied")
out("mean-field value. That is consistent with entropy already removed above T_N by paramagnetic short-range")
out("order in a strongly frustrated lattice (|θ_p|/T_N ≈ 6-9; θ_p = -1650 to -2600 K, Chaussy), and with the")
out("broad tails. The single-order-parameter mean field cannot capture this; it is stated as a limitation.")
(ROOT / "results/step5_interpretation.txt").write_text("\n".join(lines) + "\n")
