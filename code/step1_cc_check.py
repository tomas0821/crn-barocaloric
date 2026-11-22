"""Step 1c: Clausius–Clapeyron consistency of the CrN inputs. This is INPUT VALIDATION, not a result.

Chaussy (1971) already checked it (28 J/g from Clapeyron vs 29 J/g calorimetric), and so did Browne
(1971). Here the check is redone for every pairing, to show which (ΔS, dT/dP, ΔV) combinations are
mutually consistent. Inputs are paired by source.

dT/dP = ΔV_m / ΔS_m, with ΔV_m = (ΔV/V)·N_A·V_fu.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NA = 6.02214076e23
VFU = 17.84e-30          # m^3
VM = NA * VFU            # m^3/mol
M = 66.003e-3            # kg/mol

dS_sets = {                                   # J/mol/K
    "Chaussy calorimetric (29 J/g @ 281.65 K)": 29 * 66.003 / 281.65,
    "this work, digitized whole anomaly":        6.47,
    "this work, digitized sharp part":           5.45,
    "Browne spin-only (0.130 J/g/K)":            0.130 * 66.003,
}
dVV_sets = {"-0.56 % (Bertaut/Sayetat 1969)": 0.0056, "-0.60 % (Chaussy text)": 0.0060, "-0.62 % (Wang 2023)": 0.0062}
measured_slopes = {"Browne 7.35-7.47": (7.35, 7.47), "Rivadulla replot 7.8": (7.8, 7.8), "Chaussy 9.0": (9.0, 9.0)}

lines = []
def out(s=""):
    print(s); lines.append(s)

out("Step 1c — Clausius–Clapeyron consistency (dT/dP = ΔV_m/ΔS_m), K/GPa")
out(f"V_m = {VM*1e6:.3f} cm^3/mol")
hdr = f"{'ΔS source':42s}" + "".join(f"{k:>30s}" for k in dVV_sets)
out(hdr)
for name, dS in dS_sets.items():
    row = f"{name:42s}"
    for dvv in dVV_sets.values():
        row += f"{dvv*VM/dS*1e9:30.2f}"
    out(row + f"   (ΔS = {dS:.2f} J/mol/K = {dS/M:.0f} J/kg/K)")
out("")
out("measured slopes (K/GPa): " + "; ".join(f"{k}" for k in measured_slopes))
out("")
out("Inverse check: the ΔS implied by each measured slope and ΔV/V (J/mol/K), and L at 281.65 K (J/g)")
for sk, (a1, a2) in measured_slopes.items():
    a = 0.5 * (a1 + a2) * 1e-9
    for dk, dvv in dVV_sets.items():
        dS = dvv * VM / a
        out(f"  slope {sk:22s} ΔV/V {dk:30s} -> ΔS = {dS:.2f} J/mol/K, L = {dS*281.65/66.003:.1f} J/g")
out("")
out("Reading: Chaussy's own pairing (9.0 K/GPa, -0.56 %) gives L = 28.5 J/g, i.e. their '28 J/g by Clapeyron';")
out("Browne's pairing (7.4 K/GPa, -0.6 %) gives 8.7 J/mol/K, close to his spin-only 0.130 J/g/K = 8.58 J/mol/K.")
out("The digitized whole-anomaly ΔS (6.47) pairs with 9.5-10 K/GPa. Never mix families or sources.")
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results/step1_cc_check.txt").write_text("\n".join(lines) + "\n")
