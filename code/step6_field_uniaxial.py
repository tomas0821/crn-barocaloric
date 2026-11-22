"""Step 6: (A) the field contrast and (B) a MODEL-FREE uniaxial-stress estimate.

(A) Magnetic field. The magnetic Clausius–Clapeyron analogue for a first-order AFM->PM transition with
    linear magnetization in each phase gives
        ΔT_N(H) = -½·Δχ·H² / ΔS_t
    with Δχ = χ_PM - χ_AFM at T_N (measured: 2.0-2.8e-4 emu mol⁻¹ Oe⁻¹; Lacour-Gayet 1973, the Browne curve
    in Tsuchiya 1996, and Wang 2023). The upper bound uses χ_AFM = 0, i.e. Δχ = χ_PM ≈ 6.0-6.6e-4.
    Also derived: the field needed for |ΔT_N| = h, H = sqrt(2·h·ΔS_t/Δχ), and the first-application
    ΔS_M ≤ ΔS_t·|ΔT_N|/W.

(B) Uniaxial stress, Clausius–Clapeyron only (no microscopic model). For a uniaxial stress σ along n:
        dT_N/dσ = V_m·|Δε_nn| / ΔS_t
    where Δε_nn is the transformation strain along n for the stress-selected domain.
    - The 88.3° distortion makes the cubic <110> face diagonals into the orthorhombic a and b axes. On
      ordering, a contracts and b expands by ~1.3-1.6 %; c is nearly unchanged.
      Sources: Browne et al. 1971, on heating Δa/a = +1.6 %, Δb/b = -1.3 %, Δc/c = +0.25 %; Chaussy et al.
      1971, at 22 K ε = -1.48 %, η = +1.56 %, ζ = -0.08 %, and 10-20 % smaller just below T_N.
    - Compression along cubic <110> therefore selects the domain with a ∥ σ (Browne's stress-cooling
      experiment shows this domain selection) and raises T_N.
    - Range used: |Δε| = 1.25-1.60 %.
    - The quasi-direct reversible response under stress uses the validated code/baro.py with a = dT_N/dσ.
      The lattice term becomes the linear expansion α_V/3.
    CAVEATS, stated in the output: a single-domain or textured sample is assumed, and polycrystal
    averaging lowers the slope; the mechanical strength and fatigue of bulk CrN at ~0.3-0.5 GPa cyclic
    stress are unverified.
Run it capped (code/README.md).
"""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baro import Baro, first_crossing
import step3_montecarlo as s3

ROOT = Path(__file__).resolve().parents[1]
NA = 6.02214076e23
VM = NA * 17.84e-30                 # m^3/mol
M = 66.003e-3
RNG = np.random.default_rng(611)
lines = []
def out(x=""):
    print(x); lines.append(x)

# ---------------- (A) field contrast
out("(A) Field contrast — ΔT_N(H) = -½ Δχ H² / ΔS_t  (cgs: ½ΔχH² in erg/mol; 1 erg = 1e-7 J)")
dchis = {"Lacour-Gayet 1973": 2.0e-4, "Browne via Tsuchiya": 2.6e-4, "Wang 2023": 2.8e-4, "bound χ_AFM=0": 6.6e-4}
dSs = {"Chaussy 6.80": 6.80, "Browne 8.58": 8.58}
for H_T in (1, 2, 5, 10):
    H = H_T * 1e4  # Oe
    row = []
    for dk, dchi in dchis.items():
        vals = [-0.5 * dchi * H**2 * 1e-7 / dS * 1e3 for dS in dSs.values()]   # mK
        row.append(f"{dk}: {vals[0]:.2f}/{vals[1]:.2f} mK")
    out(f"  H = {H_T:2d} T  ->  " + "; ".join(row) + "   (Chaussy/Browne ΔS_t)")
W = 5.3
for dk in ("Browne via Tsuchiya", "bound χ_AFM=0"):
    dT5 = 0.5 * dchis[dk] * (5e4)**2 * 1e-7 / 6.80
    out(f"  first-application ΔS_M at 5 T ≤ ΔS_t·|ΔT_N|/W = {6.80/M*dT5/W:.3f} J/kg/K ({dk}, W = {W} K FWHM)")
for h in (2, 3, 5):
    for dk in ("Lacour-Gayet 1973", "Wang 2023"):
        Hreq = np.sqrt(2 * h * 6.80e7 / dchis[dk])   # Oe; ΔS in erg/mol/K
        out(f"  field for |ΔT_N| = h = {h} K ({dk}, ΔS_t 6.80): {Hreq/1e4:.0f} T")
out("")

# ---------------- (B) uniaxial stress
out("(B) Uniaxial stress along cubic <110>, Clausius–Clapeyron: dT_N/dσ = V_m·|Δε|/ΔS_t")
for de in (0.0125, 0.0140, 0.0160):
    out(f"  |Δε| = {de*100:.2f} %: dT_N/dσ = {VM*de/6.80*1e9:.1f} K/GPa (Chaussy ΔS_t) | {VM*de/8.58*1e9:.1f} K/GPa (Browne ΔS_t)"
        f"   vs hydrostatic 7.4-9 K/GPa")
T, Cex, Cbg, Tpk = s3.chaussy_shape()
TG = s3.TG
SG = np.round(np.arange(0.0, 1.2001, 0.01), 3)        # stress grid (GPa)
S_DT = (0.2, 0.3, 0.5, 0.75, 1.0)
N = 1000
sets = {"Chaussy": dict(dS=(6.30, 7.00), shift=0.0), "Browne": dict(dS=(8.20, 8.90), shift=287.0 - Tpk)}
bands = {}
for name, st in sets.items():
    F, DSR, DTR, ON, P50, AS = [], [], [], [], [], []
    for _ in range(N):
        dS = RNG.uniform(*st["dS"]); de = RNG.uniform(0.0125, 0.016); a = VM * de / dS * 1e9
        h = RNG.uniform(3.0, 5.0); sig = RNG.uniform(0.0, 1.5); aV = RNG.uniform(0.9, 1.1)
        Cx = np.interp(T - st["shift"], T, Cex, left=0.0, right=0.0)
        b = Baro(T, Cx, Cbg, a=a, h=h, v=s3.V_kg, alphaV=lambda TT, aV=aV: aV * (s3.A0 + s3.A1 * TT) / 3.0,
                 dS_target=dS / M, sigma_broad=sig)
        f, dSr, dS1, dT = b.summary(TG, SG, S_DT)
        F.append(f); DSR.append(dSr); DTR.append([dT[p] for p in S_DT]); AS.append(a)
        ON.append(first_crossing(SG, f, 0.01)); P50.append(first_crossing(SG, f, 0.5))
    q = lambda x: np.nanpercentile(np.array(x), [2.5, 16, 50, 84, 97.5], axis=0)
    bands[name] = dict(DSR=q(DSR), DTR=q(DTR), ON=q(ON), P50=q(P50), A=q(AS))
    B = bands[name]
    out(f"== {name} set, uniaxial (N = {N}; |Δε| 1.25-1.60 %, h 3-5 K, σ_TN 0-1.5 K, linear α = α_V/3 ±10 %)")
    out(f"   dT_N/dσ [2.5,16,50,84,97.5]% = " + ", ".join(f"{x:.1f}" for x in B["A"]) + " K/GPa")
    out(f"   onset = {B['ON'][2]:.2f} GPa (68 %: {B['ON'][1]:.2f}-{B['ON'][3]:.2f}); σ50 = {B['P50'][2]:.2f} GPa (68 %: {B['P50'][1]:.2f}-{B['P50'][3]:.2f})")
    for p in S_DT:
        i = np.argmin(abs(SG - p)); j = S_DT.index(p)
        out(f"   σ = {p:4.2f} GPa: |ΔS_rev| median {B['DSR'][2, i]:5.1f} J/kg/K (68 %: {B['DSR'][1, i]:.1f}-{B['DSR'][3, i]:.1f}); "
            f"ΔT_rev median {B['DTR'][2, j]:.1f} K (68 %: {B['DTR'][1, j]:.1f}-{B['DTR'][3, j]:.1f})")
out("")
out("CAVEATS: single-domain/textured sample assumed (polycrystal averaging lowers dT_N/dσ); the <110> mapping of")
out("the a/b axes comes from the 88.3° distortion geometry plus Browne's stress-cooling domain selection; the")
out("mechanical strength and fatigue of bulk CrN under ~0.3-0.5 GPa cyclic stress are unverified. Model-free")
out("Clausius–Clapeyron only: no microscopic model is used.")
(ROOT / "results/step6_field_uniaxial.txt").write_text("\n".join(lines) + "\n")

# Figure: uniaxial vs hydrostatic (hydrostatic wide-prior median band recomputed with small N for overlay).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
PG = s3.PG
hyd = []
for _ in range(400):
    dS = RNG.uniform(6.30, 7.00); a0, bc = s3.sample_slope(9.0, dict(kind="wide")); h = RNG.uniform(3, 5)
    bb = s3.make(T, Cex, Cbg, 0.0, dS, a0, h, RNG.uniform(0, 1.5), RNG.uniform(0.9, 1.1), b=bc)
    f, dSr, _, _ = bb.summary(TG, PG, (1.0,))
    hyd.append(dSr)
H = np.nanpercentile(np.array(hyd), [16, 50, 84], axis=0)
fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=110)
ax.fill_between(PG, H[0], H[2], color="tab:blue", alpha=0.3); ax.plot(PG, H[1], color="tab:blue", label="hydrostatic P (Chaussy set)")
B = bands["Chaussy"]["DSR"]
ax.fill_between(SG, B[1], B[3], color="tab:red", alpha=0.3); ax.plot(SG, B[2], color="tab:red", label="uniaxial σ along ⟨110⟩ (Chaussy set, CC estimate)")
ax.axvline(0.6, color="gray", ls=":", lw=1)
ax.set_xlim(0, 2.0); ax.set_xlabel("applied pressure / uniaxial stress (GPa)"); ax.set_ylabel("peak reversible |ΔS| (J kg⁻¹ K⁻¹)")
ax.set_title("CrN: the shear channel is ~2-3x more efficient than hydrostatic pressure (h = 3-5 K)", fontsize=8)
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(ROOT / "figures/step6_uniaxial_vs_hydrostatic.png"); plt.close(fig)
