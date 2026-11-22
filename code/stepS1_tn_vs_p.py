"""Supplementary figure S1: measured shifts of T_N under pressure against the Monte-Carlo slope prior.

Plotted as ΔT_N(P) = T_N(P) - T_N(0), so that sources with different T_N(0) can be compared.
  - Browne et al. 1971 (classic, CrN0.997): 0.735 K/kbar (warming), 0.747 K/kbar (cooling), to 0.4 GPa.
  - Chaussy et al. 1971: 0.9 K/kbar, quoted without a pressure range (drawn to 0.4 GPa, dashed beyond).
  - Rivadulla et al. 2009 as replotted by Shulumba et al. 2014: ≈7.8 K/GPa over 0.25-1.78 GPa.
  - High-pressure-synthesized family (T_N(0) ≈ 240 K; Wang 2012, Yan 2020): ≈11-14 K/GPa (excluded).
  - Prior (step 3, wide): a0 = 9.0 ±10 %, b in [-0.8, 1.2] K/GPa², mean slope over 0-1.8 GPa in [6.5, 11].
"""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import step3_montecarlo as s3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7})
P = np.linspace(0, 1.8, 181)
draws = np.array([s3.sample_slope(9.0, dict(kind="wide")) for _ in range(3000)])
band = np.percentile(draws[:, :1] * P + draws[:, 1:] * P**2, [2.5, 50, 97.5], axis=0)
fig, ax = plt.subplots(figsize=(3.35, 2.6))
ax.fill_between(P, band[0], band[2], color="tab:blue", alpha=0.2, lw=0, label="slope prior, Chaussy set (95 %)")
ax.plot(P, band[1], color="tab:blue", lw=0.8)
p4 = P[P <= 0.4]
ax.plot(p4, 7.35 * p4, "-", color="tab:orange", lw=1.4, label="Browne 1971 (warming / cooling)")
ax.plot(p4, 7.47 * p4, "-", color="tab:orange", lw=0.8)
ax.plot(p4, 9.0 * p4, "-", color="k", lw=1.4, label="Chaussy 1971 (quoted)")
ax.plot(P[P >= 0.4], 9.0 * P[P >= 0.4], "--", color="k", lw=0.8)
pr = P[(P >= 0.25) & (P <= 1.78)]
ax.plot(pr, 7.8 * pr, "-", color="tab:green", lw=1.4, label="Rivadulla 2009 (replot)")
ax.plot(P, 11 * P, ":", color="gray", lw=1.0, label=r"HP-synthesized, 11–14 K GPa$^{-1}$")
ax.plot(P, 14 * P, ":", color="gray", lw=1.0)
ax.set_xlabel("pressure (GPa)"); ax.set_ylabel(r"$\Delta T_\mathrm{N}$ (K)")
ax.set_xlim(0, 1.8); ax.set_ylim(0, 25); ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(ROOT / f"figures/stepS1_tn_vs_p.{ext}", dpi=300)
lines = ["Supplementary figure S1: measured T_N(P) shifts vs the Monte-Carlo slope prior (step 3, wide)."]
for name, a_meas in (("Chaussy", 9.0), ("Browne", 7.41)):
    d = draws if name == "Chaussy" else np.array([s3.sample_slope(a_meas, dict(kind="wide")) for _ in range(3000)])
    lines.append(f"{name} prior: mean slope over 0-1.8 GPa, 2.5/50/97.5 % = "
                 + ", ".join(f"{v:.2f}" for v in np.percentile(d[:, 0] + 1.8 * d[:, 1], [2.5, 50, 97.5]))
                 + f" K/GPa; mean curvature b = {d[:, 1].mean():+.2f} K/GPa^2")
(ROOT / "results/stepS1_tn_vs_p.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
