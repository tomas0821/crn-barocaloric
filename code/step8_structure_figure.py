"""Structure figure: (a) the OVITO render of CrN in its double-layer antiferromagnetic state
(code/step8a_structure_render.py), (b) a schematic of the transformation strain on ordering.

Strains on ordering are Browne et al.'s (table 1, signs reversed from their heating values):
the face diagonal that becomes the orthorhombic a axis contracts by 1.6 %, the one that becomes b
expands by 1.3 %, and c (along the cube axis) contracts by 0.25 %, so the volume changes by only
about -0.55 %. The schematic exaggerates the in-plane strain sixfold.
Output: figures/step8_structure.{pdf,png}.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, FancyArrowPatch
import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
COL = {"up": (0.902, 0.624, 0.0), "dn": (0.0, 0.447, 0.698), "N": (0.62, 0.62, 0.62)}   # as in step8a
EPS_A, EPS_B, EPS_C = -0.016, +0.013, -0.0025     # on ordering (Browne et al. 1971)
X = 6.0                                            # exaggeration of the in-plane strain

# ---------------------------------------------------------------- panel (a): trimmed render
img = plt.imread(ROOT / "figures/step8_structure_render.png")[..., :3]
ink = np.where(img.min(axis=2) < 0.97)
r0, r1, c0, c1 = ink[0].min(), ink[0].max(), ink[1].min(), ink[1].max()
pad = 12
img = img[max(r0 - pad, 0):r1 + pad, max(c0 - pad, 0):c1 + pad]
img = np.vstack([img, np.ones((int(0.20 * img.shape[0]), img.shape[1], 3))])   # room for the axis key

fig = plt.figure(figsize=(fs.FULL, 2.55))
gs = fig.add_gridspec(1, 2, width_ratios=[1.62, 1.0], wspace=0.08)
ax = fig.add_subplot(gs[0]); ax.imshow(img, interpolation="none"); ax.set_axis_off()
fs.panel(ax, "a", x=0.0, y=1.0)
# axis key: orthographic projection with OVITO's convention (screen right = d x z, screen up = right x d)
cam = np.array(json.load(open(ROOT / "figures/step8_structure_render.json"))["camera_dir"], float)
d = cam / np.linalg.norm(cam); rgt = np.cross(d, [0, 0, 1.0]); rgt /= np.linalg.norm(rgt); upv = np.cross(rgt, d)
o = np.array([0.075, 0.018]); L = 0.062
asp = np.array([1.0, img.shape[1] / img.shape[0]])        # axes-fraction units: x and y lengths differ
for e, lab in [((1, 0, 0), "[110]"), ((0, 1, 0), r"$[1\bar{1}0]$"), ((0, 0, 1), "[001]")]:
    v = np.array([rgt @ e, upv @ e]) * L * asp
    ax.annotate("", xy=o + v, xytext=o, xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.2", mutation_scale=7, shrinkA=0, shrinkB=0))
    u = v / np.linalg.norm(v)
    ax.text(*(o + v + u * 0.012 * asp), lab, transform=ax.transAxes, fontsize=6.5, color="0.2",
            ha="left" if u[0] > 0.5 else ("right" if u[0] < -0.2 else "center"),
            va="bottom" if u[1] > 0.5 else "center")

handles = [Line2D([], [], ls="", marker="o", ms=6.5, mfc=COL["up"], mec="none", label="Cr, moment up"),
           Line2D([], [], ls="", marker="o", ms=6.5, mfc=COL["dn"], mec="none", label="Cr, moment down"),
           Line2D([], [], ls="", marker="o", ms=4.0, mfc=COL["N"], mec="none", label="N")]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=3, handletextpad=0.2,
          columnspacing=1.0)

# ---------------------------------------------------------------- panel (b): strain schematic, view along [001]
# Drawn in the frame of the two face diagonals: a (contracts) horizontal, b (expands) vertical, so the
# cubic face is a diamond. R maps cube-axis vectors into this frame.
bx = fig.add_subplot(gs[1]); bx.set_aspect("equal"); bx.set_axis_off()
fs.panel(bx, "b", x=0.0, y=1.0)
R = np.array([[1.0, 1.0], [-1.0, 1.0]]) / np.sqrt(2)
h = np.sqrt(2) / 2                                # half-diagonal of the unit cubic face
cub = np.array([[h, 0], [0, h], [-h, 0], [0, -h]])
ha, hb = h * (1 + X * EPS_A), h * (1 + X * EPS_B)
ort = np.array([[ha, 0], [0, hb], [-ha, 0], [0, -hb]])
FILL = (0.86, 0.90, 0.95)
bx.add_patch(Polygon(ort, closed=True, fc=FILL, ec="0.15", lw=1.0))
bx.add_patch(Polygon(cub, closed=True, fill=False, ls=(0, (3, 2)), lw=0.9, ec="0.45"))
arrow = dict(arrowstyle="<->", lw=0.8, color="0.15", shrinkA=0, shrinkB=0)
bx.annotate("", xy=(ha, 0), xytext=(-ha, 0), arrowprops=arrow)
bx.annotate("", xy=(0, hb), xytext=(0, -hb), arrowprops=arrow)
fmt = lambda e: f"{100 * e:+.1f}%".replace("-", "\u2212")
bx.text(0.95, -0.06, f"$a$  {fmt(EPS_A)}", ha="center", va="top")
bx.text(0.06, hb + 0.03, f"$b$  {fmt(EPS_B)}", ha="left", va="bottom")
for sgn in (+1, -1):                              # uniaxial compression along a
    bx.add_patch(FancyArrowPatch((sgn * 1.18, 0), (sgn * (ha + 0.04), 0), arrowstyle="-|>",
                                 mutation_scale=9, lw=1.4, color="0.1"))
bx.text(0.95, 0.05, r"$\sigma$", ha="center", va="bottom")
bx.text(-0.95, 0.05, r"$\sigma$", ha="center", va="bottom")
bx.text(-1.22, 1.00, "view along [001]", ha="left", va="center", fontsize=7)
bx.text(-1.22, -0.86, "$c$: \u22120.25%\n$\\Delta V/V \\approx$ \u22120.6%\nin-plane strain \u00d76",
        ha="left", va="center", fontsize=7, linespacing=1.3)
leg = [Line2D([], [], ls=(0, (3, 2)), color="0.45", lw=0.9, label="cubic (paramagnet)"),
       Polygon([[0, 0]], closed=True, fc=FILL, ec="0.15", lw=1.0, label="orthorhombic (ordered)")]
bx.legend(handles=leg, loc="lower right", bbox_to_anchor=(1.03, -0.02), fontsize=7, handlelength=1.6,
          borderaxespad=0.0)
o = np.array([0.80, 0.80])                         # cube-axis key, rotated into the drawing frame
for v, lab in [((1, 0), "[100]"), ((0, 1), "[010]")]:
    w = R @ np.array(v) * 0.20
    bx.annotate("", xy=o + w, xytext=o, arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.3",
                                                        mutation_scale=6, shrinkA=0, shrinkB=0))
    bx.text(*(o + w * 1.25), lab, fontsize=6, ha="left", va="center", color="0.3")
bx.set_xlim(-1.25, 1.25); bx.set_ylim(-1.12, 1.08)

fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02)
fs.save(fig, ROOT / "figures/step8_structure"); plt.close(fig)
print("figures/step8_structure.{pdf,png} written; in-plane strain x", X,
      f"| volume change on ordering {100 * (EPS_A + EPS_B + EPS_C):+.2f} %")
