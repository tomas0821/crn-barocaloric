"""Step 1a: digitize Chaussy et al. (1971) Fig. 3, the specific heat of CrN.

Source: J. Chaussy, R. Georges, P. Mollard, J. Voiron, J. Phys. Colloq. 32, C1-977 (1971),
doi:10.1051/jphyscol:19711347, Fig. 3 on p. C1-978 ("Chaleur specifique de CrN").
Axes: C_p in cal mol^-1 K^-1 (per mole CrN), 0..~76; T in degC, -200..+50.

Input:  data/raw/chaussy_fig3_crop.png, a 300 dpi grayscale crop of PDF page 3. Recreate it with
        pdftoppm -r 300 -gray -f 3 -l 3 -png pdfs/10_1051_jphyscol_19711347.pdf data/raw/chaussy_p978
        and the crop box in CROP below.
Output: data/digitized/chaussy1971_fig3_cp.csv  (T_C, T_K, Cp_cal_per_mol_K, Cp_J_per_mol_K, branch)
        figures/step1_chaussy_overlay.png        (digitized points over the scan, for visual QA)

Method: a dark-pixel mask, with gridlines, the frame, the 8.5 degC dash-dot marker and the text labels
masked out. Smooth parts are traced column by column. The near-vertical peak (T in [-2, 20] degC,
C_p > 14) is traced row by row: the left edge is the rising branch and the right edge the falling one.
Axis calibration is a linear fit through the detected gridline pixel positions.

Lightweight by design (~2 MB image, NumPy only): safe on the 5 GB WSL VM.
"""
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "data/raw/chaussy_fig3_crop.png"
OUT_CSV = ROOT / "data/digitized/chaussy1971_fig3_cp.csv"
OUT_PNG = ROOT / "figures/step1_chaussy_overlay.png"
CROP = "rows 2%-42%, cols 47%-99% of the 300 dpi page (see step1 README note)"

# Gridline pixel positions, detected by step1 line search (long dark runs) on the crop.
X_PX = np.array([147.0, 325.5, 502.5, 681.5, 858.5, 1034.5])   # frame -200, -150, -100, -50, 0, frame +50
X_T = np.array([-200.0, -150.0, -100.0, -50.0, 0.0, 50.0])
Y_PX = np.array([1109.5, 1001.5, 890.5, 780.5, 670.5, 562.5, 453.5, 343.0])  # 0 (frame), 10..70
Y_C = np.array([0.0, 10, 20, 30, 40, 50, 60, 70])
DASHDOT_PX = 888.5                                              # the 8.5 degC marker
TOP_FRAME_PX = 279.5

kx, x0 = np.polyfit(X_T, X_PX, 1)      # px = x0 + kx*T
ky, y0 = np.polyfit(Y_C, Y_PX, 1)      # px = y0 + ky*C   (ky < 0)
T_of = lambda px: (px - x0) / kx
C_of = lambda py: (py - y0) / ky
px_of_T = lambda T: x0 + kx * T
py_of_C = lambda C: y0 + ky * C


def main():
    a = np.asarray(Image.open(IMG).convert("L")).astype(np.int16)
    dark = a < 128
    H, W = dark.shape

    # Keep only the plot interior.
    keep = np.zeros_like(dark)
    keep[int(TOP_FRAME_PX) + 3: int(Y_PX[0]) - 2, int(X_PX[0]) + 3: int(X_PX[-1]) - 2] = True
    m = dark & keep
    # Mask gridlines (+-2 px). Curve gaps there are bridged by interpolation later.
    for yp in Y_PX[1:]:
        m[int(round(yp)) - 2: int(round(yp)) + 3, :] = False
    apex_rows = slice(0, int(py_of_C(66.0)))   # above C_p = 66 the curve apex coincides with the 8.5 degC marker
    for xp in list(X_PX[1:-1]) + [DASHDOT_PX]:
        c1, c2 = int(round(xp)) - 3, int(round(xp)) + 4         # +-3 px: the scan is slightly skewed
        if xp == DASHDOT_PX:
            m[apex_rows.stop:, c1:c2] = False                    # keep the apex rows unmasked
        else:
            m[:, c1:c2] = False
    # Mask the text labels (data coordinates -> pixel boxes).
    def mask_box(T1, T2, C1, C2):
        c1, c2 = int(px_of_T(T1)), int(px_of_T(T2))
        r1, r2 = int(py_of_C(C2)), int(py_of_C(C1))
        m[max(r1, 0):min(r2, H), max(c1, 0):min(c2, W)] = False
    mask_box(-200, -95, 70.5, 77)   # "Cp (cal/mole/K)"
    mask_box(-112, -50, 40.5, 49.5)  # "CrN"
    mask_box(9.5, 50, -1, 5.0)       # "8,5 deg" and "T(degC)"

    pts = []  # (T, C, branch)
    # (a) Column-wise trace outside the spike.
    for c in range(int(X_PX[0]) + 3, int(X_PX[-1]) - 2):
        T = T_of(c)
        rows = np.where(m[:, c])[0]
        if rows.size == 0:
            continue
        if -2.0 <= T <= 20.0:
            rows = rows[C_of(rows) <= 14.5]   # inside the spike window keep only the low part
            if rows.size == 0:
                continue
        # Group rows into clusters; keep the largest (the curve stroke plus data dots).
        splits = np.where(np.diff(rows) > 3)[0] + 1
        clusters = np.split(rows, splits)
        best = max(clusters, key=len)
        if len(best) < 2:
            continue
        branch = "low" if T < 0 else ("high" if T > 15 else "spike-base")
        pts.append((T, C_of(best.mean()), branch))
    # (b) Row-wise trace of the spike (C > 14).
    cL, cR = int(px_of_T(-2.0)), int(px_of_T(20.0))
    r_top, r_bot = int(TOP_FRAME_PX) + 3, int(py_of_C(14.0))
    for r in range(r_top, r_bot):
        cols = np.where(m[r, cL:cR])[0] + cL
        if cols.size == 0:
            continue
        splits = np.where(np.diff(cols) > 3)[0] + 1
        clusters = [g for g in np.split(cols, splits) if len(g) >= 1]
        C = C_of(r)
        left, right = clusters[0], clusters[-1]
        pts.append((T_of(left.mean()), C, "rise"))
        if len(clusters) > 1:
            pts.append((T_of(right.mean()), C, "fall"))

    pts = np.array([(p[0], p[1]) for p in pts]), [p[2] for p in pts]
    TC, CC = pts[0][:, 0], pts[0][:, 1]
    br = np.array(pts[1])

    # (c) Robust cleaning: drop isolated specks (scan dust, gridline remnants) that sit far from the
    # running median of their own branch. Spike edges are cleaned in T at fixed C; smooth parts in C
    # at fixed T.
    keep_pt = np.ones(TC.size, bool)
    def clean(sel, along, other, win, tol):
        idx = np.where(sel)[0]
        o = idx[np.argsort(along[idx])]
        for j, i in enumerate(o):
            lo, hi = max(0, j - win), min(len(o), j + win + 1)
            if abs(other[i] - np.median(other[o[lo:hi]])) > tol:
                keep_pt[i] = False
    for b in ("rise", "fall"):
        clean(br == b, CC, TC, win=8, tol=0.8)        # spike edges: T(C) within 0.8 K of local median
    for b in ("low", "spike-base", "high"):
        clean(br == b, TC, CC, win=8, tol=0.8)        # smooth parts: C(T) within 0.8 cal of local median
    n_drop = int((~keep_pt).sum())
    TC, CC, br = TC[keep_pt], CC[keep_pt], br[keep_pt]
    print(f"cleaning dropped {n_drop} isolated points")

    order = np.argsort(TC, kind="stable")
    TC, CC, br = TC[order], CC[order], br[order]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w") as f:
        f.write("# Chaussy et al. 1971 Fig. 3 (p. C1-978), digitized by code/step1_digitize_chaussy.py\n")
        f.write(f"# calibration: px = {x0:.3f} + {kx:.5f}*T_C ; py = {y0:.3f} + {ky:.5f}*Cp_cal ; 1 px ~ {1/kx:.3f} K, {1/abs(ky):.3f} cal/mol/K\n")
        f.write("T_C,T_K,Cp_cal_per_mol_K,Cp_J_per_mol_K,branch\n")
        for T, C, b in zip(TC, CC, br):
            f.write(f"{T:.3f},{T+273.15:.3f},{C:.3f},{4.184*C:.3f},{b}\n")
    print(f"wrote {OUT_CSV.relative_to(ROOT)}: {len(TC)} points; peak Cp = {CC.max():.1f} cal/mol/K at {TC[CC.argmax()]:.2f} degC")

    # Overlay for visual QA.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7.6), dpi=110)
    ax.imshow(a, cmap="gray", vmin=0, vmax=255)
    colors = {"low": "tab:blue", "spike-base": "tab:purple", "rise": "tab:orange", "fall": "tab:green", "high": "tab:red"}
    for b, col in colors.items():
        s = br == b
        ax.plot(px_of_T(TC[s]), py_of_C(CC[s]), ".", ms=1.6, color=col, label=b)
    ax.set_xlim(X_PX[0] - 20, X_PX[-1] + 20); ax.set_ylim(Y_PX[0] + 20, TOP_FRAME_PX - 20)
    ax.legend(loc="upper left", fontsize=7, markerscale=5); ax.set_axis_off()
    fig.tight_layout(); fig.savefig(OUT_PNG); plt.close(fig)
    print(f"wrote {OUT_PNG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
