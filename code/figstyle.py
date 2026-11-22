"""Shared figure style: drawn at the printed size (full text width 6.3 in), 8 pt text, vector PDF output."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FULL, HALF = 6.3, 3.35          # inches: full text width and one column
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "legend.frameon": False, "lines.linewidth": 1.1, "axes.linewidth": 0.7,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 300, "pdf.fonttype": 42,
    "mathtext.default": "regular",
})

def panel(ax, letter, x=-0.16, y=1.02):
    """Bold (a), (b), ... label outside the top-left corner of the axes."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontweight="bold", va="bottom", ha="left")

def save(fig, path_without_ext):
    """Save PDF (for the manuscript) and PNG (for quick viewing)."""
    fig.savefig(f"{path_without_ext}.pdf")
    fig.savefig(f"{path_without_ext}.png", dpi=300)
