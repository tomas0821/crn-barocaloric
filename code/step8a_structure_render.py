"""Render rock-salt CrN in its double-layer antiferromagnetic state with OVITO (Tachyon, offscreen).

Run with an environment that has the `ovito` Python module (pip install ovito); the panel layout is
done by step8_structure_figure.py, which needs only Matplotlib.

Structure: Cr and N on interpenetrating fcc lattices (a = 4.148 A, classic samples, table 1). The
Cr moments are parallel within (110) planes, which are stacked up-up-down-down along [110]
(Corliss et al. 1960). The block is built in the frame of the magnetic order ([110], [1-10], [001]); the (110)
plane index is L = i + j, and the moment is "up" for L mod 4 in {0, 1}.
Output: figures/step8_structure_render.png (white background).

Headless rendering (WSL, no GPU) needs OVITO's own Qt platform and Mesa's software Vulkan driver, e.g.
  env -u DISPLAY QT_QPA_PLATFORM=ovitoheadless VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
      LD_LIBRARY_PATH=<ovito venv>/extra/<dir with libOpenGL.so.0> python -P code/step8a_structure_render.py
"""
import json
import sys
from pathlib import Path
import numpy as np

from ovito.data import DataCollection, Particles, SimulationCell, ParticleType
from ovito.pipeline import Pipeline, StaticSource
from ovito.modifiers import CreateBondsModifier
from ovito.vis import Viewport, TachyonRenderer, BondsVis

ROOT = Path(__file__).resolve().parents[1]
A = 4.148                     # cubic lattice parameter (Angstrom)

# colours shared with step8_structure_figure.py
COL = {"Cr_up": (0.902, 0.624, 0.0), "Cr_dn": (0.0, 0.447, 0.698), "N": (0.62, 0.62, 0.62)}
RAD = {"Cr_up": 0.40, "Cr_dn": 0.40, "N": 0.22}

# Build the block in the frame of the magnetic order: u = [110] (stacking), v = [1-10], w = [001].
# Site (i, j, k) of the cubic rock-salt lattice (units of a/2) sits at u = (i+j) d, v = (i-j) d, w = k a/2,
# with d = a/(2 sqrt 2) the (110) plane spacing. Shown: u planes L = 0..8 (two magnetic periods),
# v rows 0..2 (one repeat), w layers 0..2 (one cubic cell).
D = A / (2 * np.sqrt(2))
NL, NV, NW = 8, 2, 2
pos, typ = [], []
for L in range(NL + 1):
    for m in range(NV + 1):
        if (L + m) % 2:                      # i = (L+m)/2 and j = (L-m)/2 must be integers
            continue
        for k in range(NW + 1):
            pos.append((L * D, m * D, k * A / 2))
            if (L + k) % 2 == 0:             # i + j + k even -> Cr
                typ.append(1 if L % 4 in (0, 1) else 2)
            else:
                typ.append(3)
pos = np.array(pos)

data = DataCollection()
cell = data.create_cell(np.diag([NL * D, NV * D, NW * A / 2]), pbc=(False, False, False))
cell.vis.line_width = 0.04
cell.vis.rendering_color = (0.35, 0.35, 0.35)
parts = data.create_particles(count=len(pos))
parts.create_property("Position", data=pos)
tprop = parts.create_property("Particle Type", data=np.array(typ))
for tid, name in [(1, "Cr_up"), (2, "Cr_dn"), (3, "N")]:
    tprop.types.append(ParticleType(id=tid, name=name, color=COL[name], radius=RAD[name]))

pipe = Pipeline(source=StaticSource(data=data))
bonds = CreateBondsModifier(mode=CreateBondsModifier.Mode.Uniform, cutoff=0.55 * A)
bonds.vis.width = 0.09
bonds.vis.coloring_mode = BondsVis.ColoringMode.Uniform
bonds.vis.color = (0.80, 0.80, 0.80)
pipe.modifiers.append(bonds)
pipe.add_to_scene()

# View almost along v = [1-10], so the (110) planes appear edge-on as slabs (up-up-down-down along u),
# with a small rotation and tilt for depth. Optional CLI override: camera_dir x y z, output suffix.
CAM = tuple(float(v) for v in sys.argv[1:4]) if len(sys.argv) >= 4 else (0.30, 1.0, -0.35)
vp = Viewport(type=Viewport.Type.Ortho, camera_dir=CAM)
vp.zoom_all()
vp.fov *= 1.15                # margin so the axis tripod clears the atoms
# The axis key is drawn by step8_structure_figure.py from the same camera direction (saved below).

out = ROOT / f"figures/step8_structure_render{sys.argv[4] if len(sys.argv) >= 5 else ''}.png"
vp.render_image(size=(1800, 1500), filename=str(out), background=(1, 1, 1), alpha=False,
                renderer=TachyonRenderer(ambient_occlusion=True, antialiasing_samples=16, shadows=False))
json.dump({"camera_dir": CAM, "frame": ["[110]", "[1-10]", "[001]"]},
          open(out.with_suffix(".json"), "w"))
print(f"{len(pos)} sites ({typ.count(1)} Cr up, {typ.count(2)} Cr down, {typ.count(3)} N) -> {out}")
