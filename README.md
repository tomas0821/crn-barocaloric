# CrN barocaloric analysis: data and code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24441835.svg)](https://doi.org/10.5281/zenodo.24441835)

Data and analysis code for

> T. Rojas and R. Thorat, *Barocaloric response of CrN: a large transition entropy behind a weak
> volume channel* (submitted to *Physica Scripta*).

Every number and figure in the paper can be regenerated from this repository with plain Python
(NumPy, SciPy, Matplotlib). No first-principles calculations are involved. All inputs are
published data.

## Contents

| Path | What it is |
|---|---|
| `code/baro.py` | Quasi-direct barocaloric core (either sign of dT/dP, branch-specific slopes, slope curvature) |
| `code/step1_digitize_chaussy.py` | Digitization of the CrN specific heat (Chaussy et al. 1971, Fig. 3) |
| `code/step1_validate_cp.py` | Integration of the anomaly over several backgrounds; latent heat, ΔS_t, width, background C_p |
| `code/step1_cc_check.py` | Clausius–Clapeyron consistency of the input pairings |
| `code/step2_quasidirect.py` | Reversible ΔS(T,P) and ΔT_ad(T,P); onset / P50 / P90 against hysteresis |
| `code/step3_montecarlo.py` | Monte-Carlo uncertainty propagation (per source set) |
| `code/step4_validation.py` | Held-out test on FeRh and Mn3NiN (pre-specified criteria stated in the docstring) |
| `code/step4_posthoc.py` | Post-hoc diagnostics of the validation |
| `code/step4_crossbranch_posthoc.py` | Post-hoc check: the cross-branch construction scored on FeRh; overlap vs cross-branch for CrN |
| `code/step5_interpretation.py` | Mean-field coupling decomposition and entropy bookkeeping |
| `code/step6_field_uniaxial.py` | Magnetic-field contrast and Clausius–Clapeyron uniaxial-stress estimate |
| `code/step7_design_map.py` | Cross-material comparison (benchmark table and design map) |
| `code/stepS1_tn_vs_p.py` | Supplementary figure: measured T_N(P) shifts against the Monte-Carlo slope prior |
| `code/figstyle.py` | Shared figure style (printed size, vector PDF output) |
| `data/inputs_by_source.csv` | All CrN inputs with source, anchor and kind (measured / derived / digitized) |
| `data/digitized/` | Digitized CrN specific heat (raw trace and uniform grid) |
| `data/benchmarks/barocaloric_benchmarks.csv` | Benchmark data for 11 barocaloric materials, with DOIs, anchors and verification flags |
| `data/validation/` | Curves extracted from the vector graphics of the FeRh and Mn3NiN publications |
| `results/` | Text output of every script (the numbers quoted in the paper) |
| `figures/` | Figures produced by the scripts |

## Reproducing

```bash
uv venv .venv && . .venv/bin/activate && uv pip install -r requirements.txt
python code/step1_validate_cp.py      # step 1a needs the original scan (see below)
python code/step1_cc_check.py
python code/step2_quasidirect.py
python code/step3_montecarlo.py
python code/step4_validation.py
python code/step4_posthoc.py
python code/step4_crossbranch_posthoc.py
python code/step5_interpretation.py
python code/step6_field_uniaxial.py
python code/step7_design_map.py
python code/stepS1_tn_vs_p.py
```

The scripts are light: seconds to a few minutes each, and well under 1 GB of memory.

`step1_digitize_chaussy.py` needs a 300 dpi render of the original page, which is not
redistributed here. The paper is open access on HAL (jpa-00214382). Render it with

```bash
pdftoppm -r 300 -gray -f 3 -l 3 -png <HAL PDF> data/raw/chaussy_p978
```

and crop as described in the script. Its output, `data/digitized/chaussy1971_fig3_cp.csv`, is
included, so every later step runs without it.

## Data sources

Digitized and extracted data come from published figures. Please cite the original works when
reusing them:

- **CrN specific heat:** J. Chaussy, R. Georges, P. Mollard, J. Voiron, *J. Phys. Colloq.* **32**,
  C1-977 (1971), doi:10.1051/jphyscol:19711347.
- **FeRh barocaloric curves:** E. Stern-Taulats *et al.*, *Phys. Rev. B* **89**, 214105 (2014),
  doi:10.1103/PhysRevB.89.214105.
- **Mn3NiN barocaloric curves:** D. Boldrin *et al.*, *Phys. Rev. X* **8**, 041035 (2018),
  doi:10.1103/PhysRevX.8.041035.
- **Benchmark values:** the primary publications listed in the CSV.

Verification flags in the benchmark CSV:
- **S:** stated in the text or a table
- **F:** read from a figure
- **D:** derived, for example a density from lattice parameters
- **C:** compiled from a secondary table
- **n.v.:** not verified in the primary source

## Versions

- **v1.1.0:** reproduces every number and figure of the submitted manuscript.
- **v1.0.0:** an earlier state of the analysis, before the revision; superseded by v1.1.0.

The concept DOI in the badge above always resolves to the latest version.

## License

Code: MIT (see `LICENSE`). Data files in `data/` and `results/`: CC BY 4.0. Data taken from
published works remain subject to the citation requests above.

## How to cite

See `CITATION.cff`.
