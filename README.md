# Hull convergence of generative crystal structure models

Code for the paper *A stopping rule for completing computed phase diagrams with
generative models* (citation to be added). It rebuilds the convex hull of a
chemical system as MatterGen-generated, PET-OAM-XL-relaxed structures are added
to the Materials Project (MP) structures, and measures when the hull stops
changing.

The package `hullconv` computes:

| Quantity | Module |
|---|---|
| MP2020-corrected energies of MLIP-relaxed structures | `energies.py` |
| MP-only and updated hulls, stable candidates, TCS and MCS | `hull.py` |
| Permutation rarefaction curves | `rarefaction.py` |
| Species (formula + space group) and sample coverage | `coverage.py` |
| SISSO fits of the candidate curve | `sisso_fit.py` |
| Holdout test | `holdout.py` |
| Stopping rule | `stopping_rule.py` |
| DFT energy above the hull | `dft_ehull.py` |
| Materials Project download (optional) | `mp_data.py` |

## Installation

The code was tested in two environments: one for the hull analysis
(Python 3.11) and one for the SISSO fits (Python 3.9). The tested versions are
pinned in `requirements.txt` and `requirements-sisso.txt`.

```bash
# hull analysis
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# SISSO fits
python3.9 -m venv .venv-sisso && source .venv-sisso/bin/activate
pip install -r requirements-sisso.txt && pip install -e .
```

Modules import pymatgen or TorchSISSO only when needed, so each script runs in
the environment that has the libraries it uses (marked below).

## Data

Download the data set from Zenodo (DOI: [10.5281/zenodo.24073365](https://doi.org/10.5281/zenodo.24073365)) and unpack it into
`./data`, or point the environment variable `HULLCONV_DATA` (or the option
`--data`) to it. Files are found by name anywhere below the data directory.

| File | Content |
|---|---|
| `<TAG>_hull_data.csv` | MP and generated structures with formation energies (below) |
| `<TAG>_generated.extxyz` | generated structures |
| `<TAG>_relaxed_PET-OAM-XL.extxyz` | final relaxed structures with `gen_index`, `formula`, `energy` (eV), `energy_per_atom` (eV/atom), `fmax` (eV/Angstrom), `converged`, forces and stress; `n_steps` in the LIO and new LRO pool files |
| `<TAG>_holdout_subsamples.csv` | holdout subsamples: columns `subsample`, `mp_id` |
| `dft/dft_summary.csv` and `dft/<calculation>/` | DFT calculations: columns `calculation`, `system`, `role` (`candidate` or `competing`) |

`<TAG>` is a system (`LMO`, `LCO`, `LNO`, `LZO`, `LHO`, `LRO`, `LNMO`, `LNRO`,
`LIO`), the LRO and LNRO generations extended to 15,000 structures
(`LRO_15000`, `LNRO_15000`), or the independently generated LRO pool
(`LRO_independent`).

Columns of `<TAG>_hull_data.csv` (energies in eV/atom):

| Column | MP rows | Generated rows |
|---|---|---|
| `source` | `MP` | `generated` |
| `mp_id` | MP identifier | empty |
| `gen_index` | empty | generation index |
| `formula` | formula | formula |
| `n_atoms` | atoms in the cell | atoms in the cell |
| `formation_energy_eV_per_atom` | MP2020-corrected formation energy | PET-OAM-XL + MP2020 formation energy of the final relaxed structure |
| `e_above_hull_mp_only_eV_per_atom` | Ehull on the MP-only hull | Ehull on the MP-only hull (negative below it) |
| `e_above_hull_updated_eV_per_atom` | Ehull on the hull with all generated structures | Ehull on the hull with all generated structures |

The MP formation energies are those of Materials Project database release
v2025.09.25 (GGA/GGA+U), so the analysis does not need the MP API.
Generated rows are processed in `gen_index` order, which is the order in which
the structures were generated. ASE reads the `energy` key of an extxyz frame
into a single-point calculator; `hullconv.energies.frame_energy` reads it from
either place.

## Reproducing the results

All scripts take `--data`, `--out` (default `./results`) and `--systems`.
Scripts marked *SISSO* read only CSV files written by the other scripts and run
in the SISSO environment; all others run in the hull-analysis environment.

All results use `formation_energy_eV_per_atom`, the formation energy of the
final relaxed structure.

| Result | Command |
|---|---|
| Table 2 (stable candidates, TCS, MCS) | `python scripts/table2_hull_metrics.py` |
| Rarefaction curves (Figure 2) | `python scripts/rarefaction_curves.py` |
| Table 1 and coverage curves (Figure 4c) | `python scripts/coverage_table1.py` |
| Table 3 (convergence criteria), *SISSO* | `python scripts/table3_convergence.py` |
| Table S5 (SISSO fits, next candidate), *SISSO* | `python scripts/tableS5_sisso_extrapolation.py` |
| Holdout test (Figure 3) | `python scripts/holdout_test.py` |
| Stopping rule, curves | `python scripts/stopping_rule_test.py --stage curves --systems LMO LCO LNO LZO LHO LRO LIO` |
| Stopping rule, conditions, *SISSO* | `python scripts/stopping_rule_test.py --stage signals` |
| Table S3 (independent LRO pool) | `python scripts/tableS3_independent_pool.py` |
| Table S9 (DFT Ehull) | `python scripts/tableS9_dft_ehull.py` |

Order of execution: `rarefaction_curves.py` and `coverage_table1.py` before
`table3_convergence.py`; `rarefaction_curves.py --systems LRO_15000` before
`tableS5_sisso_extrapolation.py`; `coverage_table1.py --systems LMO LCO LNO LZO
LHO LRO LIO` before the `signals` stage of the stopping rule. The quaternary
systems use the same scripts, for example
`python scripts/rarefaction_curves.py --systems LNMO LNRO`.

To build a hull table for a new system from relaxed structures, set
`MP_API_KEY`, install `requirements-mp.txt` and run
`python scripts/build_hull_data.py --tag <TAG> --elements <El1> <El2> <El3> --relaxed <file>`.

## Definitions

- **MP-only hull**: built from the MP structures of the system (GGA/GGA+U,
  MP2020-corrected). **Updated hull**: built from the MP structures and the
  generated structures added so far.
- **Stable candidate**: a generated structure on the updated hull (within
  1e-6 eV/atom), counted once per reduced formula.
- **TCS**: sum over the MP structures of the change in their Ehull from the
  MP-only hull to the updated hull. **MCS**: TCS divided by the number of MP
  structures.
- **Rarefaction**: 50 orderings without replacement,
  `numpy.random.default_rng(i).permutation(N_gen)` for i = 0..49, hull rebuilt
  every 50 structures; curves are the mean with a band from the 2.5th to the
  97.5th percentile.
- **Sample coverage**: `C = 1 - f1/N`, species = reduced formula + space-group
  number from spglib at `symprec = 0.1` Angstrom; curves over 50 orderings
  (`default_rng(42 + i)`) every 100 structures.
- **SISSO**: TorchSISSO with operators {+, -, *, /, exp, ln, sqrt},
  `n_expansion = 3`, `n_term = 3`, `k = 30`, fitted to the mean candidate curve.
- **Holdout**: MP structures reduced to 70 (Li, the transition metal and O2
  kept), 100 subsamples, 16 orderings per subsample (`default_rng(1000 t)` for
  subsample t), hull rebuilt every 250 structures.
- **Stopping rule**: from 8,000 structures and every 1,000 after, using only
  the structures generated so far: coverage >= 0.90; TCS rises by at most
  1 meV/atom over the second half of the rarefaction curve (20 orderings,
  `default_rng(10000 + r)`); the candidate count exceeds the value predicted by
  a SISSO fit to the first half of the curve by at most 0.5. Stable candidates
  that reproduce an MP structure or that DFT places above the MP-only hull are
  removed first.
- **DFT Ehull**: candidate energy minus the lowest-energy combination of the
  competing phases at its composition, all calculated with the same settings
  and MP2020-corrected.

## Structure generation and relaxation

`generation/` contains the commands used to produce the structures; they are
not needed to reproduce the analysis.

- `generate_mattergen.sh`: MatterGen with the checkpoint
  `chemical_system_energy_above_hull`, conditioned on the chemical system and on
  an energy above the hull of 0.03 eV/atom, guidance factor 2.0.
- `relax_pet.py`: PET-OAM-XL v1.0.0, FIRE with a Frechet cell filter,
  `fmax = 0.01` eV/Angstrom, 1,000 steps and up to 4,000 further steps for
  structures not yet converged. Requirements in `generation/requirements.txt`.

## License

MIT (see `LICENSE`). `VasprunLike` in `energies.py` is adapted from MatterGen
(Microsoft Corporation, MIT License); its license notice is in `NOTICE`.
