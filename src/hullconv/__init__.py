"""Hull convergence of generative crystal structure models.

Tools to rebuild the convex hull of a chemical system as generated structures
are added to the Materials Project (MP) structures, and to measure when the
hull stops changing: stable candidates, total and mean convex-hull shift
(TCS, MCS), sample coverage, SISSO extrapolation of the candidate curve, the
holdout test, the stopping rule, and the DFT energy above the hull.

Submodules that need pymatgen, spglib or TorchSISSO import them on use, so the
package can be imported in an environment that has only one of them.
"""

__version__ = "1.0.0"
