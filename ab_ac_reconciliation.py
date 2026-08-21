"""Reconcile numeric conductance of the bare (no-QPC) ring against the analytic
Buttiker-type formula in ab_ac_proof.py, saving every attempt and retrying with
a different numerical approach whenever the two disagree by more than 1%.

Why this module exists
-----------------------
ab_ac_proof.py already computes both G_numeric (time-domain wavepacket + CAP)
and G_analytical (thesis eq. 3.2) for the transparent ring, but it does not
save a per-point audit trail, does not check a tolerance, and does not retry
with a different method when the two disagree. This module adds exactly that
loop, and folds in three bugs/physics issues found while building it (see the
commit history on this branch for the order they were found in):

1. `run_cap_conductance` (conductance.py) integrated the CAP absorption rate
   without the spatial quadrature weights used everywhere else in that module,
   so T+R came out at ~0.37 instead of ~1 even when probability was actually
   being absorbed correctly. Fixed.

2. The CAP/wavepacket method injects a Gaussian packet with a real energy
   spread. Because theta = k_F*pi*R ~ 12.6*pi for this ring (many
   wavelengths around the circumference), the analytic curve oscillates on
   an energy scale finer than that spread, so *any* wavepacket run
   systematically damps resonance peaks relative to a single-energy formula
   -- this is not a bug, it is a real limitation of comparing a wavepacket
   simulation to a monochromatic analytic prediction. Addressed by adding
   `run_exact_transmission` (conductance.py): an energy-domain Green's
   function / Fisher-Lee calculation (Datta, "Electronic Transport in
   Mesoscopic Systems", 1995, Ch. 3) that attaches analytic self-energies for
   semi-infinite 1-D leads and solves for T(E) at one sharp energy with a
   single sparse factorization -- no packet, no CAP tuning, no echo window,
   and about 100x faster per point than the CAP run.

3. Even with (2), a large, *structured* disagreement with the analytic
   formula remained (e.g. T ~ 1.0-1.3 vs G_analytical ~ 1.7 at Phi=0,
   alpha=0). Measuring the actual junction scattering matrix of this graph
   directly (see `measure_junction_scattering` below) shows why: the
   3-lead Y-junction built by tools.py realizes the "democratic"/Kirchhoff
   vertex condition (reflection r ~= -1/3, transmission t ~= 2/3 into each
   arm -- the unique unitary, fully-symmetric solution for 3 equal 1-D
   chains meeting at a point; see Kottos & Smilansky, PRL 79, 4794 (1997)
   for this vertex condition on quantum graphs). The analytic formula
   currently in ab_ac_proof.py, however, implicitly assumes an idealised,
   critically-coupled junction that reaches G_max = 2*G0 (full constructive
   interference) -- see Buttiker, Imry & Azbel, Phys. Rev. A 30, 1982
   (1984), where this is exactly the special case reached only for one
   particular value of their coupling parameter epsilon. A Kirchhoff
   junction and a critically-coupled junction are physically different
   3-port scatterers, so no amount of numerical refinement of the ring
   simulation will bring the two into agreement -- the analytic reference
   needs to be re-derived for epsilon corresponding to r=-1/3, t=2/3 (Xia,
   PRB 45, 3593 (1992); Texier & Montambaux, J. Phys. A 34, 10307 (2001)
   give the general network/graph machinery for this).

   `tools.py`'s own `junction_correction` option, meant to improve
   transmission by matching on-site energies, was also found (via the same
   junction-scattering measurement) to move the junction further from the
   Kirchhoff condition (|r| rises from ~0.34 to ~0.81), i.e. it makes each
   junction MORE reflective, not less. It is measured here for the record
   but not treated as the fix.

Because of (3), this module's retry loop will generally NOT reach 1%
agreement with the existing analytic formula for most (Phi, alpha) -- that
is an honest, evidenced finding, not a bug to keep patching. Every attempt is
still saved so the disagreement is fully auditable, and the closest-to-date
alternative reference (`analytical_G_kirchhoff_ring`, a from-scratch network
cascade using the *measured* junction S-matrix, spinless/alpha=0 only for
now) is included for comparison since it tracks the exact solver much more
closely than the legacy formula.

Run this module
----------------
    python ab_ac_reconciliation.py             # default grid, all methods
    python ab_ac_reconciliation.py --quick      # small grid, fast smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

import tools as t
import ab_ac_proof as legacy
from conductance import (
    run_exact_transmission,
    lead_surface_self_energy,
)

OUTPUT_DIR = Path(__file__).parent / "ab_ac_reconciliation"
OUTPUT_DIR.mkdir(exist_ok=True)

TOLERANCE_PCT = 1.0   # user-specified acceptance threshold


# ---------------------------------------------------------------------------
# Numerical methods to try, in order (cheapest / most-trusted first)
# ---------------------------------------------------------------------------

def method_exact_gf_democratic(Phi: float, alpha: float, E_F: float) -> float:
    """Exact energy-domain transmission, junction_correction disabled.

    Uses the graph's natural (uncorrected) on-site energies, which measure
    to the Kirchhoff/democratic Y-junction (r ~= -1/3, t ~= 2/3). This is the
    method most consistent with the actual physics realised by the lattice.
    """
    p = legacy.transparent_ring_params(Phi=Phi, alpha=alpha).with_changes(
        junction_correction=False)
    return run_exact_transmission(p, fermi_energy_mev=E_F, verbose=False).T


def method_exact_gf_corrected(Phi: float, alpha: float, E_F: float) -> float:
    """Exact energy-domain transmission, junction_correction enabled (default)."""
    p = legacy.transparent_ring_params(Phi=Phi, alpha=alpha).with_changes(
        junction_correction=True)
    return run_exact_transmission(p, fermi_energy_mev=E_F, verbose=False).T


def method_cap_wavepacket(Phi: float, alpha: float, E_F: float) -> float:
    """Time-domain CAP wavepacket method (legacy, bug-fixed integration weights)."""
    return legacy.simulate_G(Phi, alpha, verbose=False)


METHODS: list[tuple[str, Callable[[float, float, float], float]]] = [
    ("exact_gf_democratic", method_exact_gf_democratic),
    ("exact_gf_corrected",  method_exact_gf_corrected),
    ("cap_wavepacket",      method_cap_wavepacket),
]


# ---------------------------------------------------------------------------
# Independent spinless cross-check: network cascade using the *measured*
# junction S-matrix (see module docstring, point 3).
# ---------------------------------------------------------------------------

def measure_junction_scattering(p: t.PhysicsParams, E: float,
                                 corrected: bool) -> np.ndarray:
    """Return the 3x3 unitary S-matrix of one isolated Y-junction of this graph.

    Ports ordered [lead, upper-arm, lower-arm]. Built from a single vertex
    site with analytic self-energies for the three attached semi-infinite
    chains (hopping t_lead for the lead port, t_ring for the two arm ports) --
    the same self-energy formula used by run_exact_transmission, applied to
    a 1-site "device" so the result is the junction's own scattering matrix,
    uncontaminated by the rest of the ring.
    """
    t_lead, t_ring = p.t_lead, p.t_ring
    onsite = t_lead + 2.0 * t_ring
    if corrected:
        onsite += p.t_junction_correction
    hops = (t_lead, t_ring, t_ring)
    sigmas = np.array([lead_surface_self_energy(h, E) for h in hops])
    gammas = -2.0 * sigmas.imag
    G = 1.0 / (E - onsite - sigmas.sum())
    S = np.eye(3, dtype=complex) - 1j * np.sqrt(np.outer(gammas, gammas)) * G
    return S


def analytical_G_kirchhoff_ring(Phi: float, p: t.PhysicsParams | None = None,
                                 E_F: float | None = None,
                                 corrected: bool = False) -> float:
    """Spinless (alpha=0) two-terminal G/G0 for a ring built from two measured
    Y-junctions, connected by uniform phase-only arms, via direct network
    cascade (10x10 linear solve; see derivation notes in the module docstring
    of ab_ac_reconciliation.py history / commit messages).

    This is NOT the legacy thesis formula -- it is derived from scratch here
    to test the "idealised vs. Kirchhoff junction" hypothesis. It matches
    the exact solver to within a few percent for most Phi (see validation
    results saved under ab_ac_reconciliation/), which is already far closer
    than the legacy formula's 40-100%+ deviation, but does not reach 1%
    everywhere -- flagged as unresolved future work, not claimed as final.
    """
    if p is None:
        p = legacy.transparent_ring_params(Phi=Phi, alpha=0.0)
    if E_F is None:
        E_F = legacy.E_F_MEV

    x_ring = 1.0 - E_F / (2.0 * p.t_ring)
    k_ring = np.arccos(np.clip(x_ring, -1.0, 1.0)) / p.delta_s
    theta_arm = k_ring * p.L_ring

    S = measure_junction_scattering(p, E_F, corrected=corrected)
    gauge_U = np.pi * Phi
    gauge_D = -gauge_U

    idx = {"a2": 0, "a3": 1, "b1": 2, "b2": 3, "b3": 4,
           "d2": 5, "d3": 6, "e1": 7, "e2": 8, "e3": 9}
    n = 10
    A = np.zeros((n, n), dtype=complex)
    rhs = np.zeros(n, dtype=complex)
    row = 0
    for k, out in enumerate(["b1", "b2", "b3"]):
        A[row, idx[out]] = 1.0
        A[row, idx["a2"]] -= S[k, 1]
        A[row, idx["a3"]] -= S[k, 2]
        rhs[row] = S[k, 0]
        row += 1
    for k, out in enumerate(["e1", "e2", "e3"]):
        A[row, idx[out]] = 1.0
        A[row, idx["d2"]] -= S[k, 1]
        A[row, idx["d3"]] -= S[k, 2]
        row += 1
    A[row, idx["d2"]] = 1.0; A[row, idx["b2"]] = -np.exp(1j*(theta_arm+gauge_U)); row += 1
    A[row, idx["a2"]] = 1.0; A[row, idx["e2"]] = -np.exp(1j*(theta_arm-gauge_U)); row += 1
    A[row, idx["d3"]] = 1.0; A[row, idx["b3"]] = -np.exp(1j*(theta_arm+gauge_D)); row += 1
    A[row, idx["a3"]] = 1.0; A[row, idx["e3"]] = -np.exp(1j*(theta_arm-gauge_D)); row += 1

    sol = np.linalg.solve(A, rhs)
    T_single = abs(sol[idx["e1"]]) ** 2
    return 2.0 * T_single   # both spin channels, identical at alpha=0


# ---------------------------------------------------------------------------
# Save / compare / retry harness
# ---------------------------------------------------------------------------

@dataclass
class Attempt:
    method: str
    G_numeric: float
    G_analytical_legacy: float
    rel_error_pct: float
    passed: bool
    wall_seconds: float


@dataclass
class PointResult:
    Phi: float
    alpha: float
    attempts: list[Attempt] = field(default_factory=list)
    final_passed: bool = False
    G_kirchhoff_ring: float | None = None   # spinless cross-check, alpha=0 only

    def to_dict(self) -> dict[str, Any]:
        return {
            "Phi": self.Phi,
            "alpha": self.alpha,
            "final_passed": self.final_passed,
            "G_kirchhoff_ring": self.G_kirchhoff_ring,
            "attempts": [vars(a) for a in self.attempts],
        }


def reconcile_point(Phi: float, alpha: float, E_F: float = 4.19,
                     tolerance_pct: float = TOLERANCE_PCT,
                     verbose: bool = True) -> PointResult:
    """Try each method in METHODS order until one matches the legacy analytic
    formula to within tolerance_pct, saving every attempt regardless."""
    G_ana = legacy.analytical_G(Phi, alpha)
    result = PointResult(Phi=Phi, alpha=alpha)

    for name, fn in METHODS:
        t0 = time.perf_counter()
        G_num = fn(Phi, alpha, E_F)
        dt_wall = time.perf_counter() - t0
        rel_err = float(abs(G_num - G_ana) / max(abs(G_ana), 1e-9) * 100.0)
        passed = bool(rel_err <= tolerance_pct)
        result.attempts.append(Attempt(
            method=name, G_numeric=float(G_num), G_analytical_legacy=float(G_ana),
            rel_error_pct=rel_err, passed=passed, wall_seconds=float(dt_wall),
        ))
        if verbose:
            print(f"    [{name:22s}] G_num={G_num:9.5f}  G_ana={G_ana:9.5f}  "
                  f"err={rel_err:9.3f}%  {'PASS' if passed else 'retry'}"
                  f"  ({dt_wall:.3f}s)")
        if passed:
            result.final_passed = True
            break

    if abs(alpha) < 1e-12:
        result.G_kirchhoff_ring = float(analytical_G_kirchhoff_ring(Phi, E_F=E_F))

    return result


def run_reconciliation_sweep(Phi_values: np.ndarray, alpha_values: np.ndarray,
                              E_F: float = 4.19,
                              tolerance_pct: float = TOLERANCE_PCT) -> list[PointResult]:
    points: list[PointResult] = []
    total = len(Phi_values) * len(alpha_values)
    count = 0
    for alpha in alpha_values:
        for Phi in Phi_values:
            count += 1
            print(f"[{count}/{total}] Phi={Phi:.3f}  alpha={alpha:.2f} meV*nm")
            points.append(reconcile_point(Phi, alpha, E_F, tolerance_pct))
    return points


def save_results(points: list[PointResult], tag: str) -> Path:
    payload = {
        "tolerance_pct": TOLERANCE_PCT,
        "fermi_energy_mev": legacy.E_F_MEV,
        "n_points": len(points),
        "n_passed": sum(p.final_passed for p in points),
        "points": [p.to_dict() for p in points],
    }
    path = OUTPUT_DIR / f"reconciliation_{tag}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved full audit trail: {path}")
    return path


def print_summary(points: list[PointResult]) -> None:
    n = len(points)
    n_pass = sum(p.final_passed for p in points)
    print("\n" + "=" * 70)
    print(f"  Reconciliation summary: {n_pass}/{n} points within {TOLERANCE_PCT}% "
          f"of the legacy analytic formula")
    print("=" * 70)
    by_method_pass = {name: 0 for name, _ in METHODS}
    for p in points:
        if p.final_passed:
            by_method_pass[p.attempts[-1].method] += 1
    for name, cnt in by_method_pass.items():
        print(f"    passed via {name:22s}: {cnt}")
    failed = [p for p in points if not p.final_passed]
    if failed:
        closest = min(failed, key=lambda p: min(a.rel_error_pct for a in p.attempts))
        closest_attempt = min(closest.attempts, key=lambda a: a.rel_error_pct)
        print(f"\n  {len(failed)} point(s) never reached {TOLERANCE_PCT}%.")
        print(f"  Closest miss: Phi={closest.Phi:.3f} alpha={closest.alpha:.2f} "
              f"-> {closest_attempt.method} at {closest_attempt.rel_error_pct:.2f}% error.")
        print("  Root cause (see module docstring): the legacy analytic formula "
              "assumes an idealised critically-coupled junction; this graph's Y-"
              "junction is the Kirchhoff/democratic vertex (r~=-1/3, t~=2/3). "
              "These are different physical junctions and will not converge by "
              "further numerical refinement alone -- the analytic reference "
              "needs re-derivation at the graph's measured epsilon.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true",
                         help="Small grid (5 Phi x 2 alpha) for a fast smoke test.")
    parser.add_argument("--n-phi", type=int, default=11)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 20.0])
    args = parser.parse_args()

    n_phi = 5 if args.quick else args.n_phi
    alphas = [0.0, 20.0] if args.quick else args.alphas

    Phi_values = np.linspace(-1.0, 1.0, n_phi)
    alpha_values = np.array(alphas)

    wall0 = time.perf_counter()
    points = run_reconciliation_sweep(Phi_values, alpha_values)
    print_summary(points)
    tag = time.strftime("%Y%m%d_%H%M%S")
    save_results(points, tag)
    print(f"\nTotal wall time: {time.perf_counter()-wall0:.1f} s")
