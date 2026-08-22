"""Reproduces the DTBC paper's OWN numerical demonstration (Akramov et al.,
arXiv:2608.05338, Section III, Fig. 2 and Fig. 3) as an independent
corroboration of the boundary-condition formulas re-derived in dtbc.py,
BEFORE trusting them on the JCE26 ring (see dtbc.py's module docstring for
why Eq. 15 was used as the anchor instead of Eq. 28/32).

This script does NOT import tools.py or use the JCE26 spinor/ring
machinery at all -- it is a standalone, bare 1D scalar lattice in the
paper's own hbar=m=1 units, matching their exact parameters:
    J = 400 (so the domain has 401 grid points, x in [0, 10]),
    h = 0.025, dt = 6.25e-6, sigma = 1, k0 = 5, packet centered at x=L/2.
Left boundary: homogeneous Dirichlet (Psi_0 = 0), exactly as in the paper.
Right boundary: the exact DTBC, Eq. (*) of dtbc.py's docstring (equivalent
to the paper's Eq. 15/32, re-derived independently here).

Success criterion (matching the paper's own Fig. 3): the discrete norm
M(t) = h * sum |Psi_j|^2 should decay smoothly and MONOTONICALLY from 1 to
~0 as the packet exits through the right boundary, with no oscillation or
late-time increase (which would signal spurious reflection).

Run: python dtbc_selftest.py
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import j1 as _j1
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def bessel_kernel(t_array: np.ndarray, h2: float) -> np.ndarray:
    """K(t) = exp(-i t/h^2) J1(t/h^2) / t,  K(0) = 1/(2 h^2).  (paper's units)"""
    t_array = np.asarray(t_array, dtype=float)
    out = np.empty(t_array.shape, dtype=complex)
    zero_mask = t_array <= 0.0
    nz = ~zero_mask
    x = t_array[nz] / h2
    out[nz] = np.exp(-1j * t_array[nz] / h2) * _j1(x) / t_array[nz]
    out[zero_mask] = 1.0 / (2.0 * h2)
    return out


def main() -> None:
    # ---- paper's exact parameters (Section III) -----------------------------
    J = 400
    h = 0.025
    L = J * h                       # = 10.0
    dt = 6.25e-6
    sigma = 1.0
    k0 = 5.0
    x0 = L / 2.0
    total_time = 2.0                # matches Fig. 2/3's t range

    n_sites = J + 1                 # j = 0 .. J
    x = np.arange(n_sites) * h
    time_steps = int(round(total_time / dt))

    print(f"J={J}  h={h}  dt={dt:.3e}  time_steps={time_steps}  "
          f"(this reproduces Akramov et al. 2026, Fig. 2/3)")

    # ---- Hamiltonian: bulk on-site 1/h^2 (paper's hbar=m=1 convention), -----
    # hopping -1/(2h^2); matches i dPsi_j/dt = (1/(2h^2))(2Psi_j-Psi_{j-1}-Psi_{j+1})
    #
    # IMPORTANT: on-site energy must be BUILT FROM ACTUAL BONDS (t_hop per
    # bond), not set to the uniform bulk value 2*t_hop everywhere -- a site
    # with only one physical neighbour (both edges of this finite chain,
    # before any boundary condition is added) only picks up t_hop on-site,
    # not 2*t_hop. Using sp.diags with a flat "2*t_hop" diagonal at every
    # row (as an earlier version of this script did) silently gives the
    # truncated edges the WRONG (bulk) on-site energy and over-corrects
    # once the +t_hop DTBC term is added on top -- this was caught by an
    # independent residual check against a long, unbounded reference chain
    # (dtbc.py's own boundary-matrix construction, via tools.py's bond-by-
    # bond Hamiltonian assembly, does not have this bug).
    t_hop = 1.0 / (2.0 * h * h)
    off = np.full(n_sites - 1, -t_hop, dtype=complex)
    H = sp.diags([off, off], offsets=[-1, 1], format="lil")
    for j in range(n_sites):
        n_bonds = (1 if j > 0 else 0) + (1 if j < n_sites - 1 else 0)
        H[j, j] = n_bonds * t_hop

    # Left boundary: homogeneous Dirichlet -> simplest realization is to just
    # leave site 0's bulk row as built (only one physical bond, to site 1);
    # that IS already a hard wall / Dirichlet-like truncation matching the
    # paper's "Psi_0^n = 0 for all n" (we additionally pin it explicitly below
    # for exact fidelity to their setup).

    # Right boundary: add the DTBC correction (+t_hop on-site, bringing the
    # truncated edge's on-site energy up to the bulk value 2*t_hop, matching
    # dtbc.py's derivation) at site J (index n_sites-1).
    right_site = n_sites - 1
    H[right_site, right_site] += t_hop
    H = H.tocsr()

    h2 = h * h   # the paper's own "h^2" appears directly, no unit translation needed here
    K0 = 1.0 / (2.0 * h2)
    diag_correction = (dt ** 2) * (t_hop ** 2) / 2.0   # hbar=1 here, matches dtbc.py's formula with hbar->1

    identity  = sp.identity(n_sites, format="csr", dtype=complex)
    prefactor = 1j * dt / 2.0   # hbar = 1
    A = (identity + prefactor * H).tolil()
    A[right_site, right_site] += diag_correction
    A = A.tocsr()
    B = (identity - prefactor * H).tocsr()

    solver = spla.factorized(A.tocsc())

    # ---- initial Gaussian packet, moving right (k0>0) ------------------------
    psi = np.exp(-0.5 * ((x - x0) / sigma) ** 2) * np.exp(1j * k0 * x)
    psi[0] = 0.0   # enforce Dirichlet at the left, matching the paper exactly
    norm0 = h * np.sum(np.abs(psi) ** 2)
    psi = psi / np.sqrt(norm0)   # normalize M(0)=1, matching their Fig. 3

    t_axis = np.arange(time_steps + 1, dtype=float) * dt
    K_vals = bessel_kernel(t_axis, h2)

    boundary_hist = np.empty(time_steps + 1, dtype=complex)
    boundary_hist[0] = psi[right_site]

    M_history = np.empty(time_steps + 1, dtype=float)
    M_history[0] = h * np.sum(np.abs(psi) ** 2)

    snapshot_times = [0.0, 0.67, 1.33, 2.00]
    snapshots = {}

    def maybe_snapshot(n_step, t_now):
        for st in snapshot_times:
            if st not in snapshots and t_now >= st - dt / 2:
                snapshots[st] = np.abs(psi) ** 2

    maybe_snapshot(0, 0.0)

    wall0 = time.perf_counter()
    for n in range(1, time_steps + 1):
        t_n = t_axis[n]
        psi0_b = boundary_hist[0]
        if n >= 2:
            lags = K_vals[n - 1:0:-1]
            past = boundary_hist[1:n]
            hist_sum = dt * np.dot(lags, past)
        else:
            hist_sum = 0.0 + 0.0j
        Src_n = t_hop * ((dt / 2.0) * K_vals[n] * psi0_b + hist_sum)
        source = np.zeros(n_sites, dtype=complex)
        source[right_site] = -dt * Src_n   # hbar=1; real coefficient, see dtbc.py docstring

        psi = solver(B @ psi + source)
        psi[0] = 0.0   # re-enforce Dirichlet each step

        boundary_hist[n] = psi[right_site]
        M_history[n] = h * np.sum(np.abs(psi) ** 2)
        maybe_snapshot(n, t_n)

        if n % max(1, time_steps // 20) == 0:
            print(f"  t={t_n:.3f}  M(t)={M_history[n]:.6f}  "
                  f"[{time.perf_counter()-wall0:.1f}s elapsed]")

    print(f"\nFinal M(t={total_time}) = {M_history[-1]:.6f}  "
          f"(paper's Fig. 3 shows this decaying smoothly to ~0)")
    print(f"Total wall time: {time.perf_counter()-wall0:.1f}s")

    # ---- monotonicity check (the actual pass/fail criterion) ----------------
    dM = np.diff(M_history)
    n_increases = int(np.sum(dM > 1e-10))
    max_increase = float(np.max(dM)) if n_increases else 0.0
    print(f"\nMonotonicity check: {n_increases}/{len(dM)} steps show M(t) INCREASING "
          f"(should be 0, or only tiny numerical noise).")
    if n_increases:
        print(f"  Largest single-step increase: {max_increase:.3e}")

    # ---- plots (reproduce Fig. 2 and Fig. 3 style) ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for st in snapshot_times:
        if st in snapshots:
            ax.plot(x, snapshots[st], label=f"t={st:.2f}")
    ax.set_xlabel("x"); ax.set_ylabel(r"$|\Psi_j|^2$")
    ax.set_title("Reproduccion de la Fig. 2 del articulo")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_axis, M_history, color="black", linewidth=1.5)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("t"); ax.set_ylabel("M(t)")
    ax.set_title("Reproduccion de la Fig. 3 (norma discreta) del articulo")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Corroboracion: DTBC de Akramov et al. 2026, reproducido con dtbc.py", fontsize=12)
    fig.tight_layout()
    fig.savefig("dtbc_selftest_paper_reproduction.png", dpi=160)
    print("\nSaved: dtbc_selftest_paper_reproduction.png")

    np.savez_compressed("dtbc_selftest_results.npz",
                        t_axis=t_axis, M_history=M_history, x=x,
                        n_increases=n_increases, max_increase=max_increase)


if __name__ == "__main__":
    main()
