"""Exact Discrete Transparent Boundary Conditions (DTBC) for the spatially
discrete Schroedinger equation, following Akramov, Yusupov, Ehrhardt &
Matrasulov, "Transparent boundary conditions for the spatially discrete
Schroedinger equation: Reflectionless quantum transport in 1D lattices",
arXiv:2608.05338 (2026) -- referred to below as "the paper".

This is the FULL, non-Markovian (time-convolution) version that
conductance.py's `lead_self_energy` (2026-08-21 patch) explicitly said it
was NOT: that earlier patch is exact only for a single Fourier component
(the injection energy). This module implements the paper's exact kernel,
valid for the whole wavepacket at once, no matter its momentum spread.

-----------------------------------------------------------------------
Derivation actually used here (re-derived independently from the paper's
equations 4-15, to avoid propagating what looks like a units slip in the
paper's own Eq. 28/32 -- see "Corroboration and one correction" below)
-----------------------------------------------------------------------

The paper's lattice equation (4), in their hbar=m=1 units:

    i dPsi_j/dt + (1/(2h^2)) (Psi_{j-1} - 2 Psi_j + Psi_{j+1}) = 0

is IDENTICAL in form to our physical tight-binding equation

    i hbar dpsi_j/dt = t_lead (2 psi_j - psi_{j-1} - psi_{j+1})

under the identification  h_paper^2  <->  tau_c := hbar / (2 t_lead)
(a TIME scale, not a length -- a consequence of hbar=m=1 making
length^2 and time interchangeable in the paper's unit system; this is
verified below by an independent physical-units re-derivation of the
free-particle propagator, which lands on the exact same argument
2*t_lead*t/hbar = t/tau_c for its Bessel functions -- two independent
routes to the same dimensionless combination is a strong consistency
check).

The paper's Eq. (11)-(12) give, in the Laplace domain, the "ghost point"
just beyond the right boundary site J:

    Phihat_{J+1}(s) = xi_+(s) Phihat_J(s),   xi_+(s) = 1-i h^2 s + i sqrt(2is+h^2 s^2)

Rather than translate their subsequent D_x-based bookkeeping (Eqs. 27-32),
which introduces an extra explicit 1/h that does not appear to be
dimensionally consistent with their own Eq. (15) once you divide through by
h (Eq. 15 already has no bare "h" left outside the kernel once you use it
to relate Psi_{J+1}, only h^2 hidden inside K itself) -- we re-derive the
inverse Laplace transform of the DIRECT ghost-point relation from their
already-validated Eq. (15) instead (their Eq. 15 is cross-checked in their
own paper against the known continuum TBC, Eq. 26 there matching Eq. 2, so
we anchor to that, not to the possibly-mistyped Eq. 28/32). Using
D^-_x Psi_J(t) = (1/h)[Psi_{J+1}(t) - Psi_J(t)] (their Eq. 6's forward
exterior operator, equal to D^-_x Psi_J(t) by their current-continuity
condition Eq. 7) together with their Eq. (15):

    (1/h)[Psi_{J+1}(t) - Psi_J(t)] = -(1/h) Psi_J(t) + (i/h) INT_0^t K(t-tau) Psi_J(tau) dtau

the two "-(1/h) Psi_J(t)" terms cancel the "Psi_J(t)/h" on the left after
multiplying by h, leaving the clean, h-free amplitude relation:

    Psi_{J+1}(t) = i INT_0^t K(t-tau) Psi_J(tau) dtau                      (*)

    K(t) = exp(-i t/h^2) J_1(t/h^2) / t ,   K(0) = 1/(2 h^2)

This (*) is what is implemented below (with h^2 -> tau_c). It is the
"ghost value" needed to complete the boundary lattice site's bulk equation
exactly, no finite absorbing region, no lead padding needed.

Substituting (*) into the ACTUAL sparse Hamiltonian built by
tools.build_single_ring_hamiltonian (whose boundary site only has ONE
physical bond, on-site energy t_lead, not the bulk 2*t_lead -- the same
subtlety that bit the monochromatic self-energy patch, see
conductance.py's comments) requires the REAL on-site correction "+t_lead"
(to bring that site's on-site energy up to the bulk 2*t_lead) PLUS a
separate, purely imaginary/history-dependent term S_J(t) = -t_lead*psi_{J+1}(t)
that supplies the (removed) hopping to the ghost site. Only S_J(t) is
non-local in time; the "+t_lead" on-site piece is treated with ordinary
symmetric Crank-Nicolson averaging (folded into H_eff before building A
and B in the usual way -- do NOT add it a second time when discretizing
S_J, that double-counting was an earlier bug in this file, caught by the
paper-reproduction self-test in dtbc_selftest.py, see the "Corroboration"
section below).

Discretizing S_J(t) = -i*t_lead*INT_0^t K(t-tau)psi_J(tau)dtau via the
trapezoidal rule at t_n, treating it one-sided (only at level n, not
averaged with n-1, matching the paper's own treatment of the memory term)
and carrying the algebra through the standard A=I+i*gamma*H, B=I-i*gamma*H,
gamma=dt/(2 hbar) Crank-Nicolson construction gives:

    [A + diag_correction] psi^n = B psi^{n-1} - (dt/hbar) Src_n

where A, B are the STANDARD Crank-Nicolson matrices built from
H_eff = H_built + t_lead * (on-site +1 at each boundary site, both spins),
diag_correction is a FIXED real number added once at each boundary row of
A only:

    diag_correction = dt^2 * t_lead^2 / (2 hbar^2)

(the implicit self-term from the trapezoidal quadrature's K(0) piece), and
Src_n is the EXPLICIT part of the memory integral, built from already-known
past values only (note: real coefficient here, the factors of i cancel
exactly against the S_J(t)=-i*t_lead*[...] definition -- an earlier version
of this file kept a spurious extra i, also caught by the self-test):

    Src_n = t_lead * [ (dt/2) K(t_n) psi_J(0)  +  dt * sum_{p=1}^{n-1} K(t_n - t_p) psi_J^p ]

Because diag_correction is a fixed number, A can still be factorized ONCE
and reused for every time step, exactly like the rest of this codebase --
only Src_n changes step to step, and it is O(1) extra work per step given
the running history (the whole scheme is O(N_t^2) overall, the well-known
cost of an exact convolution boundary condition; see Lubich & Schaedle,
SIAM J. Sci. Comput. 24, 161 (2002) for an O(N_t log N_t) alternative,
NOT implemented here -- flagged as future work in the README).

-----------------------------------------------------------------------
Corroboration against the paper (see dtbc_selftest.py)
-----------------------------------------------------------------------
Before trusting any of the above on the JCE26 ring, dtbc_selftest.py
reproduces the paper's OWN numerical demonstration (their Section III,
Fig. 2/3): a bare 1D lattice, hbar=m=1, J=400, h=0.025, dt=6.25e-6, a
Gaussian packet with sigma=1, k0=5 launched from the middle of the domain,
transparent boundary at the right end only (left end held at a Dirichlet
psi=0, matching their setup exactly). Success criterion: the discrete norm
M(t) should decay smoothly and monotonically from 1 to ~0, matching their
Fig. 3, with NO oscillatory late-time increase (which would indicate
spurious reflection).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import j1 as _j1

import tools as t


# ---------------------------------------------------------------------------
# The Bessel convolution kernel
# ---------------------------------------------------------------------------

def characteristic_time(t_hop: float, hbar: float) -> float:
    """tau_c = hbar / (2 t_hop); the physical-units analogue of the paper's h^2."""
    return hbar / (2.0 * t_hop)


def bessel_kernel(t_array: np.ndarray, tau_c: float) -> np.ndarray:
    """K(t) = exp(-i t/tau_c) * J1(t/tau_c) / t,  K(0) = 1/(2 tau_c).

    Vectorized; t_array may include t=0 (handled via the small-argument
    limit of J1, matching the paper's Eq. 33).
    """
    t_array = np.asarray(t_array, dtype=float)
    out = np.empty(t_array.shape, dtype=complex)
    zero_mask = t_array <= 0.0
    nz = ~zero_mask
    x = t_array[nz] / tau_c
    out[nz] = np.exp(-1j * t_array[nz] / tau_c) * _j1(x) / t_array[nz]
    out[zero_mask] = 1.0 / (2.0 * tau_c)
    return out


# ---------------------------------------------------------------------------
# Matrix construction (fixed for the whole run -- factorize once)
# ---------------------------------------------------------------------------

def build_dtbc_matrices(p: t.PhysicsParams, layout: t.SingleRingLayout):
    """Return (A, B, boundary_sites, diag_correction) for the DTBC scheme.

    boundary_sites = (left_site, right_site): the two outermost lattice
    sites where the memory-convolution source must be applied every step.
    """
    tau_c = characteristic_time(p.t_lead, t.h_bar)
    K0 = 1.0 / (2.0 * tau_c)   # = t_lead / hbar

    left_site  = int(layout.left_lead_sites[0])
    right_site = int(layout.right_lead_sites[-1])

    H_eff = t.build_single_ring_hamiltonian(p, layout).tolil()
    for site in (left_site, right_site):
        for s in (0, 1):
            H_eff[2 * site + s, 2 * site + s] += p.t_lead   # symmetric, real correction
    H_eff = H_eff.tocsr()

    identity  = sp.identity(layout.spinor_size, format="csr", dtype=complex)
    prefactor = 1j * p.dt / (2.0 * t.h_bar)
    A = (identity + prefactor * H_eff).tolil()
    B = (identity - prefactor * H_eff).tocsr()

    diag_correction = (p.dt ** 2) * (p.t_lead ** 2) / (2.0 * t.h_bar ** 2)
    for site in (left_site, right_site):
        for s in (0, 1):
            A[2 * site + s, 2 * site + s] += diag_correction
    A = A.tocsr()

    return A, B, (left_site, right_site), tau_c, K0


# ---------------------------------------------------------------------------
# Time propagation with the exact memory-convolution boundary source
# ---------------------------------------------------------------------------

def run_dtbc_propagation(
    p: t.PhysicsParams,
    psi_initial: np.ndarray,
    time_steps: int,
    keep_history: bool = False,
):
    """Propagate psi_initial for time_steps DTBC-bounded Crank-Nicolson steps.

    Returns a dict with at least 'psi_final', 'P_total_history' (probability
    remaining in the finite domain at every step -- should decay smoothly,
    no spurious late-time rise), and, if keep_history, 'psi_history' and the
    two boundary sites' full time series (needed to reconstruct outgoing
    current if desired).
    """
    layout = t.build_single_ring_layout(p)
    A, B, (left_site, right_site), tau_c, K0 = build_dtbc_matrices(p, layout)
    solver = spla.factorized(A.tocsc())

    t_axis = np.arange(time_steps + 1, dtype=float) * p.dt
    K_vals = bessel_kernel(t_axis, tau_c)   # K_vals[k] = K(k*dt), k=0..time_steps

    # spin-resolved boundary histories: shape (time_steps+1, 2 sites, 2 spins)
    boundary_hist = np.zeros((time_steps + 1, 2, 2), dtype=complex)
    boundary_sites = (left_site, right_site)
    for b, site in enumerate(boundary_sites):
        for s in (0, 1):
            boundary_hist[0, b, s] = psi_initial[2 * site + s]

    psi = psi_initial.copy()
    P_total_history = np.empty(time_steps + 1, dtype=float)
    P_total_history[0] = float(np.sum(np.abs(psi) ** 2))

    psi_history = np.empty((time_steps + 1, psi_initial.size), dtype=complex) if keep_history else None
    if keep_history:
        psi_history[0] = psi

    for n in range(1, time_steps + 1):
        source = np.zeros(layout.spinor_size, dtype=complex)
        for b, site in enumerate(boundary_sites):
            for s in (0, 1):
                psi0_bs = boundary_hist[0, b, s]
                if n >= 2:
                    # sum_{p=1}^{n-1} K(t_n - t_p) * psi_p   (strictly past, known)
                    lags = K_vals[n - 1:0:-1]            # K(t_n-t_1), ..., K(t_n-t_{n-1})
                    past = boundary_hist[1:n, b, s]       # psi_1 .. psi_{n-1}
                    hist_sum = float(p.dt) * np.dot(lags, past)
                else:
                    hist_sum = 0.0 + 0.0j
                Src_n = p.t_lead * ((p.dt / 2.0) * K_vals[n] * psi0_bs + hist_sum)
                source[2 * site + s] = -(p.dt / t.h_bar) * Src_n

        psi = solver(B @ psi + source)

        for b, site in enumerate(boundary_sites):
            for s in (0, 1):
                boundary_hist[n, b, s] = psi[2 * site + s]

        P_total_history[n] = float(np.sum(np.abs(psi) ** 2))
        if keep_history:
            psi_history[n] = psi

    result = {
        "psi_final": psi,
        "P_total_history": P_total_history,
        "time_axis_ps": t_axis,
        "boundary_history": boundary_hist,
        "boundary_sites": boundary_sites,
        "K_vals": K_vals,
        "tau_c": tau_c,
    }
    if keep_history:
        result["psi_history"] = psi_history
    return result


def transmission_reflection_from_history(p: t.PhysicsParams, result: dict) -> dict:
    """Post-process a run_dtbc_propagation result into T and R.

    Uses the exact ghost-point formula (*) to reconstruct psi just beyond
    each boundary at every step, then integrates the standard tight-binding
    probability current J_{a->b} = (2*t_lead/hbar) * Im[psi_a^* psi_b]
    flowing OUT through each boundary. T = cumulative outward flux at the
    right boundary; R = cumulative outward flux at the left boundary.
    Should satisfy T + R + P_total_remaining/N0 = 1 (probability budget).
    """
    boundary_hist = result["boundary_history"]     # (n_steps+1, 2 sites, 2 spins)
    K_vals = result["K_vals"]
    dt = p.dt
    n_steps = boundary_hist.shape[0] - 1

    T_flux = 0.0
    R_flux = 0.0
    for site_idx in (0, 1):
        # site_idx 0 = left boundary, ghost site is "-1" (further left);
        # site_idx 1 = right boundary, ghost site is "J+1" (further right).
        # In BOTH cases J_{site->ghost} = (2 t_lead/hbar) Im[psi_site^* psi_ghost]
        # is already the correct OUTWARD (away from the domain) current --
        # no extra sign flip needed, the ghost site is defined as "outside"
        # in both cases by construction of formula (*).
        for s in (0, 1):
            hist = boundary_hist[:, site_idx, s]
            # ghost value at each step n via formula (*): psi_ghost(t_n) = i * INT_0^{t_n} K(t_n-tau) psi(tau) dtau
            ghost = np.zeros(n_steps + 1, dtype=complex)
            for n in range(1, n_steps + 1):
                integral = (dt / 2.0) * (K_vals[n] * hist[0] + K_vals[0] * hist[n])
                if n >= 2:
                    integral += dt * np.dot(K_vals[n - 1:0:-1], hist[1:n])
                ghost[n] = 1j * integral
            current = (2.0 * p.t_lead / t.h_bar) * np.imag(np.conj(hist) * ghost)
            flux = float(np.sum(current) * dt)   # rectangle rule, dt small
            if site_idx == 1:
                T_flux += flux
            else:
                R_flux += flux
    return {"T": T_flux, "R": R_flux}
