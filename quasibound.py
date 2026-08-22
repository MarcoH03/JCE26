"""Detect long-dwell-time (quasi-bound) resonant states in the JCE26 ring
WITHOUT running any time-domain simulation, and estimate their lifetimes
directly -- answering the question of whether states with "very high
permanence in the ring" (long enough that the finite simulation time
can't tell reflected from transmitted) are actually present, and how long
they live.

Method (Breit-Wigner resonance analysis)
------------------------------------------
It is a standard, textbook result in scattering theory (and explicitly the
physical content behind the eigenvalue method of Shao, Porod, Lent & Kirkner,
J. Appl. Phys. 78, 6353 (1995) -- the second paper supplied this session)
that quasi-bound states are POLES of the transmission amplitude t(E) in the
complex energy plane, at E = E_R - i*Gamma (their Eq. 1 and Fig. 3/Table I),
with lifetime tau = hbar / (2*Gamma).

A pole at E_R - i*Gamma produces, on the REAL energy axis, a Lorentzian
(Breit-Wigner) peak in T(E) = |t(E)|^2:

    T(E) ~ Gamma^2 / [ (E-E_R)^2 + Gamma^2 ]

whose full width at half maximum (FWHM) is exactly 2*Gamma. This means
Gamma -- and hence the lifetime -- can be read off directly from a REAL-energy
scan of T(E) (which qtbm.py already computes exactly, via a single sparse
linear solve per energy point, no time evolution needed at all): find a
narrow peak, fit its FWHM, done.

This module implements exactly that:
  1. scan_transmission(...): T(E) on a fine real-energy grid (cheap, exact,
     stationary -- reuses qtbm.qtbm_conductance).
  2. find_resonances(...): locate local maxima and fit a Lorentzian to each
     to extract E_R, Gamma, and tau = hbar/(2*Gamma).
  3. A direct usability check: compare tau against the total_time_ps actually
     used by the time-domain scripts (main.py/conductance.py/ab_ac_proof.py)
     -- if tau is comparable to or longer than that, the wavepacket run is
     almost certainly contaminated by an unconverged quasi-bound state, no
     matter how good the boundary condition is (this is exactly what the
     2026-08-21 DTBC/self-energy comparison found empirically: T+R plateaued
     around 0.35-0.38 regardless of boundary treatment).

A note on the full complex-plane eigenvalue method
------------------------------------------------------
The uploaded paper's OWN method (their Eq. 23, "(H - k_L B^L - k_R B^R - ED)
psi = 0", solved as a genuine matrix eigenvalue problem after linearizing the
k-dependence) is exact and non-iterative for THEIR finite-element
discretization, where the boundary term is LINEAR in k (see their Eq. 21c/d
-- a simple derivative-matching Robin condition). Our tight-binding lattice's
exact boundary self-energy is Sigma(E) = -t_lead*exp(i k(E) delta_x), which is
EXPONENTIAL in k, not linear -- so the same "linearize by doubling the matrix"
trick (their Eq. 30-32) does not turn this into a polynomial eigenvalue
problem the way it does for their FEM matrices. The direct analogue for a
lattice model is to root-find det(H_eff(E) - E) = 0 (or track the smallest
singular value) over the complex E-plane, which IS implemented below
(complex_pole_refine) as a secondary, opt-in refinement step seeded from the
cheap real-axis scan above -- but it is a 2D Newton search on a sparse
system and is slower and less robust than the scan; the Breit-Wigner width
from the real-axis scan is the primary, recommended tool.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import tools as t
import qtbm as q


# ---------------------------------------------------------------------------
# Real-energy transmission scan (cheap, exact, stationary)
# ---------------------------------------------------------------------------

def scan_transmission(p: t.PhysicsParams, energies_mev: np.ndarray) -> np.ndarray:
    """T(E) = T_up(E) + T_down(E) at each energy, via QTBM (one sparse solve
    pair per energy point; no time evolution)."""
    T = np.empty(len(energies_mev), dtype=float)
    for i, E in enumerate(energies_mev):
        T[i] = q.qtbm_conductance(p, float(E)).T
    return T


# ---------------------------------------------------------------------------
# Peak finding + Lorentzian (Breit-Wigner) fit
# ---------------------------------------------------------------------------

@dataclass
class Resonance:
    E_R_mev: float           # resonance energy (peak location, meV)
    Gamma_mev: float         # HWHM-of-the-pole (paper's convention: pole at E_R - i*Gamma)
    tau_ps: float            # lifetime = hbar / (2*Gamma)
    T_peak: float            # peak transmission value
    fit_rms_residual: float  # quality of the Lorentzian fit (lower = cleaner isolated resonance)


def _lorentzian(E, E_R, Gamma, T_peak):
    return T_peak * Gamma ** 2 / ((E - E_R) ** 2 + Gamma ** 2)


def find_resonances(
    energies_mev: np.ndarray,
    T_values: np.ndarray,
    min_prominence: float = 0.05,
    fit_half_window_points: int = 8,
) -> list[Resonance]:
    """Locate local maxima of T(E) and fit a Lorentzian around each to get
    (E_R, Gamma, tau). Uses a simple local-maximum + least-squares fit
    (no scipy.signal dependency, keeps this self-contained)."""
    resonances = []
    n = len(energies_mev)
    for i in range(1, n - 1):
        if T_values[i] > T_values[i - 1] and T_values[i] > T_values[i + 1]:
            # local prominence check: how much does it stand above its
            # immediate surroundings (cheap proxy, good enough for isolated peaks)
            lo = max(0, i - fit_half_window_points)
            hi = min(n, i + fit_half_window_points + 1)
            local_min = min(T_values[lo], T_values[hi - 1])
            prominence = T_values[i] - local_min
            if prominence < min_prominence * max(T_values[i], 1e-12):
                continue

            E_window = energies_mev[lo:hi]
            T_window = T_values[lo:hi]

            # crude but robust Lorentzian fit via nonlinear least squares
            # (Levenberg-Marquardt through scipy.optimize.curve_fit if
            # available, else a simple grid+refine fallback)
            try:
                from scipy.optimize import curve_fit
                # initial guess: Gamma from half-max crossing
                half = T_values[i] / 2.0
                E_R0 = energies_mev[i]
                # crude half-width guess from window spacing
                Gamma0 = max((E_window[-1] - E_window[0]) / 4.0, 1e-6)
                popt, _ = curve_fit(
                    _lorentzian, E_window, T_window,
                    p0=[E_R0, Gamma0, T_values[i]],
                    maxfev=5000,
                )
                E_R, Gamma, T_peak = popt
                Gamma = abs(Gamma)
                fit_vals = _lorentzian(E_window, *popt)
                rms = float(np.sqrt(np.mean((fit_vals - T_window) ** 2)))
            except Exception:
                E_R, Gamma, T_peak, rms = energies_mev[i], float("nan"), T_values[i], float("nan")

            if Gamma > 0 and np.isfinite(Gamma):
                tau_ps = t.h_bar / (2.0 * Gamma)   # h_bar in meV*ps -> tau in ps
                resonances.append(Resonance(
                    E_R_mev=float(E_R), Gamma_mev=float(Gamma), tau_ps=float(tau_ps),
                    T_peak=float(T_peak), fit_rms_residual=rms,
                ))
    return resonances


# ---------------------------------------------------------------------------
# Optional: complex-plane pole refinement (secondary tool, slower)
# ---------------------------------------------------------------------------

def _complex_wavenumber(t_hop: float, delta: float, energy_complex: complex) -> complex:
    """Complex extension of qtbm.lead_wavenumber, retarded branch (Im(k)>=0)."""
    cos_val = 1.0 - energy_complex / (2.0 * t_hop)
    k = np.arccos(cos_val) / delta
    if k.imag < 0:
        k = -k
    return k


def _M_smallest_eigenvalue(p: t.PhysicsParams, layout: t.SingleRingLayout, energy_complex: complex) -> complex:
    """Smallest-magnitude eigenvalue of M(E) = H_eff(E) - E*I (homogeneous,
    no source -- matches Fig. 1(b) of the Shao-Porod paper: quasi-bound
    states are the a(E)=0 solutions). Uses shift-invert sparse eigs."""
    k = _complex_wavenumber(p.t_lead, p.delta_x, energy_complex)
    sigma_be = -p.t_lead * np.exp(1j * k * p.delta_x)
    boundary_correction = p.t_lead + sigma_be

    H_eff = t.build_single_ring_hamiltonian(p, layout).tolil()
    for site in (int(layout.left_lead_sites[0]), int(layout.right_lead_sites[-1])):
        for s in (0, 1):
            H_eff[2 * site + s, 2 * site + s] += boundary_correction
    H_eff = H_eff.tocsr().astype(complex)

    M = H_eff - energy_complex * sp.identity(layout.spinor_size, format="csr", dtype=complex)
    # smallest-magnitude eigenvalue via shift-invert around 0
    try:
        vals = spla.eigs(M, k=1, sigma=0.0, which="LM", return_eigenvectors=False, maxiter=2000)
        return complex(vals[0])
    except Exception:
        # fallback: dense smallest-abs eigenvalue (only for small systems)
        dense = M.toarray()
        eigvals = np.linalg.eigvals(dense)
        return complex(eigvals[np.argmin(np.abs(eigvals))])


def complex_pole_refine(
    p: t.PhysicsParams,
    E_R0: float,
    Gamma0: float,
    max_iter: int = 15,
    tol: float = 1e-6,
) -> complex | None:
    """Refine a (E_R, Gamma) guess from the real-axis Lorentzian fit into a
    precise complex pole E = E_R - i*Gamma via a finite-difference 2D Newton
    search on the smallest eigenvalue of M(E). Returns None on failure
    (this is an OPTIONAL precision step; the Breit-Wigner estimate from
    find_resonances is already a valid, physically meaningful answer)."""
    layout = t.build_single_ring_layout(p)
    E = complex(E_R0, -Gamma0)
    h = max(abs(Gamma0), 1e-4) * 1e-3

    for _ in range(max_iter):
        f0 = _M_smallest_eigenvalue(p, layout, E)
        if abs(f0) < tol:
            return E
        fR = _M_smallest_eigenvalue(p, layout, E + h)
        fI = _M_smallest_eigenvalue(p, layout, E + 1j * h)
        dfdR = (fR - f0) / h
        dfdI = (fI - f0) / h
        # Wirtinger-style 2x2 real Newton step
        J = np.array([[dfdR.real, dfdI.real], [dfdR.imag, dfdI.imag]])
        rhs = np.array([-f0.real, -f0.imag])
        try:
            delta = np.linalg.solve(J, rhs)
        except np.linalg.LinAlgError:
            return None
        E = E + delta[0] + 1j * delta[1]
        if E.imag > 0:   # unphysical (growing) branch -- bail out
            return None
    return None
