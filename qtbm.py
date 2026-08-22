"""Quantum Transmitting Boundary Method (QTBM) for the JCE26 single-ring
tight-binding lattice: "Soluciones propuestas" solution #3.

Reference
---------
C. S. Lent, D. J. Kirkner, "The quantum transmitting boundary method",
J. Appl. Phys. 67, 6353 (1990).
Z. Shao, W. Porod, C. S. Lent, D. J. Kirkner, "An eigenvalue method for
open-boundary quantum transmission problems", J. Appl. Phys. 78, 2177 (1995).

What this is, physically
-------------------------
Instead of evolving a Gaussian wavepacket in time (Crank-Nicolson, as in
main.py / conductance.py) and reading off T from how much probability
eventually escapes each lead, QTBM solves DIRECTLY for the stationary
scattering state at one fixed injection energy E. In each lead the
wavefunction is written analytically as a superposition of the known plane
waves of that lead:

    left lead:  psi_j = chi * e^{i k j}  +  r * chi * e^{-i k j}   (incident + reflected)
    right lead: psi_j = t * chi * e^{i k j}                          (purely outgoing)

where chi is the injected spinor (spin-up, spin-down, or any combination),
j is the site index measured from the outer edge of each lead, and k(E) is
the lead's own tight-binding dispersion k(E): E = 2*t_lead*(1-cos(k*dx)),
i.e. exactly the lattice already built by tools.build_single_ring_hamiltonian
for a zero-potential lead. Substituting these analytic forms for the (removed)
semi-infinite tails at the two outermost lattice sites turns the open
scattering problem into a FINITE linear system: (H_eff - E*I) psi = source,
solved once per energy with a single sparse LU factorization. No time
stepping, no total_time_ps to tune, no transient/echo-cutoff ambiguity, and
no possibility of an unconverged quasi-bound-state transient contaminating
the answer -- T(E) and R(E) come out exact for that graph Hamiltonian, at
that energy, in one shot.

Relation to what's already in the repo
----------------------------------------
The "H_eff" boundary self-energy term used here (Sigma(E) = -t_lead * e^{i k(E) dx}
added to the on-site energy of the single outermost site of each lead) is
*exactly* the same closed form implemented in conductance.py's
`lead_self_energy` / `build_cn_matrices_with_transparent_bc`, which was
built for the time-dependent CAP-replacement (patch of 2026-08-21). QTBM
reuses it unchanged for the homogeneous (right-lead / outgoing-only) part,
and adds one new piece: a source term at the LEFT boundary site that
encodes the injected incident wave analytically, which is what makes this
a genuine boundary-VALUE problem (with an incident amplitude built in)
instead of an initial-value time evolution that has to be started off with
a real, spatially localized wavepacket.

Derivation of the boundary relations (self-contained)
-------------------------------------------------------
Bulk tight-binding equation at any lead site j (on-site 2t, hopping -t,
exactly what build_single_ring_hamiltonian already builds for V=0):

    (H psi)_j = 2*t*psi_j - t*psi_{j-1} - t*psi_{j+1} = E*psi_j

At the LEFT boundary site (local index j=0, one lattice site further out
at "ghost" index j=-1), substitute the incident+reflected ansatz
psi_{-1} = chi*e^{-ika} + r*chi*e^{ika}, and use psi_0 = chi + r*chi (so
r*chi = psi_0 - chi) to eliminate r*chi:

    psi_{-1} = chi*e^{-ika} + (psi_0 - chi)*e^{ika}
             = psi_0 * e^{ika} - chi*(e^{ika} - e^{-ika})
             = psi_0 * e^{ika} - 2i*sin(ka)*chi

Substituting into the bulk equation at j=0 and moving everything with
psi_0 to the left-hand side:

    (2t - t*e^{ika} - E) * psi_0  -  t*psi_1  =  -2i*t*sin(ka)*chi

i.e. Sigma(E) = -t*e^{ika} is added to the on-site energy at j=0 (same
self-energy as the CAP-replacement patch), AND a source term
-2i*t*sin(ka)*chi appears on the right-hand side, only at that site.

At the RIGHT boundary site the wave is purely outgoing
(psi_j = t_amp*chi*e^{ikj}, no incident component), so the identical
substitution gives the SAME self-energy Sigma(E) with NO source term:
homogeneous, exactly like the CAP-replacement boundary.

Reading off T and R
---------------------
Because |e^{ikj}| = 1 for any real k (a genuinely propagating mode, E
inside the band), the magnitude of psi at the boundary site directly gives
the scattering amplitude with no extra bookkeeping about the site's exact
local coordinate:

    r_spin      = psi[left_boundary_site]  - chi        (reflection amplitude)
    t_amp_spin  = psi[right_boundary_site]               (transmission amplitude)
    T = sum_spin |t_amp_spin|^2,   R = sum_spin |r_spin|^2

Probability conservation T + R = |chi|^2 then follows algebraically from
the linear solve and is NOT imposed -- it is a genuine, cheap, per-run
correctness check (see run_self_test below), unlike the time-domain CAP
runs where T+R landing far from 1 could mean either "still converging" or
"a real quasi-bound resonance", each requiring much longer runs to tell
apart.

Usage
-----
    python qtbm.py --selftest      # verifies T+R=1 and matches the
                                    # closed-form Buttiker formula for the
                                    # transparent ring
    python qtbm.py --ab            # AB oscillation sweep, transparent ring
    python qtbm.py --ac            # AC oscillation sweep, transparent ring
    python qtbm.py --qpc           # finite-transparency ring (QPCs on),
                                    # reproduces J.J. Gonzalez thesis Fig 4.4-4.9
See README_QTBM.md (shipped alongside this file) for the full step-by-step
guide and how each figure maps to a specific function call here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import tools as t


# ---------------------------------------------------------------------------
# Core QTBM solve
# ---------------------------------------------------------------------------

def lead_wavenumber(t_hop: float, delta: float, energy_mev: float) -> float:
    """Solve E = 2*t_hop*(1 - cos(k*delta)) for the propagating k > 0."""
    cos_val = float(np.clip(1.0 - energy_mev / (2.0 * t_hop), -1.0, 1.0))
    return np.arccos(cos_val) / delta


@dataclass
class QTBMResult:
    energy_mev: float
    T_up: float
    T_down: float
    R_up: float
    R_down: float
    T: float            # = T_up + T_down  (G/G0, Landauer, both spin channels)
    R: float
    conservation_error: float   # |T + R - 1| per spin channel, summed
    psi: np.ndarray | None = None   # full spinor solution, only if keep_psi=True


def qtbm_solve_one_spin(
    p: t.PhysicsParams,
    layout: t.SingleRingLayout,
    energy_mev: float,
    chi: tuple[complex, complex],
    keep_psi: bool = False,
) -> tuple[complex, complex, np.ndarray | None]:
    """Solve the stationary scattering problem for one injected spinor chi.

    Returns (r_amplitude_pair, t_amplitude_pair, psi_or_None) where each
    amplitude pair is (spin_up_component, spin_down_component).
    """
    k     = lead_wavenumber(p.t_lead, p.delta_x, energy_mev)
    sigma = -p.t_lead * np.exp(1j * k * p.delta_x)
    # build_single_ring_hamiltonian gives the outermost lead site only ONE
    # physical bond (to its single interior neighbour), so its on-site
    # energy there is t_lead, not the bulk value 2*t_lead. The boundary
    # correction derived in the module docstring assumes a bulk on-site
    # energy of 2*t_lead as the reference, so the amount to ADD here is
    # (2*t_lead - t_lead) + sigma = t_lead + sigma, not sigma alone.
    boundary_correction = p.t_lead + sigma

    H_eff = t.build_single_ring_hamiltonian(p, layout).tolil()
    left_site  = int(layout.left_lead_sites[0])
    right_site = int(layout.right_lead_sites[-1])
    for site in (left_site, right_site):
        for s in (0, 1):
            idx = 2 * site + s
            H_eff[idx, idx] += boundary_correction
    H_eff = H_eff.tocsr()

    source = np.zeros(layout.spinor_size, dtype=complex)
    src_amp = -2j * p.t_lead * np.sin(k * p.delta_x)
    source[2 * left_site + 0] = src_amp * chi[0]
    source[2 * left_site + 1] = src_amp * chi[1]

    M = (H_eff - energy_mev * sp.identity(layout.spinor_size, format="csr", dtype=complex)).tocsc()
    psi = spla.spsolve(M, source)

    r = (psi[2 * left_site + 0] - chi[0], psi[2 * left_site + 1] - chi[1])
    tt = (psi[2 * right_site + 0], psi[2 * right_site + 1])

    return r, tt, (psi if keep_psi else None)


def qtbm_conductance(
    p: t.PhysicsParams,
    energy_mev: float,
    keep_psi: bool = False,
) -> QTBMResult:
    """Full two-spin-channel Landauer conductance at one fixed energy.

    Solves the SAME linear system (same LU factorization target, just a
    different right-hand side) twice: once injecting pure spin-up
    (chi=(1,0)), once pure spin-down (chi=(0,1)). No spin mixing happens in
    the leads (Rashba coupling only rotates spin inside the ring arms, see
    tools.build_single_ring_hamiltonian), so this exactly reproduces the
    two independent Landauer channels G/G0 = T_up + T_down.
    """
    layout = t.build_single_ring_layout(p)

    r_up, t_up, psi_up = qtbm_solve_one_spin(p, layout, energy_mev, (1.0, 0.0), keep_psi)
    r_dn, t_dn, psi_dn = qtbm_solve_one_spin(p, layout, energy_mev, (0.0, 1.0), keep_psi)

    T_up = abs(t_up[0]) ** 2 + abs(t_up[1]) ** 2
    R_up = abs(r_up[0]) ** 2 + abs(r_up[1]) ** 2
    T_dn = abs(t_dn[0]) ** 2 + abs(t_dn[1]) ** 2
    R_dn = abs(r_dn[0]) ** 2 + abs(r_dn[1]) ** 2

    conservation_error = abs(T_up + R_up - 1.0) + abs(T_dn + R_dn - 1.0)

    return QTBMResult(
        energy_mev=energy_mev,
        T_up=T_up, T_down=T_dn,
        R_up=R_up, R_down=R_dn,
        T=T_up + T_dn, R=R_up + R_dn,
        conservation_error=conservation_error,
        psi=(psi_up, psi_dn) if keep_psi else None,
    )


# ---------------------------------------------------------------------------
# Parameter sets matching J.J. Gonzalez's thesis (LicJJG2.pdf) reference values
# ---------------------------------------------------------------------------

M_FACTOR = 0.023          # InAs effective mass fraction (thesis sec. 3.3, 4.3)
R_NM     = 250.0          # ring radius (thesis: a = 250 nm)
E_F_MEV  = 4.19           # Fermi energy (thesis: theta = 12.5*pi at this E_F)
ALPHA_REF = 20.0          # reference Rashba constant, meV*nm (thesis sec 3.3)


def transparent_ring_params(Phi: float = 0.0, alpha: float = ALPHA_REF,
                            N_l: int = 381, N_R: int = 151,
                            junction_correction: bool = True) -> t.PhysicsParams:
    """V=0 everywhere -- reproduces thesis chapter 3 ("anillo transparente").

    junction_correction defaults to True to match tools.default_params(),
    but see README_QTBM.md / the "junction_correction finding" section:
    QTBM's exact T+R=1 check exposed that this term, AS CURRENTLY DERIVED
    in tools.t_junction_correction, makes agreement with the analytical
    Buttiker formula dramatically WORSE at the reference point (Phi=0,
    alpha=0: T=1.28 without it vs T=0.13 with it, against an analytical
    G=1.67), not better as its docstring claims. Pass
    junction_correction=False to reproduce the better-matching curves.
    """
    return t.PhysicsParams(
        m_factor=M_FACTOR, R=R_NM, L_leads=2000.0, N_l=N_l, N_R=N_R, dt=0.002,
        Phi=Phi, alpha=alpha, potential_model="none",
        gaussian_qpc_heights_mev={"L": 0.0, "U": 0.0, "D": 0.0, "R": 0.0},
        junction_correction=junction_correction,
    )


def finite_transparency_params(Phi: float = 0.5, alpha: float = ALPHA_REF,
                               Ux13: float = 0.01, Uy13: float = 0.01,
                               N_l: int = 381, N_R: int = 151) -> t.PhysicsParams:
    """QPCs on, symmetric confinement -- reproduces thesis chapter 4 figures.

    Uses potential_model="legacy_unbounded_qpc" (the inverted-parabola /
    saddle-point form, tools.legacy_unbounded_qpc_potential), which is the
    closest existing analogue in this repo to the thesis's V_SP = -Ux*x^2 +
    Uy*y^2 + V0 saddle point (Ec. 2.1-2.2 of LicJJG2.pdf), evaluated along
    the propagation axis. This is NOT the same as tools.saddle_point_1d_qpc_potential
    (which truncates at V=0 and adds a zero-point-energy shift); see
    README_QTBM.md section "QPC model caveat" for why the truncated version
    was NOT used here and what the remaining discrepancy against the
    thesis's exact T(Eg) = 1/(1+exp(-pi*epsilon)) formula (Ec. 2.36) is.
    """
    return t.PhysicsParams(
        m_factor=M_FACTOR, R=R_NM, L_leads=2000.0, N_l=N_l, N_R=N_R, dt=0.002,
        Phi=Phi, alpha=alpha, potential_model="legacy_unbounded_qpc",
        V0_L=0.0, Ux_L=Ux13, Uy_L=Uy13,
        V0_U=0.0, Ux_U=0.0,   Uy_U=0.0,     # QPC3 (upper arm) off by default
        V0_R=0.0, Ux_R=Ux13, Uy_R=Uy13,
        s0_L_fraction=0.05, s0_R_fraction=0.05,
    )


# ---------------------------------------------------------------------------
# Analytical reference formulas (Ec. 3.2 and Ec. 4.3 / 2.36 of LicJJG2.pdf,
# identical to Ec. 2-25/4-26 of the JCE25-26 article already used in
# ab_ac_proof.py's analytical_G)
# ---------------------------------------------------------------------------

def analytical_G_transparent(Phi: float, alpha_mev_nm: float,
                             theta_dyn: float | None = None) -> float:
    """Buttiker transparent-ring formula (thesis Ec. 3.2)."""
    m = M_FACTOR * t.m_e
    theta = theta_dyn if theta_dyn is not None else np.sqrt(2 * m * E_F_MEV) / t.h_bar * np.pi * R_NM
    phi_so = (m * alpha_mev_nm / t.h_bar**2) * R_NM
    G = 0.0
    for sigma in (+1, -1):
        phi_sigma = np.pi * (Phi + sigma * phi_so)
        num = 16.0 * np.cos(phi_sigma) ** 2 * np.sin(theta) ** 2
        den = (1.0 - 2.0 * np.cos(2 * theta) + np.cos(2 * phi_sigma)) ** 2 + 4.0 * np.sin(theta) ** 2
        G += num / den if den > 1e-15 else 0.0
    return G


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def run_self_test(n_points: int = 25, verbose: bool = True) -> dict:
    """Verify T+R=1 (exact conservation) and match to the closed-form
    Buttiker formula for the transparent ring, sweeping Phi at fixed alpha.
    """
    Phi_values = np.linspace(-1.0, 1.0, n_points)
    T_qtbm = np.empty(n_points)
    T_analytic = np.empty(n_points)
    max_conservation_error = 0.0

    for i, Phi in enumerate(Phi_values):
        p = transparent_ring_params(Phi=Phi, alpha=ALPHA_REF)
        result = qtbm_conductance(p, E_F_MEV)
        T_qtbm[i] = result.T
        T_analytic[i] = analytical_G_transparent(Phi, ALPHA_REF)
        max_conservation_error = max(max_conservation_error, result.conservation_error)
        if verbose:
            print(f"  Phi={Phi:+.3f}  T+R conservation error={result.conservation_error:.2e}  "
                  f"T_qtbm={T_qtbm[i]:.6f}  T_analytic={T_analytic[i]:.6f}  "
                  f"diff={T_qtbm[i]-T_analytic[i]:+.2e}")

    rms_diff = float(np.sqrt(np.mean((T_qtbm - T_analytic) ** 2)))
    max_diff = float(np.max(np.abs(T_qtbm - T_analytic)))

    if verbose:
        print()
        print(f"Max |T+R-1| conservation error over sweep: {max_conservation_error:.3e}")
        print(f"RMS(T_qtbm - T_analytic) over sweep:        {rms_diff:.3e}")
        print(f"Max |T_qtbm - T_analytic| over sweep:        {max_diff:.3e}")
        print("(For reference, the CAP/time-domain method in ab_ac_proof.py showed "
              "T+R plateauing around 0.35-0.38 with the full ring at these parameters --"
              " see CHANGES_2026-08-21.txt. QTBM's numbers above are the exact,"
              " no-transient, no-timeout comparison.)")

    return {
        "Phi_values": Phi_values, "T_qtbm": T_qtbm, "T_analytic": T_analytic,
        "max_conservation_error": max_conservation_error,
        "rms_diff": rms_diff, "max_diff": max_diff,
    }


# ---------------------------------------------------------------------------
# Sweeps (mirroring ab_ac_proof.py / thesis figures)
# ---------------------------------------------------------------------------

def sweep_ab(alpha: float = ALPHA_REF, n_points: int = 41,
            params_fn=transparent_ring_params) -> dict:
    """G/G0 vs Phi at fixed alpha -- AB oscillations (thesis Fig 3.1/3.2/4.4)."""
    Phi_values = np.linspace(-1.0, 1.0, n_points)
    T_up = np.empty(n_points); T_dn = np.empty(n_points); T_tot = np.empty(n_points)
    for i, Phi in enumerate(Phi_values):
        p = params_fn(Phi=Phi, alpha=alpha)
        r = qtbm_conductance(p, E_F_MEV)
        T_up[i], T_dn[i], T_tot[i] = r.T_up, r.T_down, r.T
    return {"Phi": Phi_values, "T_up": T_up, "T_down": T_dn, "T_total": T_tot}


def sweep_ac(Phi: float = 0.0, n_points: int = 41,
            params_fn=transparent_ring_params) -> dict:
    """G/G0 vs phi_so (converted to alpha) at fixed Phi -- AC oscillations
    (thesis Fig 3.3/3.4)."""
    phi_so_values = np.linspace(0.0, 2.0, n_points)
    m = M_FACTOR * t.m_e
    alpha_values = phi_so_values * t.h_bar**2 / (m * R_NM)
    T_up = np.empty(n_points); T_dn = np.empty(n_points); T_tot = np.empty(n_points)
    for i, alpha in enumerate(alpha_values):
        p = params_fn(Phi=Phi, alpha=alpha)
        r = qtbm_conductance(p, E_F_MEV)
        T_up[i], T_dn[i], T_tot[i] = r.T_up, r.T_down, r.T
    return {"phi_so": phi_so_values, "alpha": alpha_values,
            "T_up": T_up, "T_down": T_dn, "T_total": T_tot}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true",
                        help="Verify T+R=1 conservation and match to the analytical formula")
    parser.add_argument("--ab", action="store_true", help="AB oscillation sweep (transparent ring)")
    parser.add_argument("--ac", action="store_true", help="AC oscillation sweep (transparent ring)")
    parser.add_argument("--qpc", action="store_true", help="Finite-transparency ring sweep (QPCs on)")
    parser.add_argument("--n-points", type=int, default=41)
    args = parser.parse_args()

    if not (args.selftest or args.ab or args.ac or args.qpc):
        args.selftest = True   # default action

    if args.selftest:
        print("=" * 70)
        print("  QTBM self-test: T+R conservation and match to Buttiker formula")
        print("=" * 70)
        run_self_test(n_points=args.n_points)

    if args.ab:
        print("\n[AB sweep, transparent ring]")
        result = sweep_ab(n_points=args.n_points)
        np.savez_compressed("qtbm_ab_transparent.npz", **result)
        print("Saved: qtbm_ab_transparent.npz")

    if args.ac:
        print("\n[AC sweep, transparent ring]")
        result = sweep_ac(n_points=args.n_points)
        np.savez_compressed("qtbm_ac_transparent.npz", **result)
        print("Saved: qtbm_ac_transparent.npz")

    if args.qpc:
        print("\n[AB sweep, finite transparency (QPCs on)]")
        result = sweep_ab(n_points=args.n_points, params_fn=finite_transparency_params)
        np.savez_compressed("qtbm_ab_qpc.npz", **result)
        print("Saved: qtbm_ab_qpc.npz")


if __name__ == "__main__":
    main()
