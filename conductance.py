"""Landauer conductance calculation and parameter-sweep engine for the JCE26 ring.

Usage
-----
Run directly as a script for a quick demo sweep::

    python conductance.py

Or import and call ``sweep_conductance`` / ``run_single_conductance`` from your
own driver scripts.

Physics of the transmission coefficient
----------------------------------------
We use a wavepacket scattering approach in the time domain:

1. **Initial norm** (denominator):
       N₀ = Σᵢ |ψᵢ(t=0)|² · wᵢ
   where wᵢ are the trapezoidal quadrature weights (nm) stored in
   ``layout.integration_weights_nm``.  Using the physical weights is important
   because the ring spacing δs ≠ δx of the leads.

2. **Right-lead integrated probability** at each time step:
       P_R(t) = Σᵢ∈R |ψᵢ(t)|² · wᵢ

3. **Transmission coefficient**:
       T = max_{t < t_echo} P_R(t) / N₀
   The maximum is taken only *before* the first hard-wall echo returns to the
   scattering region (estimated by ``estimate_reflection_horizons_ps`` in
   main.py).  Using the global maximum would be fine for most runs, but gating
   on t_echo is more principled and avoids contamination from boundary
   reflections.

4. **Landauer conductance**:
       G = G₀ · T,   G₀ = e²/h ≈ 3.874 × 10⁻⁵ S
   expressed in units of G₀ (so G/G₀ = T).

Note on spin: the spinor wavefunction already carries both spin components and
their mixing via Rashba, so no extra factor of 2 is added.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

import tools as t


# ---------------------------------------------------------------------------
# Physical constant
# ---------------------------------------------------------------------------
G0_SIEMENS = 3.87404e-5   # e²/h in Siemens (quantum of conductance)

# ---------------------------------------------------------------------------
# Complex Absorbing Potential (CAP) utilities
# ---------------------------------------------------------------------------
# The CAP adds a negative-imaginary on-site term -i*W(x) to the outer fraction
# of each lead.  This makes H intentionally non-Hermitian:
#   - Norm lost in the LEFT  CAP region  =  reflected probability R
#   - Norm lost in the RIGHT CAP region  =  transmitted probability T
# Integrating the absorption over time gives R and T without needing to gate
# on an echo cutoff, and the simulation can run for as long as desired.
#
# W(x) rises as a smooth monomial ramp over the outer cap_fraction of the lead:
#   W(x) = cap_strength * ((x - x_start) / cap_length)^cap_order   (meV)
#
# The CAP is disabled by default (cap_strength=0); set cap_strength > 0 to use it.

def build_cap_vector(
    layout: t.SingleRingLayout,
    p: t.PhysicsParams,
    cap_fraction: float = 0.25,
    cap_strength: float = 0.0,
    cap_order: int = 3,
) -> np.ndarray:
    """Return a real vector W [meV] of CAP absorption strengths, one per site.

    Only the outer ``cap_fraction`` of each lead is absorbing.  All ring and
    junction sites have W = 0.

    Parameters
    ----------
    cap_fraction : float
        Fraction of each lead length covered by the CAP ramp (default 0.25).
    cap_strength : float
        Peak absorption strength in meV (default 0 = disabled).
    cap_order : int
        Polynomial order of the ramp (default 3; higher = sharper onset).
    """
    n_sites = layout.unique_site_count
    W       = np.zeros(n_sites, dtype=float)
    if cap_strength <= 0.0:
        return W

    n_lead = layout.left_lead_sites.size    # = p.N_l
    n_cap  = max(1, int(np.ceil(cap_fraction * n_lead)))

    # Left lead: absorber at the FAR end (sites 0..n_cap-1)
    for i in range(n_cap):
        ramp_coord = (n_cap - 1 - i) / (n_cap - 1) if n_cap > 1 else 1.0
        W[layout.left_lead_sites[i]] = cap_strength * ramp_coord**cap_order

    # Right lead: absorber at the FAR end (last n_cap sites)
    for i in range(n_cap):
        ramp_coord = i / (n_cap - 1) if n_cap > 1 else 1.0
        W[layout.right_lead_sites[-(i + 1)]] = cap_strength * ramp_coord**cap_order

    return W


def build_cn_matrices_with_cap(
    p: t.PhysicsParams,
    layout: t.SingleRingLayout,
    cap_vector: np.ndarray,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Return (A, B) Crank-Nicolson matrices for a non-Hermitian H + CAP.

    H_eff = H_physical - i * diag(W)  (spinor-expanded)

    The spinor expansion repeats each site's W for both spin components.
    """
    H_phys = t.build_single_ring_hamiltonian(p, layout)

    # Expand CAP to spinor space: site s -> rows 2s and 2s+1
    W_spinor = np.repeat(cap_vector, 2)
    H_cap    = sp.diags(-1j * W_spinor, format="csr")
    H_eff    = H_phys + H_cap

    identity  = sp.identity(layout.spinor_size, format="csr", dtype=complex)
    prefactor = 1j * p.dt / (2 * t.h_bar)
    A = (identity + prefactor * H_eff).tocsr()
    B = (identity - prefactor * H_eff).tocsr()
    return A, B


# ---------------------------------------------------------------------------
# Exact single-site lead self-energy boundary condition ("transparent BC")
# ---------------------------------------------------------------------------
# This targets the same problem as the CAP above (spurious reflections at the
# truncated ends of the leads, section 4.3/5.2 of the JCE25-26 article) but
# instead of an approximate, finite-width absorbing ramp it embeds the EXACT
# retarded self-energy of the removed semi-infinite continuation of the lead
# at a single boundary site, evaluated at the injection (Fermi) energy.
#
# Derivation (standard tight-binding lead embedding, e.g. Datta, "Electronic
# Transport in Mesoscopic Systems", or any NEGF/quantum-transport reference):
# a uniform 1-D chain with on-site energy 2t and hopping -t (exactly the bulk
# form produced by ``build_single_ring_hamiltonian`` for a zero-potential
# lead) has surface Green's function g_s(E) solving the Dyson equation
#     g_s(E) = [E - 2t - t^2 g_s(E)]^-1.
# Writing E = 2t(1 - cos(k*delta_x)) (the exact tight-binding dispersion of
# this lattice) and selecting the retarded (outgoing-wave) branch gives the
# closed form self-energy of the truncated semi-infinite tail:
#     Sigma(E) = t^2 * g_s(E) = -t * exp(i * k(E) * delta_x).
# Adding Sigma(E_F) to the on-site energy of the single outermost site of
# each lead makes that site behave EXACTLY as if the lead continued to
# infinity, for a wave at energy E_F: Im(Sigma) < 0 gives the exact escape
# rate (no reflection at all for that Fourier component), and no finite
# absorbing region or extra lead padding is required.
#
# Caveat (documented honestly, see also the project notes on open problems):
# this is a MONOCHROMATIC / memoryless approximation to the full discrete
# transparent boundary condition (DTBC) recommended in the literature review
# (Akramov et al. 2026; Arnold-Ehrhardt 1999), which uses a time-convolution
# kernel and is exact for the *entire* wavepacket, not just its central
# Fourier component. Residual reflection here is expected to scale with the
# wavepacket's momentum spread Delta_k / k_F, not with an ad hoc ramp shape.
# Implementing the full convolution kernel remains future work (see the
# CHANGES txt shipped with this patch).

def lead_wavenumber(p: t.PhysicsParams, energy_mev: float) -> float:
    """Solve E = 2*t_lead*(1 - cos(k*delta_x)) for the propagating k > 0.

    Uses p.t_lead / p.delta_x, i.e. the LEFT/RIGHT lead discretization, which
    is what the boundary sites actually live on.
    """
    cos_val = 1.0 - energy_mev / (2.0 * p.t_lead)
    cos_val = float(np.clip(cos_val, -1.0, 1.0))
    return np.arccos(cos_val) / p.delta_x


def lead_self_energy(p: t.PhysicsParams, k: float) -> complex:
    """Exact retarded self-energy of the truncated semi-infinite lead tail."""
    return -p.t_lead * np.exp(1j * k * p.delta_x)


def build_cn_matrices_with_transparent_bc(
    p: t.PhysicsParams,
    layout: t.SingleRingLayout,
    energy_mev: float,
) -> tuple[sp.csr_matrix, sp.csr_matrix, complex]:
    """Return (A, B, Sigma) with the exact single-site self-energy boundary.

    Sigma is embedded at exactly one site per lead: the outermost node
    (``left_lead_sites[0]`` and ``right_lead_sites[-1]``), for both spin
    components. No ramp / absorbing region is needed.
    """
    k     = lead_wavenumber(p, energy_mev)
    sigma = lead_self_energy(p, k)

    H_eff = t.build_single_ring_hamiltonian(p, layout).tolil()
    boundary_sites = (int(layout.left_lead_sites[0]), int(layout.right_lead_sites[-1]))
    for site in boundary_sites:
        for spin in (0, 1):
            idx = 2 * site + spin
            H_eff[idx, idx] += sigma
    H_eff = H_eff.tocsr()

    identity  = sp.identity(layout.spinor_size, format="csr", dtype=complex)
    prefactor = 1j * p.dt / (2 * t.h_bar)
    A = (identity + prefactor * H_eff).tocsr()
    B = (identity - prefactor * H_eff).tocsr()
    return A, B, sigma



# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ConductanceResult:
    """Everything produced by a single-parameter-set conductance calculation."""

    params: t.PhysicsParams
    fermi_energy_mev: float

    # Transmission and conductance
    T: float                  # dimensionless transmission coefficient [0, 1]
    G_over_G0: float          # G / G₀  (= T for two-terminal Landauer)
    G_siemens: float          # absolute conductance in Siemens

    # Diagnostics
    N0_weighted: float        # initial norm with integration weights
    P_R_max_weighted: float   # max right-lead probability (weighted)
    P_R_max_time_ps: float    # time at which P_R peaked
    echo_cutoff_ps: float     # time horizon used for the max search
    relative_norm_drift: float  # (P_total_final - P_total_initial) / P_total_initial
    wall_seconds: float

    # Optional: full time series for diagnostics / plotting
    time_axis_ps: np.ndarray | None = None
    P_R_history: np.ndarray | None  = None
    P_total_history: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha_mev_nm": self.params.alpha,
            "V0_U": self.params.V0_U,
            "Ux_U": self.params.Ux_U,
            "Uy_U": self.params.Uy_U,
            "V0_L": self.params.V0_L,
            "Ux_L": self.params.Ux_L,
            "Uy_L": self.params.Uy_L,
            "V0_R": self.params.V0_R,
            "Ux_R": self.params.Ux_R,
            "Uy_R": self.params.Uy_R,
            "potential_model": self.params.potential_model,
            "fermi_energy_mev": self.fermi_energy_mev,
            "T": self.T,
            "G_over_G0": self.G_over_G0,
            "G_siemens": self.G_siemens,
            "N0_weighted": self.N0_weighted,
            "P_R_max_weighted": self.P_R_max_weighted,
            "P_R_max_time_ps": self.P_R_max_time_ps,
            "echo_cutoff_ps_used": self.echo_cutoff_ps,
            "relative_norm_drift": self.relative_norm_drift,
            "wall_seconds": self.wall_seconds,
        }


@dataclass
class SweepResult:
    """Collection of ConductanceResult objects from a parameter sweep."""

    sweep_parameter: str          # e.g. "alpha" or "V0_U"
    sweep_values: list[float]
    results: list[ConductanceResult] = field(default_factory=list)

    @property
    def T_values(self) -> np.ndarray:
        return np.array([r.T for r in self.results])

    @property
    def G_over_G0_values(self) -> np.ndarray:
        return np.array([r.G_over_G0 for r in self.results])

    def to_table(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.results]


# ---------------------------------------------------------------------------
# Core conductance calculation for a single parameter set
# ---------------------------------------------------------------------------

def compute_wave_number(p: t.PhysicsParams, fermi_energy_mev: float) -> float:
    """Convert Fermi energy to lead wave-number."""
    return np.sqrt(2 * p.m * fermi_energy_mev) / t.h_bar


def build_initial_wavefunction(
    p: t.PhysicsParams,
    layout: t.SingleRingLayout,
    k: float,
    packet_center_fraction: float = 0.8,
    packet_width_nm: float = 150.0,
    spin: str = "up",
) -> np.ndarray:
    """Gaussian wavepacket in the left lead moving towards the ring.

    Parameters
    ----------
    spin : str
        "up"   – inject spin-up only  (spinor [1, 0])
        "down" – inject spin-down only (spinor [0, 1])
        "both" – inject equal superposition [1/√2, 1/√2], so that both
                 channels contribute and G/G₀ can reach 2.
                 Equivalent to summing two independent runs.

    Notes
    -----
    The Landauer conductance G/G₀ = T_up + T_down.  A spin-polarised packet
    can only probe one channel (max G/G₀ = 1).  To measure the full
    two-channel conductance without running the simulation twice, use
    ``spin="both"`` and divide the resulting T by 0.5 (since both channels
    contribute equally to the norm).  Alternatively, use
    ``run_both_spin_channels`` which does the two runs explicitly and sums T.
    """
    psi = np.zeros(layout.spinor_size, dtype=complex)
    left_positions_nm = np.arange(layout.left_lead_sites.size, dtype=float) * p.delta_x
    center_nm = packet_center_fraction * p.L_leads
    width_nm  = max(3.0 * p.delta_x, packet_width_nm)

    if spin == "up":
        chi = np.array([1.0, 0.0], dtype=complex)
    elif spin == "down":
        chi = np.array([0.0, 1.0], dtype=complex)
    elif spin == "both":
        chi = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    else:
        raise ValueError(f"spin must be 'up', 'down', or 'both', got {spin!r}")

    for site, x_nm in zip(layout.left_lead_sites, left_positions_nm):
        amp = (np.exp(-0.5 * ((x_nm - center_nm) / width_nm) ** 2)
               * np.exp(1j * k * x_nm))
        psi[2 * site]     = amp * chi[0]
        psi[2 * site + 1] = amp * chi[1]

    return psi


def estimate_echo_cutoff_ps(
    p: t.PhysicsParams,
    k: float,
    packet_center_fraction: float = 0.8,
) -> float:
    """Estimate the earliest time a hard-wall echo could contaminate the device.

    We return the smaller of the left-echo and right-echo return times so that
    the max(P_R) search is always within the clean scattering window.
    """
    v_nm_ps = t.lead_group_velocity(p, k)
    center_nm = packet_center_fraction * p.L_leads

    # Left boundary echo: packet travels to left wall and back to left junction
    left_echo = (center_nm + p.L_leads) / v_nm_ps

    # Right boundary echo: packet crosses ring, hits right wall, returns to device
    right_echo = (
        (p.L_leads - center_nm + p.L_ring) / v_nm_ps   # to right junction
        + 2.0 * p.L_leads / v_nm_ps                     # right wall and back
    )

    return min(left_echo, right_echo)


def _weighted_right_lead_probability(
    psi_spinor: np.ndarray,
    layout: t.SingleRingLayout,
) -> float:
    """Weighted integral of |ψ|² over the right lead (excluding junction)."""
    # Sites to integrate: right lead excluding the junction node itself
    sites = layout.right_lead_sites[1:]
    weights = layout.integration_weights_nm[sites]
    spinor_block = psi_spinor.reshape(-1, 2)[sites]       # shape (n_sites, 2)
    density      = np.sum(np.abs(spinor_block)**2, axis=1) # shape (n_sites,)
    return float(np.dot(density, weights))


def _weighted_total_probability(
    psi_spinor: np.ndarray,
    layout: t.SingleRingLayout,
) -> float:
    """Weighted integral of |ψ|² over the entire graph."""
    weights     = layout.integration_weights_nm            # shape (n_sites,)
    spinor_block = psi_spinor.reshape(-1, 2)               # shape (n_sites, 2)
    density      = np.sum(np.abs(spinor_block)**2, axis=1)
    return float(np.dot(density, weights))


def run_single_conductance(
    p: t.PhysicsParams,
    fermi_energy_mev: float = 4.19,
    total_time_ps: float = 13.5,
    packet_center_fraction: float = 0.8,
    packet_width_nm: float = 150.0,
    keep_time_series: bool = False,
    verbose: bool = True,
    spin: str = "both",
) -> ConductanceResult:
    """Run the full Crank-Nicolson evolution and return a ConductanceResult.

    Parameters
    ----------
    p:
        The complete parameter set for this run.
    fermi_energy_mev:
        Injection energy.
    total_time_ps:
        Physical duration of the simulation.
    packet_center_fraction:
        Where in the left lead the Gaussian packet starts (0=junction, 1=wall).
    packet_width_nm:
        Gaussian width of the initial packet.
    keep_time_series:
        If True, the full P_R(t) and P_total(t) arrays are stored in the result.
    verbose:
        Print progress to stdout.
    spin : str
        Which spin channel(s) to inject: "up", "down", or "both" (default).
        Using "both" injects an equal superposition and scales T by 2 so that
        G/G₀ = T_up + T_down (can reach 2). For the single-channel Landauer
        formula (one spin), use "up" or "down" (G/G₀ ≤ 1).
    """
    wall_start = time.perf_counter()

    layout     = t.build_single_ring_layout(p)
    k          = compute_wave_number(p, fermi_energy_mev)
    time_steps = t.time_steps_for_duration(p, total_time_ps)
    echo_cutoff_ps = estimate_echo_cutoff_ps(p, k, packet_center_fraction)

    if verbose:
        print(f"  α={p.alpha:.1f} meV·nm | V0_U={p.V0_U:.4f} Ux_U={p.Ux_U:.4f}"
              f" | k={k:.4f} nm⁻¹ | echo_cutoff={echo_cutoff_ps:.2f} ps")

    A, B, _ = t.build_cn_matrices(p, layout)
    solver   = spla.factorized(A.tocsc())

    psi = build_initial_wavefunction(
        p, layout, k, packet_center_fraction, packet_width_nm, spin=spin)
    N0  = _weighted_total_probability(psi, layout)
    # When spin="both" we inject 2 channels at once. The resulting T measures
    # the average of both channels. Multiply by 2 to get G/G₀ = T_up + T_down.
    _spin_scale = 2.0 if spin == "both" else 1.0

    # --- time evolution with running P_R tracking ---------------------------
    P_R_max        = 0.0
    P_R_max_step   = 0
    echo_step      = min(time_steps, int(np.ceil(echo_cutoff_ps / p.dt)))

    P_R_series     = np.empty(time_steps + 1) if keep_time_series else None
    P_total_series = np.empty(time_steps + 1) if keep_time_series else None

    if keep_time_series:
        P_R_series[0]     = _weighted_right_lead_probability(psi, layout)
        P_total_series[0] = N0

    for step in range(time_steps):
        psi = solver(B @ psi)

        if keep_time_series:
            p_r = _weighted_right_lead_probability(psi, layout)
            P_R_series[step + 1]     = p_r
            P_total_series[step + 1] = _weighted_total_probability(psi, layout)
        elif step <= echo_step:
            p_r = _weighted_right_lead_probability(psi, layout)
        else:
            p_r = 0.0   # beyond echo window: skip the computation

        if step <= echo_step and p_r > P_R_max:
            P_R_max      = p_r
            P_R_max_step = step + 1

    # Final total probability for norm-drift diagnostic
    P_total_final = _weighted_total_probability(psi, layout)

    # T_raw is the fraction of the injected probability that was transmitted.
    # For spin="both": each spin contributes independently, so G/G₀ = 2 * T_raw.
    T_raw     = P_R_max / N0 if N0 > 0 else 0.0
    T         = min(_spin_scale * T_raw, 2.0)  # clamp to physical maximum
    G_over_G0 = T
    G_siemens = T * G0_SIEMENS

    wall_seconds = time.perf_counter() - wall_start

    if verbose:
        print(f"    → T={T:.6f}  G/G₀={G_over_G0:.6f}  "
              f"(spin={spin}, scale={_spin_scale})  [{wall_seconds:.1f} s]")

    return ConductanceResult(
        params=p,
        fermi_energy_mev=fermi_energy_mev,
        T=T,
        G_over_G0=G_over_G0,
        G_siemens=G_siemens,
        N0_weighted=N0,
        P_R_max_weighted=P_R_max,
        P_R_max_time_ps=P_R_max_step * p.dt,
        echo_cutoff_ps=echo_cutoff_ps,
        relative_norm_drift=(P_total_final - N0) / N0 if N0 > 0 else 0.0,
        wall_seconds=wall_seconds,
        time_axis_ps=(np.arange(time_steps + 1, dtype=float) * p.dt
                      if keep_time_series else None),
        P_R_history=P_R_series,
        P_total_history=P_total_series,
    )



# ---------------------------------------------------------------------------
# CAP-based conductance (open-boundary scattering)
# ---------------------------------------------------------------------------

@dataclass
class CAPConductanceResult:
    """Result from a CAP-based conductance measurement."""
    params: t.PhysicsParams
    fermi_energy_mev: float
    T: float            # transmission coefficient
    R: float            # reflection coefficient
    T_plus_R: float     # should be ~1.0 if CAP is well-tuned
    G_over_G0: float
    G_siemens: float
    total_time_ps: float
    wall_seconds: float
    # Optional time series
    time_axis_ps: np.ndarray | None = None
    P_left_cap_rate: np.ndarray | None = None   # dR/dt
    P_right_cap_rate: np.ndarray | None = None  # dT/dt


def run_cap_conductance(
    p: t.PhysicsParams,
    fermi_energy_mev: float = 4.19,
    total_time_ps: float = 30.0,
    packet_center_fraction: float = 0.8,
    packet_width_nm: float = 150.0,
    cap_fraction: float = 0.25,
    cap_strength: float = 2.0,
    cap_order: int = 3,
    keep_time_series: bool = False,
    verbose: bool = True,
    spin_both: bool = True,
) -> "CAPConductanceResult":
    """Measure conductance using a Complex Absorbing Potential.

    The CAP absorbs outgoing probability at both ends of each lead.
    Norm absorbed in the right CAP = transmitted probability T.
    Norm absorbed in the left  CAP = reflected probability R.
    T + R should equal 1 (verify this as a diagnostic).

    The simulation can run much longer than the echo_cutoff because the
    echoes are absorbed before they can return to the scattering region.

    Parameters
    ----------
    total_time_ps : float
        Run duration.  Should be long enough for the packet to fully clear
        the scattering region and be absorbed.  Typical: ~35 ps for the
        InAs ring (packet transit ~5 ps + CAP absorption time).
    cap_fraction : float
        Fraction of each lead covered by the absorber (default 0.25).
    cap_strength : float
        Peak CAP strength in meV.  Rule of thumb: 1-5 meV.  Too small leaves
        reflections; too large causes artificial back-reflection at the ramp onset.
    cap_order : int
        Ramp polynomial order (default 3).
    spin_both : bool
        If True (default), inject both spin channels equally and scale G by 2
        so that G/G₀ = T_up + T_down ∈ [0, 2].
        If False, inject spin-up only and G/G₀ = T_up ∈ [0, 1].
    """
    wall_start = time.perf_counter()
    layout     = t.build_single_ring_layout(p)
    k          = compute_wave_number(p, fermi_energy_mev)
    time_steps = t.time_steps_for_duration(p, total_time_ps)

    if verbose:
        v = t.lead_group_velocity(p, k)
        transit = (p.L_leads + p.L_ring) / v
        print(f"  CAP run: α={p.alpha:.1f} meV·nm | cap_str={cap_strength:.2f} meV "
              f"| transit≈{transit:.1f} ps | total={total_time_ps:.1f} ps")

    cap_vec = build_cap_vector(layout, p, cap_fraction, cap_strength, cap_order)
    A, B    = build_cn_matrices_with_cap(p, layout, cap_vec)
    solver  = spla.factorized(A.tocsc())

    spin_arg = "both" if spin_both else "up"
    psi = build_initial_wavefunction(
        p, layout, k, packet_center_fraction, packet_width_nm, spin=spin_arg)
    N0  = _weighted_total_probability(psi, layout)
    _spin_scale = 2.0 if spin_both else 1.0

    # Identify CAP sites (W > 0)
    left_cap_sites  = layout.left_lead_sites[cap_vec[layout.left_lead_sites] > 0]
    right_cap_sites = layout.right_lead_sites[cap_vec[layout.right_lead_sites] > 0]
    # CAP absorption strengths at those sites (spinor: repeat twice)
    W_left  = cap_vec[left_cap_sites]
    W_right = cap_vec[right_cap_sites]

    T_absorbed = 0.0   # cumulative transmitted probability
    R_absorbed = 0.0   # cumulative reflected probability

    P_left_series  = np.empty(time_steps + 1) if keep_time_series else None
    P_right_series = np.empty(time_steps + 1) if keep_time_series else None
    if keep_time_series:
        P_left_series[0]  = 0.0
        P_right_series[0] = 0.0

    for step in range(time_steps):
        psi = solver(B @ psi)

        # Absorption rate at each CAP site: dP/dt = (2/hbar) * W * |psi|^2
        psi_block = psi.reshape(-1, 2)

        density_left  = np.sum(np.abs(psi_block[left_cap_sites])**2,  axis=1)
        density_right = np.sum(np.abs(psi_block[right_cap_sites])**2, axis=1)

        # Integrate absorption over time step (factor 2/hbar from the imaginary term)
        dR = (2.0 / t.h_bar) * np.dot(W_left,  density_left)  * p.dt
        dT = (2.0 / t.h_bar) * np.dot(W_right, density_right) * p.dt
        R_absorbed += dR
        T_absorbed += dT

        if keep_time_series:
            P_left_series[step + 1]  = R_absorbed
            P_right_series[step + 1] = T_absorbed

    # Normalise by initial norm
    T_raw = T_absorbed / N0 if N0 > 0 else 0.0
    R_raw = R_absorbed / N0 if N0 > 0 else 0.0
    # Scale by 2 when injecting both spins: G/G₀ = T_up + T_down = 2 * T_raw
    T = min(_spin_scale * T_raw, 2.0)
    R = min(_spin_scale * R_raw, 2.0)

    wall_seconds = time.perf_counter() - wall_start
    if verbose:
        print(f"    → T={T:.6f}  R={R:.6f}  T+R={T+R:.4f}  "
              f"(spin={'both' if spin_both else 'up'})  [{wall_seconds:.1f} s]")

    return CAPConductanceResult(
        params=p,
        fermi_energy_mev=fermi_energy_mev,
        T=T,
        R=R,
        T_plus_R=T + R,
        G_over_G0=T,
        G_siemens=T * G0_SIEMENS,
        total_time_ps=total_time_ps,
        wall_seconds=wall_seconds,
        time_axis_ps=(np.arange(time_steps + 1, dtype=float) * p.dt
                      if keep_time_series else None),
        P_left_cap_rate=P_left_series,
        P_right_cap_rate=P_right_series,
    )


def run_transparent_conductance(
    p: t.PhysicsParams,
    fermi_energy_mev: float = 4.19,
    total_time_ps: float = 30.0,
    packet_center_fraction: float = 0.8,
    packet_width_nm: float = 150.0,
    keep_time_series: bool = False,
    verbose: bool = True,
    spin_both: bool = True,
) -> "CAPConductanceResult":
    """Measure conductance using the exact single-site self-energy boundary.

    Same bookkeeping as ``run_cap_conductance`` (T/R accumulated from the
    absorption rate at the boundary sites) so the two are directly
    comparable, but the "absorber" here is the exact lead self-energy
    Sigma(E_F) at a single site per lead instead of a finite CAP ramp.
    See the module docstring above ``lead_self_energy`` for the derivation
    and its documented limitation (monochromatic, not a full time-convolution
    DTBC).
    """
    wall_start = time.perf_counter()
    layout     = t.build_single_ring_layout(p)
    k          = compute_wave_number(p, fermi_energy_mev)
    time_steps = t.time_steps_for_duration(p, total_time_ps)

    A, B, sigma = build_cn_matrices_with_transparent_bc(p, layout, fermi_energy_mev)
    solver      = spla.factorized(A.tocsc())

    if verbose:
        v = t.lead_group_velocity(p, k)
        transit = (p.L_leads + p.L_ring) / v
        print(f"  Transparent-BC run: alpha={p.alpha:.1f} meV*nm | "
              f"Sigma={sigma:.4f} meV | transit={transit:.1f} ps | "
              f"total={total_time_ps:.1f} ps")

    spin_arg = "both" if spin_both else "up"
    psi = build_initial_wavefunction(
        p, layout, k, packet_center_fraction, packet_width_nm, spin=spin_arg)
    N0  = _weighted_total_probability(psi, layout)
    _spin_scale = 2.0 if spin_both else 1.0

    left_site  = int(layout.left_lead_sites[0])
    right_site = int(layout.right_lead_sites[-1])
    # Absorption rate at a single site: dP/dt = (2/hbar) * (-Im(Sigma)) * |psi|^2
    gamma_half = -float(np.imag(sigma))   # = t*sin(k*delta_x), >= 0 for a propagating mode

    T_absorbed = 0.0
    R_absorbed = 0.0

    P_left_series  = np.empty(time_steps + 1) if keep_time_series else None
    P_right_series = np.empty(time_steps + 1) if keep_time_series else None
    if keep_time_series:
        P_left_series[0]  = 0.0
        P_right_series[0] = 0.0

    for step in range(time_steps):
        psi = solver(B @ psi)

        psi_block   = psi.reshape(-1, 2)
        density_left  = float(np.sum(np.abs(psi_block[left_site])**2))
        density_right = float(np.sum(np.abs(psi_block[right_site])**2))

        dR = (2.0 / t.h_bar) * gamma_half * density_left  * p.dt
        dT = (2.0 / t.h_bar) * gamma_half * density_right * p.dt
        R_absorbed += dR
        T_absorbed += dT

        if keep_time_series:
            P_left_series[step + 1]  = R_absorbed
            P_right_series[step + 1] = T_absorbed

    T_raw = T_absorbed / N0 if N0 > 0 else 0.0
    R_raw = R_absorbed / N0 if N0 > 0 else 0.0
    T = min(_spin_scale * T_raw, 2.0)
    R = min(_spin_scale * R_raw, 2.0)

    wall_seconds = time.perf_counter() - wall_start
    if verbose:
        print(f"    -> T={T:.6f}  R={R:.6f}  T+R={T+R:.4f}  "
              f"(spin={'both' if spin_both else 'up'})  [{wall_seconds:.1f} s]")

    return CAPConductanceResult(
        params=p,
        fermi_energy_mev=fermi_energy_mev,
        T=T,
        R=R,
        T_plus_R=T + R,
        G_over_G0=T,
        G_siemens=T * G0_SIEMENS,
        total_time_ps=total_time_ps,
        wall_seconds=wall_seconds,
        time_axis_ps=(np.arange(time_steps + 1, dtype=float) * p.dt
                      if keep_time_series else None),
        P_left_cap_rate=P_left_series,
        P_right_cap_rate=P_right_series,
    )


# ---------------------------------------------------------------------------
# Parameter sweep engine
# ---------------------------------------------------------------------------

def sweep_conductance(
    base_params: t.PhysicsParams,
    sweep_parameter: str,
    sweep_values: list[float] | np.ndarray,
    fermi_energy_mev: float = 4.19,
    total_time_ps: float = 13.5,
    packet_center_fraction: float = 0.8,
    packet_width_nm: float = 150.0,
    keep_time_series: bool = False,
    verbose: bool = True,
) -> SweepResult:
    """Sweep one parameter and collect conductance results.

    Parameters
    ----------
    base_params:
        Starting parameter set.  All fields not being swept are frozen here.
    sweep_parameter:
        Name of the ``PhysicsParams`` field to vary, e.g. ``"alpha"``,
        ``"V0_U"``, ``"Ux_U"``, ``"Uy_U"``, ``"V0_L"``, etc.
    sweep_values:
        Values to assign to ``sweep_parameter`` in sequence.

    Returns
    -------
    SweepResult with one ConductanceResult per value.
    """
    sweep_values = list(sweep_values)
    result = SweepResult(sweep_parameter=sweep_parameter, sweep_values=sweep_values)

    print(f"\n=== Sweep: {sweep_parameter} over {len(sweep_values)} values ===")
    for i, val in enumerate(sweep_values):
        print(f"[{i+1}/{len(sweep_values)}] {sweep_parameter}={val}")
        p_i = base_params.with_changes(**{sweep_parameter: val})
        cr  = run_single_conductance(
            p_i,
            fermi_energy_mev=fermi_energy_mev,
            total_time_ps=total_time_ps,
            packet_center_fraction=packet_center_fraction,
            packet_width_nm=packet_width_nm,
            keep_time_series=keep_time_series,
            verbose=verbose,
        )
        result.results.append(cr)

    return result


def sweep_two_parameters(
    base_params: t.PhysicsParams,
    param1: str,
    values1: list[float] | np.ndarray,
    param2: str,
    values2: list[float] | np.ndarray,
    fermi_energy_mev: float = 4.19,
    total_time_ps: float = 13.5,
    verbose: bool = True,
) -> dict[tuple[float, float], ConductanceResult]:
    """Sweep two parameters jointly (full grid).

    Returns a dict keyed by (val1, val2) → ConductanceResult.
    Useful for producing 2-D conductance maps (e.g. alpha vs V0_U).
    """
    values1 = list(values1)
    values2 = list(values2)
    total   = len(values1) * len(values2)
    results = {}
    count   = 0

    print(f"\n=== 2-D Sweep: {param1} × {param2}  ({total} points) ===")
    for v1 in values1:
        for v2 in values2:
            count += 1
            print(f"[{count}/{total}] {param1}={v1}  {param2}={v2}")
            p_i = base_params.with_changes(**{param1: v1, param2: v2})
            cr  = run_single_conductance(
                p_i,
                fermi_energy_mev=fermi_energy_mev,
                total_time_ps=total_time_ps,
                verbose=verbose,
            )
            results[(v1, v2)] = cr

    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_sweep_conductance(
    sweep: SweepResult,
    analytical_values: np.ndarray | None = None,
    analytical_label: str = "Analytical",
    title: str | None = None,
    save_path: str | None = None,
    y_label: str = r"$G/G_0$",
) -> None:
    """Plot G/G₀ vs the swept parameter, optionally overlaying analytical data.

    Parameters
    ----------
    sweep:
        Result from ``sweep_conductance``.
    analytical_values:
        Array of G/G₀ values at each ``sweep.sweep_values`` point from your
        analytic calculation.  Must have the same length as ``sweep.sweep_values``.
    """
    x   = np.array(sweep.sweep_values)
    y   = sweep.G_over_G0_values

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y, "o-", color="tab:blue", linewidth=1.8,
            markersize=5, label="Numérico (wavepacket)")

    if analytical_values is not None:
        ax.plot(x, np.asarray(analytical_values), "s--", color="tab:orange",
                linewidth=1.6, markersize=5, label=analytical_label)

    ax.set_xlabel(sweep.sweep_parameter)
    ax.set_ylabel(y_label)
    ax.set_title(title or f"Conductancia de Landauer vs {sweep.sweep_parameter}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=180)
        print(f"Figura guardada en: {save_path}")
    plt.show()


def plot_P_R_time_series(result: ConductanceResult, save_path: str | None = None) -> None:
    """Plot P_R(t) and P_total(t) for a single run (requires keep_time_series=True)."""
    if result.time_axis_ps is None:
        raise ValueError("Re-run with keep_time_series=True to get the time series.")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(result.time_axis_ps, result.P_R_history, color="tab:blue")
    axes[0].axvline(result.echo_cutoff_ps, color="red", linestyle="--",
                    linewidth=0.9, label="Echo cutoff")
    axes[0].axvline(result.P_R_max_time_ps, color="green", linestyle=":",
                    linewidth=1.1, label=f"Peak T={result.T:.4f}")
    axes[0].set_ylabel(r"$P_R(t)$ [nm]")
    axes[0].set_title("Probabilidad integrada en el lead derecho")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(result.time_axis_ps,
                 result.P_total_history / result.P_total_history[0],
                 color="tab:red")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[1].set_xlabel("Tiempo (ps)")
    axes[1].set_ylabel(r"$P_{total}(t) / P_0$")
    axes[1].set_title("Conservación de la norma")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(f"α={result.params.alpha:.1f} meV·nm  |  V0_U={result.params.V0_U:.3f}",
                 fontsize=11)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=180)
    plt.show()


def save_sweep_results_npz(sweep: SweepResult, path: str) -> None:
    """Save a 1-D sweep to a compressed .npz file for later analysis."""
    table = sweep.to_table()
    arrays = {k: np.array([row[k] for row in table]) for k in table[0]}
    np.savez_compressed(path, **arrays)
    print(f"Sweep guardado en: {path}")


# ---------------------------------------------------------------------------
# Demo / script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    base = t.default_params()

    # ---- Example 1: sweep alpha ----
    alpha_values = np.linspace(0, 60, 13)   # 0 to 60 meV·nm in 13 steps
    sweep_alpha = sweep_conductance(
        base_params=base,
        sweep_parameter="alpha",
        sweep_values=alpha_values,
        fermi_energy_mev=4.19,
        total_time_ps=13.5,
        verbose=True,
    )
    plot_sweep_conductance(
        sweep_alpha,
        title=r"$G/G_0$ vs Rashba strength $\alpha$",
    )
    save_sweep_results_npz(sweep_alpha, "conductance_vs_alpha.npz")

    # ---- Example 2: sweep V0_U (upper arm barrier offset) ----
    V0_values = np.linspace(0.0, 0.5, 11)
    sweep_V0U = sweep_conductance(
        base_params=base,
        sweep_parameter="V0_U",
        sweep_values=V0_values,
        fermi_energy_mev=4.19,
        total_time_ps=13.5,
        verbose=True,
    )
    plot_sweep_conductance(
        sweep_V0U,
        title=r"$G/G_0$ vs upper arm barrier $V_{0,U}$",
    )
    save_sweep_results_npz(sweep_V0U, "conductance_vs_V0U.npz")
