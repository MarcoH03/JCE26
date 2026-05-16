"""Core physics helpers and a stable graph Hamiltonian for the JCE26 simulation.

Refactored to accept an explicit ``PhysicsParams`` object everywhere, instead of
reading module-level globals. This makes parameter sweeps possible: callers
construct a ``PhysicsParams`` with the desired ``alpha``, ``V0_U``, etc., and
pass it through the call chain without touching module state.

The module still exposes a ``default_params()`` factory that reproduces the
original hard-coded values, so existing call sites require only minimal changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Universal constants  (never swept – unit system is fixed)
# ---------------------------------------------------------------------------
h_bar = 0.658212          # meV·ps
m_e   = 5.68563e-3        # meV·ps²/nm²


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhysicsParams:
    """All tunable physical and numerical parameters for one simulation run.

    Fields are grouped as:
      - Material / geometry  (rarely changed between sweeps)
      - Discretization       (rarely changed)
      - Magnetic / spin-orbit (alpha is the primary sweep variable)
      - Potential parameters  (V0_*, Ux_*, Uy_*, … are secondary sweep variables)
      - Potential model       (string selector)

    All lengths in nm, energies in meV, times in ps.
    """

    # ---- material / geometry ------------------------------------------------
    m_factor: float = 0.023          # effective mass as a fraction of m_e
    R: float        = 250.0          # ring radius, nm
    L_leads: float  = 2000.0         # length of each lead, nm

    # ---- discretization -----------------------------------------------------
    N_l: int   = 381    # lead nodes (including the junction node)
    N_R: int   = 151    # arm nodes  (including both junction nodes)
    dt: float  = 0.002  # time step, ps

    # ---- magnetic -----------------------------------------------------------
    Phi: float = 0.5    # Aharonov-Bohm flux in units of Φ₀

    # ---- spin-orbit ---------------------------------------------------------
    alpha: float = 20.0   # Rashba strength, meV·nm

    # ---- potential model selector -------------------------------------------
    # Options:
    # - "none": flat potential, useful as a transport baseline.
    # - "gaussian_qpc": smooth calibrated barrier used for the asymptotic benchmark.
    # - "legacy_localized_qpc": rehabilitated version of the original QPC idea,
    #   localized and capped so it does not create unphysical wells.
    # - "legacy_unbounded_qpc": original formula kept only for comparison/debugging.
    potential_model: str = "legacy_unbounded_qpc"

    # ---- legacy / localized QPC parameters ----------------------------------
    V0_L: float = 0.0
    Ux_L: float = 6
    Uy_L: float = 0.6

    V0_U: float = 0.0
    Ux_U: float = 0.2
    Uy_U: float = 0.2

    V0_R: float = 0.0
    Ux_R: float = 6
    Uy_R: float = 6

    s0_L_fraction: float = 0.05   # QPC center as a fraction of L_leads
    s0_U_fraction: float = 0.50   # QPC center as a fraction of L_ring
    s0_R_fraction: float = 0.05   # QPC center as a fraction of L_leads

    legacy_localization_width_scale: float = 6.0

    # ---- Gaussian QPC parameters --------------------------------------------
    gaussian_qpc_heights_mev: dict[str, float] = field(default_factory=lambda: {
        "L": 0.0,
        "U": 0.0,
        "D": 0.0,
        "R": 0.0,
    })
    gaussian_qpc_widths_nm: dict[str, float] = field(default_factory=lambda: {
        "L": 120.0,
        "U":  90.0,
        "D": 120.0,
        "R": 120.0,
    })

    # ---- reference energy (for barrier calibration only) --------------------
    potential_reference_energy_mev: float = 4.19

    # -------------------------------------------------------------------------
    # Derived quantities (computed lazily from the fields above)
    # -------------------------------------------------------------------------

    @property
    def m(self) -> float:
        return self.m_factor * m_e

    @property
    def L_ring(self) -> float:
        return np.pi * self.R

    @property
    def delta_x(self) -> float:
        return self.L_leads / (self.N_l - 1)

    @property
    def delta_s(self) -> float:
        return self.L_ring / (self.N_R - 1)

    @property
    def phi_link(self) -> float:
        return self.Phi / (2 * (self.N_R - 1))

    @property
    def phi_U(self) -> float:
        return self.phi_link

    @property
    def phi_D(self) -> float:
        return -self.phi_link

    @property
    def phi_so_link(self) -> float:
        return self.m * self.alpha * self.delta_s / h_bar**2

    @property
    def t_lead(self) -> float:
        return h_bar**2 / (2 * self.m * self.delta_x**2)

    @property
    def t_ring(self) -> float:
        return h_bar**2 / (2 * self.m * self.delta_s**2)

    @property
    def qpc_max_height_mev(self) -> float:
        return 0.95 * self.potential_reference_energy_mev

    @property
    def s0_L(self) -> float:
        return self.s0_L_fraction * self.L_leads

    @property
    def s0_U(self) -> float:
        return self.s0_U_fraction * self.L_ring

    @property
    def s0_R(self) -> float:
        return self.s0_R_fraction * self.L_leads

    def with_changes(self, **kwargs) -> "PhysicsParams":
        """Return a new PhysicsParams with selected fields overridden."""
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serializable snapshot of the full parameter set."""
        d = {
            "m_factor": self.m_factor,
            "R_nm": self.R,
            "L_leads_nm": self.L_leads,
            "N_l": self.N_l,
            "N_R": self.N_R,
            "dt_ps": self.dt,
            "Phi": self.Phi,
            "alpha_mev_nm": self.alpha,
            "potential_model": self.potential_model,
            "V0_L": self.V0_L, "Ux_L": self.Ux_L, "Uy_L": self.Uy_L,
            "V0_U": self.V0_U, "Ux_U": self.Ux_U, "Uy_U": self.Uy_U,
            "V0_R": self.V0_R, "Ux_R": self.Ux_R, "Uy_R": self.Uy_R,
            "s0_L_fraction": self.s0_L_fraction,
            "s0_U_fraction": self.s0_U_fraction,
            "s0_R_fraction": self.s0_R_fraction,
            "legacy_localization_width_scale": self.legacy_localization_width_scale,
            "gaussian_qpc_heights_mev": dict(self.gaussian_qpc_heights_mev),
            "gaussian_qpc_widths_nm": dict(self.gaussian_qpc_widths_nm),
            "potential_reference_energy_mev": self.potential_reference_energy_mev,
            # derived
            "m_effective_mev_ps2_nm2": self.m,
            "L_ring_nm": self.L_ring,
            "delta_x_nm": self.delta_x,
            "delta_s_nm": self.delta_s,
            "phi_link": self.phi_link,
            "phi_so_link": self.phi_so_link,
            "t_lead_mev": self.t_lead,
            "t_ring_mev": self.t_ring,
        }
        return d


def default_params() -> PhysicsParams:
    """Return a PhysicsParams that reproduces the original hard-coded values."""
    return PhysicsParams(
        gaussian_qpc_heights_mev={
            "L": 0.0,
            "U": 0.60 * 4.19,
            "D": 0.0,
            "R": 0.0,
        }
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SingleRingLayout:
    """Unique-node view of the single-ring graph used by the solver."""

    unique_site_count: int
    spinor_size: int
    left_lead_sites: np.ndarray
    upper_internal_sites: np.ndarray
    lower_internal_sites: np.ndarray
    right_lead_sites: np.ndarray
    upper_arm_sites: np.ndarray
    lower_arm_sites: np.ndarray
    plot_sites: np.ndarray
    plot_section_labels: tuple[str, ...]
    integration_weights_nm: np.ndarray
    left_junction_site: int
    right_junction_site: int


def _spin_slice(site: int) -> slice:
    return slice(2 * site, 2 * site + 2)


def _is_close_to_zero(value: complex, tolerance: float = 1e-15) -> bool:
    return abs(value) < tolerance


def U_rashba(theta: float) -> np.ndarray:
    """Return the 2×2 spin rotation for a single ring hop."""
    if theta == 0.0:
        return np.eye(2, dtype=complex)
    return np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta),  np.cos(theta)]],
        dtype=complex,
    )


def build_single_ring_layout(p: PhysicsParams | None = None) -> SingleRingLayout:
    """Build the unique-node layout for the given parameter set.

    No longer cached with lru_cache because the layout depends on p.N_l and
    p.N_R, which change between parameter sweeps.  Callers that need repeated
    access within one sweep should hold the returned object themselves.
    """
    if p is None:
        p = default_params()

    N_l = p.N_l
    N_R = p.N_R

    left_lead_sites    = np.arange(N_l, dtype=int)
    left_junction_site = int(left_lead_sites[-1])

    upper_internal_start = N_l
    upper_internal_sites = np.arange(upper_internal_start,
                                     upper_internal_start + (N_R - 2), dtype=int)

    lower_internal_start = upper_internal_start + upper_internal_sites.size
    lower_internal_sites = np.arange(lower_internal_start,
                                     lower_internal_start + (N_R - 2), dtype=int)

    right_lead_start    = lower_internal_start + lower_internal_sites.size
    right_lead_sites    = np.arange(right_lead_start,
                                    right_lead_start + N_l, dtype=int)
    right_junction_site = int(right_lead_sites[0])

    upper_arm_sites = np.concatenate((
        np.array([left_junction_site], dtype=int),
        upper_internal_sites,
        np.array([right_junction_site], dtype=int),
    ))
    lower_arm_sites = np.concatenate((
        np.array([left_junction_site], dtype=int),
        lower_internal_sites,
        np.array([right_junction_site], dtype=int),
    ))

    plot_sites = np.concatenate((
        left_lead_sites, upper_arm_sites, lower_arm_sites, right_lead_sites,
    ))
    plot_section_labels = (
        ("L",) * left_lead_sites.size
        + ("U",) * upper_arm_sites.size
        + ("D",) * lower_arm_sites.size
        + ("R",) * right_lead_sites.size
    )

    unique_site_count = int(right_lead_sites[-1] + 1)
    integration_weights_nm = np.zeros(unique_site_count, dtype=float)

    def add_edge_weights(path: np.ndarray, spacing_nm: float) -> None:
        for site_a, site_b in zip(path[:-1], path[1:]):
            integration_weights_nm[site_a] += spacing_nm / 2.0
            integration_weights_nm[site_b] += spacing_nm / 2.0

    add_edge_weights(left_lead_sites, p.delta_x)
    add_edge_weights(upper_arm_sites, p.delta_s)
    add_edge_weights(lower_arm_sites, p.delta_s)
    add_edge_weights(right_lead_sites, p.delta_x)

    return SingleRingLayout(
        unique_site_count=unique_site_count,
        spinor_size=2 * unique_site_count,
        left_lead_sites=left_lead_sites,
        upper_internal_sites=upper_internal_sites,
        lower_internal_sites=lower_internal_sites,
        right_lead_sites=right_lead_sites,
        upper_arm_sites=upper_arm_sites,
        lower_arm_sites=lower_arm_sites,
        plot_sites=plot_sites,
        plot_section_labels=plot_section_labels,
        integration_weights_nm=integration_weights_nm,
        left_junction_site=left_junction_site,
        right_junction_site=right_junction_site,
    )


# ---------------------------------------------------------------------------
# Potential helpers
# ---------------------------------------------------------------------------

def section_position_nm(p: PhysicsParams, section: str, i: int) -> float:
    if section in {"L", "R"}:
        return i * p.delta_x
    if section in {"U", "D"}:
        return i * p.delta_s
    raise ValueError(f"Unknown section {section!r}")


def section_qpc_center_nm(p: PhysicsParams, section: str) -> float:
    if section == "L":
        return p.s0_L
    if section == "U":
        return p.s0_U
    if section == "R":
        return p.s0_R
    if section == "D":
        return 0.5 * p.L_ring
    raise ValueError(f"Unknown section {section!r}")


def section_legacy_barrier_height_mev(p: PhysicsParams, section: str) -> float:
    if section == "L":
        return max(0.0, min(p.qpc_max_height_mev,
                            p.V0_L + 0.5 * h_bar * np.sqrt(2 * p.Uy_L / p.m)))
    if section == "U":
        return max(0.0, min(p.qpc_max_height_mev,
                            p.V0_U + 0.5 * h_bar * np.sqrt(2 * p.Uy_U / p.m)))
    if section == "R":
        return max(0.0, min(p.qpc_max_height_mev,
                            p.V0_R + 0.5 * h_bar * np.sqrt(2 * p.Uy_R / p.m)))
    if section == "D":
        return 0.0
    raise ValueError(f"Unknown section {section!r}")


def section_longitudinal_curvature(p: PhysicsParams, section: str) -> float:
    if section == "L":
        return p.Ux_L
    if section == "U":
        return p.Ux_U
    if section == "R":
        return p.Ux_R
    if section == "D":
        return 0.0
    raise ValueError(f"Unknown section {section!r}")


def gaussian_qpc_potential(p: PhysicsParams, section: str, s_nm: float) -> float:
    height_mev = p.gaussian_qpc_heights_mev.get(section, 0.0)
    if height_mev <= 0.0:
        return 0.0
    width_nm   = p.gaussian_qpc_widths_nm[section]
    center_nm  = section_qpc_center_nm(p, section)
    normalized = (s_nm - center_nm) / width_nm
    return float(height_mev * np.exp(-0.5 * normalized**2))


def legacy_unbounded_qpc_potential(p: PhysicsParams, section: str, s_nm: float) -> float:
    if section == "L":
        return float(p.V0_L - p.Ux_L * (s_nm - p.s0_L)**2
                     + 0.5 * h_bar * np.sqrt(2 * p.Uy_L / p.m))
    if section == "U":
        return float(p.V0_U - p.Ux_U * (s_nm - p.s0_U)**2
                     + 0.5 * h_bar * np.sqrt(2 * p.Uy_U / p.m))
    if section == "D":
        return 0.0
    if section == "R":
        return float(p.V0_R - p.Ux_R * (s_nm - p.s0_R)**2
                     + 0.5 * h_bar * np.sqrt(2 * p.Uy_R / p.m))
    raise ValueError(f"Unknown section {section!r}")


def legacy_localized_qpc_potential(p: PhysicsParams, section: str, s_nm: float) -> float:
    barrier_height_mev = section_legacy_barrier_height_mev(p, section)
    if barrier_height_mev <= 0.0:
        return 0.0
    curvature  = max(section_longitudinal_curvature(p, section), 1e-12)
    half_width = p.legacy_localization_width_scale * np.sqrt(barrier_height_mev / curvature)
    center_nm  = section_qpc_center_nm(p, section)
    shape      = max(0.0, 1.0 - ((s_nm - center_nm) / half_width)**2)
    return float(barrier_height_mev * shape**2)


def V(p: PhysicsParams, section: str, i: int) -> float:
    """Return the active potential at node ``i`` of ``section``."""
    s_nm = section_position_nm(p, section, i)
    if p.potential_model == "none":
        return 0.0
    if p.potential_model == "gaussian_qpc":
        return gaussian_qpc_potential(p, section, s_nm)
    if p.potential_model == "legacy_localized_qpc":
        return legacy_localized_qpc_potential(p, section, s_nm)
    if p.potential_model == "legacy_unbounded_qpc":
        return legacy_unbounded_qpc_potential(p, section, s_nm)
    raise ValueError(f"Unknown potential model {p.potential_model!r}")


def build_site_potential(p: PhysicsParams, layout: SingleRingLayout) -> np.ndarray:
    """Return the physical on-site potential for every unique graph node."""
    potential = np.zeros(layout.unique_site_count, dtype=float)

    for local_index, site in enumerate(layout.left_lead_sites[:-1]):
        potential[site] = V(p, "L", p.N_l - 1 - local_index)

    for local_index, site in enumerate(layout.right_lead_sites[1:], start=1):
        potential[site] = V(p, "R", local_index)

    for local_index, site in enumerate(layout.upper_internal_sites, start=1):
        potential[site] = V(p, "U", local_index)

    for local_index, site in enumerate(layout.lower_internal_sites, start=1):
        potential[site] = V(p, "D", local_index)

    potential[layout.left_junction_site]  = np.mean((V(p, "L", 0), V(p, "U", 0), V(p, "D", 0)))
    potential[layout.right_junction_site] = np.mean((V(p, "R", 0), V(p, "U", p.N_R - 1), V(p, "D", p.N_R - 1)))

    return potential


def summarize_potential_by_section(p: PhysicsParams, layout: SingleRingLayout) -> dict[str, dict[str, float]]:
    """Return min/max/center potential values for each section."""
    section_sizes = {
        "L": layout.left_lead_sites.size,
        "U": layout.upper_arm_sites.size,
        "D": layout.lower_arm_sites.size,
        "R": layout.right_lead_sites.size,
    }
    summary = {}
    for section, size in section_sizes.items():
        values = np.array([V(p, section, i) for i in range(size)], dtype=float)
        summary[section] = {
            "min_mev":    float(values.min()),
            "max_mev":    float(values.max()),
            "center_mev": float(values[size // 2]),
        }
    return summary


# ---------------------------------------------------------------------------
# Hamiltonian and Crank-Nicolson matrices
# ---------------------------------------------------------------------------

def build_single_ring_hamiltonian(p: PhysicsParams, layout: SingleRingLayout) -> sp.csr_matrix:
    """Build the Hermitian graph Hamiltonian for the given parameter set."""
    hamiltonian      = sp.lil_matrix((layout.spinor_size, layout.spinor_size), dtype=complex)
    spin_identity    = np.eye(2, dtype=complex)
    spin_rotation    = U_rashba(p.phi_so_link)
    site_potential   = build_site_potential(p, layout)

    def add_spin_block(site_row: int, site_col: int, block: np.ndarray) -> None:
        row_slice = _spin_slice(site_row)
        col_slice = _spin_slice(site_col)
        for lr in range(2):
            for lc in range(2):
                val = block[lr, lc]
                if _is_close_to_zero(val):
                    continue
                hamiltonian[row_slice.start + lr, col_slice.start + lc] += val

    def add_site_energy(site: int, energy_mev: float) -> None:
        if _is_close_to_zero(energy_mev):
            return
        add_spin_block(site, site, energy_mev * spin_identity)

    def add_link(site_left: int, site_right: int, hopping_mev: float,
                 phase: float, spin_matrix: np.ndarray) -> None:
        add_site_energy(site_left,  hopping_mev)
        add_site_energy(site_right, hopping_mev)
        forward = -hopping_mev * np.exp(1j * phase) * spin_matrix
        add_spin_block(site_left,  site_right, forward)
        add_spin_block(site_right, site_left,  forward.conj().T)

    for site, pot in enumerate(site_potential):
        add_site_energy(site, pot)

    for sl, sr in zip(layout.left_lead_sites[:-1], layout.left_lead_sites[1:]):
        add_link(sl, sr, p.t_lead, 0.0, spin_identity)

    for sl, sr in zip(layout.right_lead_sites[:-1], layout.right_lead_sites[1:]):
        add_link(sl, sr, p.t_lead, 0.0, spin_identity)

    for sl, sr in zip(layout.upper_arm_sites[:-1], layout.upper_arm_sites[1:]):
        add_link(sl, sr, p.t_ring, p.phi_U, spin_rotation)

    for sl, sr in zip(layout.lower_arm_sites[:-1], layout.lower_arm_sites[1:]):
        add_link(sl, sr, p.t_ring, p.phi_D, spin_rotation)

    return hamiltonian.tocsr()


def build_cn_matrices(p: PhysicsParams, layout: SingleRingLayout
                      ) -> tuple[sp.csr_matrix, sp.csr_matrix, int]:
    """Return (A, B, spinor_size) for the Crank-Nicolson step."""
    hamiltonian = build_single_ring_hamiltonian(p, layout)
    identity    = sp.identity(layout.spinor_size, format="csr", dtype=complex)
    prefactor   = 1j * p.dt / (2 * h_bar)
    A = (identity + prefactor * hamiltonian).tocsr()
    B = (identity - prefactor * hamiltonian).tocsr()
    return A, B, layout.spinor_size


# Keep the old name for backward compatibility with main.py
def matrix_A_B_generator_single_ring(N_R_requested: int, k: float,
                                     p: PhysicsParams | None = None):
    """Backward-compatible wrapper around build_cn_matrices."""
    if p is None:
        p = default_params()
    if N_R_requested != p.N_R:
        raise ValueError(
            f"Requested N_R={N_R_requested} but params has N_R={p.N_R}."
        )
    layout = build_single_ring_layout(p)
    return build_cn_matrices(p, layout)


# ---------------------------------------------------------------------------
# Misc helpers (unchanged in spirit, now take p explicitly)
# ---------------------------------------------------------------------------

def lead_group_velocity(p: PhysicsParams, k: float) -> float:
    return h_bar * k / p.m


def time_steps_for_duration(p: PhysicsParams, total_time_ps: float) -> int:
    return int(round(total_time_ps / p.dt))


def linearize_site_values_for_plot(site_values: np.ndarray,
                                   layout: SingleRingLayout) -> np.ndarray:
    return np.asarray(site_values[layout.plot_sites], dtype=float)


def build_potential_profile(p: PhysicsParams, layout: SingleRingLayout):
    node_indices = np.arange(layout.plot_sites.size)
    potentials   = linearize_site_values_for_plot(build_site_potential(p, layout), layout)
    return node_indices, potentials, list(layout.plot_section_labels)
