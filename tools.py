"""Core physics helpers and a stable graph Hamiltonian for the JCE26 simulation.

The earlier version assembled Crank-Nicolson row by row with explicit junction
constraints and ad-hoc boundary equations. That matrix pair was not generated
from a Hermitian Hamiltonian, so the propagation operator acquired eigenvalues
larger than one and amplified the wavefunction even for `V = 0`.

This module replaces that assembly with a physically cleaner model:

1. The left and right junctions are shared graph nodes.
2. Leads and ring branches are connected through Hermitian hopping blocks.
3. Crank-Nicolson is built from `H` as
   `A = I + i dt H / (2 h_bar)` and `B = I - i dt H / (2 h_bar)`.

With a Hermitian `H`, this propagation is norm preserving up to numerical round
off. The remaining diagnostics then distinguish real wavepacket dynamics from a
mere numerical blow-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.sparse as sp


# Universal constants in the unit system used by the project.
h_bar = 0.658212  # meV*ps
m_e = 5.68563e-3  # meV*ps^2/nm^2

# Material and geometry.
m = 0.023 * m_e  # InAs effective mass in meV*ps^2/nm^2
R = 250  # ring radius in nm
# The leads are intentionally long so the transmitted packet can separate from
# the scattering region before a hard-wall reflection from the external ends
# comes back to the ring.
L_leads = 2000  # length of each lead in nm
L_ring = np.pi * R  # arc length of a single ring arm in nm

# Spatial discretization.
# Lead and ring spacings are matched to reduce artificial reflection caused
# purely by using different finite-difference resolutions at the junctions.
N_l = 381  # lead nodes, including the junction node
N_R = 151  # arm nodes, including the two junction nodes
delta_x = L_leads / (N_l - 1)
delta_s = L_ring / (N_R - 1)

# Temporal discretization.
# `dt = 1 ps` made the free-particle phase jump too large in a single step and
# hid whether any observed transport was physical or just temporal aliasing.
dt = 0.002  # ps

# Magnetic and spin-orbit phases.
Phi = 1 / 2  # Aharonov-Bohm flux in units of Phi_0
phi_link = Phi / (2 * (N_R - 1))
phi_U = phi_link
phi_D = -phi_link
phi_L = phi_R = 0.0

alpha = 20  # Rashba strength in meV*nm
phi_so_link = theta_R = m * alpha * delta_s / h_bar**2

# Kinetic hopping energies for the finite-difference graph Hamiltonian.
t_lead = h_bar**2 / (2 * m * delta_x**2)
t_ring = h_bar**2 / (2 * m * delta_s**2)

# The old code and notes use these `lambda_*` values repeatedly, so they remain
# exposed for logging and compatibility.
lambda_lead = 1j * dt * t_lead / h_bar
lambda_ring = 1j * dt * t_ring / h_bar

# Potential-model selection.
#
# Options:
# - "none": flat potential, useful as a transport baseline.
# - "gaussian_qpc": smooth calibrated barrier used for the asymptotic benchmark.
# - "legacy_localized_qpc": rehabilitated version of the original QPC idea,
#   localized and capped so it does not create unphysical wells.
# - "legacy_unbounded_qpc": original formula kept only for comparison/debugging.
POTENTIAL_MODEL = "legacy_localized_qpc"

# This should stay aligned with `main.py` unless a different injection energy is
# intentionally explored. It is used only to calibrate barrier heights.
POTENTIAL_REFERENCE_ENERGY_MEV = 4.19
QPC_MAX_HEIGHT_MEV = 0.95 * POTENTIAL_REFERENCE_ENERGY_MEV

V0_L = 0.0
Ux_L = 0.01
Uy_L = 0.01

V0_U = 0.0
Ux_U = 0.01
Uy_U = 0.01

V0_R = 0.0
Ux_R = 0.01
Uy_R = 0.01

s0_L = 0.05 * L_leads
s0_U = 0.5 * L_ring
s0_R = 0.05 * L_leads

GAUSSIAN_QPC_HEIGHTS_MEV = {
    "L": 0.0,
    "U": 0.60 * POTENTIAL_REFERENCE_ENERGY_MEV,
    "D": 0.0,
    "R": 0.0,
}
GAUSSIAN_QPC_WIDTHS_NM = {
    "L": 120.0,
    "U": 90.0,
    "D": 120.0,
    "R": 120.0,
}

# The original QPC expression behaved like a local saddle but was evaluated
# over the full device, which forced it to -infinity far from the constriction.
# This scale factor keeps the legacy parabolic intuition while turning it into a
# finite-width barrier that relaxes back to the reservoir potential.
LEGACY_LOCALIZATION_WIDTH_SCALE = 6.0


@dataclass(frozen=True)
class SingleRingLayout:
    """Unique-node view of the single-ring graph used by the solver.

    The graph uses one physical node per actual location. Junctions are shared
    between the lead and both ring branches, so probability can be integrated
    without double counting them.
    """

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
    """Return the 2x2 spin block slice for a physical site index."""

    return slice(2 * site, 2 * site + 2)


def _is_close_to_zero(value: complex, tolerance: float = 1e-15) -> bool:
    """Return whether a scalar can be safely treated as exactly zero."""

    return abs(value) < tolerance


def U_rashba(theta: float) -> np.ndarray:
    """Return the spin rotation applied on a single ring hop.

    The previous implementation mixed this matrix with extra scalar spin phases.
    Here the spin-orbit action lives only in this 2x2 link matrix, and the
    reverse hop uses the Hermitian adjoint automatically.
    """

    if theta == 0.0:
        return np.eye(2, dtype=complex)

    return np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=complex,
    )


@lru_cache(maxsize=1)
def build_single_ring_layout() -> SingleRingLayout:
    """Build the unique-node layout used by the stable graph Hamiltonian."""

    left_lead_sites = np.arange(N_l, dtype=int)
    left_junction_site = int(left_lead_sites[-1])

    upper_internal_start = N_l
    upper_internal_sites = np.arange(upper_internal_start, upper_internal_start + (N_R - 2), dtype=int)

    lower_internal_start = upper_internal_start + upper_internal_sites.size
    lower_internal_sites = np.arange(lower_internal_start, lower_internal_start + (N_R - 2), dtype=int)

    right_lead_start = lower_internal_start + lower_internal_sites.size
    right_lead_sites = np.arange(right_lead_start, right_lead_start + N_l, dtype=int)
    right_junction_site = int(right_lead_sites[0])

    upper_arm_sites = np.concatenate(
        (
            np.array([left_junction_site], dtype=int),
            upper_internal_sites,
            np.array([right_junction_site], dtype=int),
        )
    )
    lower_arm_sites = np.concatenate(
        (
            np.array([left_junction_site], dtype=int),
            lower_internal_sites,
            np.array([right_junction_site], dtype=int),
        )
    )

    # Plot order keeps the original L -> U -> D -> R convention and duplicates
    # the shared junction densities so both branches can be visualized as lines.
    plot_sites = np.concatenate((left_lead_sites, upper_arm_sites, lower_arm_sites, right_lead_sites))
    plot_section_labels = (
        ("L",) * left_lead_sites.size
        + ("U",) * upper_arm_sites.size
        + ("D",) * lower_arm_sites.size
        + ("R",) * right_lead_sites.size
    )

    unique_site_count = int(right_lead_sites[-1] + 1)
    integration_weights_nm = np.zeros(unique_site_count, dtype=float)

    def add_edge_weights(path: np.ndarray, spacing_nm: float) -> None:
        """Assign half of each segment length to each endpoint node."""

        for site_a, site_b in zip(path[:-1], path[1:]):
            integration_weights_nm[site_a] += spacing_nm / 2.0
            integration_weights_nm[site_b] += spacing_nm / 2.0

    add_edge_weights(left_lead_sites, delta_x)
    add_edge_weights(upper_arm_sites, delta_s)
    add_edge_weights(lower_arm_sites, delta_s)
    add_edge_weights(right_lead_sites, delta_x)

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


def get_section_layout():
    """Return the plotting/storage order used across the project: L -> U -> D -> R."""

    layout = build_single_ring_layout()
    return (
        ("L", layout.left_lead_sites.size),
        ("U", layout.upper_arm_sites.size),
        ("D", layout.lower_arm_sites.size),
        ("R", layout.right_lead_sites.size),
    )


def lead_group_velocity(k: float) -> float:
    """Return the group velocity in the free leads for the chosen wave-number."""

    return h_bar * k / m


def time_steps_for_duration(total_time_ps: float) -> int:
    """Convert a target physical duration into an integer number of steps."""

    return int(round(total_time_ps / dt))


def get_simulation_parameters():
    """Expose the active physical and numerical parameters as a serializable dict."""

    return {
        "h_bar_mev_ps": h_bar,
        "m_e_mev_ps2_nm2": m_e,
        "m_effective_mev_ps2_nm2": m,
        "R_nm": R,
        "L_leads_nm": L_leads,
        "L_ring_nm": L_ring,
        "N_l": N_l,
        "N_R": N_R,
        "delta_x_nm": delta_x,
        "delta_s_nm": delta_s,
        "dt_ps": dt,
        "Phi": Phi,
        "phi_link": phi_link,
        "phi_U": phi_U,
        "phi_D": phi_D,
        "alpha_mev_nm": alpha,
        "phi_so_link": phi_so_link,
        "t_lead_mev": t_lead,
        "t_ring_mev": t_ring,
        "lambda_lead_imag": lambda_lead.imag,
        "lambda_ring_imag": lambda_ring.imag,
        "enable_qpc": POTENTIAL_MODEL != "none",
        "potential_model": POTENTIAL_MODEL,
        "potential_reference_energy_mev": POTENTIAL_REFERENCE_ENERGY_MEV,
        "qpc_max_height_mev": QPC_MAX_HEIGHT_MEV,
        "potential_parameters": {
            "V0_L": V0_L,
            "Ux_L": Ux_L,
            "Uy_L": Uy_L,
            "V0_U": V0_U,
            "Ux_U": Ux_U,
            "Uy_U": Uy_U,
            "V0_R": V0_R,
            "Ux_R": Ux_R,
            "Uy_R": Uy_R,
            "s0_L_nm": s0_L,
            "s0_U_nm": s0_U,
            "s0_R_nm": s0_R,
            "gaussian_qpc_heights_mev": dict(GAUSSIAN_QPC_HEIGHTS_MEV),
            "gaussian_qpc_widths_nm": dict(GAUSSIAN_QPC_WIDTHS_NM),
            "legacy_localization_width_scale": LEGACY_LOCALIZATION_WIDTH_SCALE,
        },
    }


def section_length_nm(section: str) -> float:
    """Return the physical length of the requested section."""

    if section in {"L", "R"}:
        return L_leads
    if section in {"U", "D"}:
        return L_ring
    raise ValueError(f"Unknown section {section!r}")


def section_step_nm(section: str) -> float:
    """Return the mesh spacing of the requested section."""

    if section in {"L", "R"}:
        return delta_x
    if section in {"U", "D"}:
        return delta_s
    raise ValueError(f"Unknown section {section!r}")


def section_position_nm(section: str, i: int) -> float:
    """Map the local node index of a section to its physical coordinate."""

    return i * section_step_nm(section)


def section_qpc_center_nm(section: str) -> float:
    """Return the center position used by the active QPC models."""

    if section == "L":
        return s0_L
    if section == "U":
        return s0_U
    if section == "R":
        return s0_R
    if section == "D":
        return 0.5 * L_ring
    raise ValueError(f"Unknown section {section!r}")


def section_legacy_barrier_height_mev(section: str) -> float:
    """Return the center barrier height implied by the original transverse mode."""

    if section == "L":
        return max(0.0, min(QPC_MAX_HEIGHT_MEV, V0_L + 0.5 * h_bar * np.sqrt(2 * Uy_L / m)))
    if section == "U":
        return max(0.0, min(QPC_MAX_HEIGHT_MEV, V0_U + 0.5 * h_bar * np.sqrt(2 * Uy_U / m)))
    if section == "R":
        return max(0.0, min(QPC_MAX_HEIGHT_MEV, V0_R + 0.5 * h_bar * np.sqrt(2 * Uy_R / m)))
    if section == "D":
        return 0.0
    raise ValueError(f"Unknown section {section!r}")


def section_longitudinal_curvature(section: str) -> float:
    """Return the longitudinal curvature parameter associated with a section."""

    if section == "L":
        return Ux_L
    if section == "U":
        return Ux_U
    if section == "R":
        return Ux_R
    if section == "D":
        return 0.0
    raise ValueError(f"Unknown section {section!r}")


def gaussian_qpc_potential(section: str, s_nm: float) -> float:
    """Return the smooth calibrated barrier used for transport benchmarks."""

    height_mev = GAUSSIAN_QPC_HEIGHTS_MEV.get(section, 0.0)
    if height_mev <= 0.0:
        return 0.0

    width_nm = GAUSSIAN_QPC_WIDTHS_NM[section]
    center_nm = section_qpc_center_nm(section)
    normalized_distance = (s_nm - center_nm) / width_nm
    return float(height_mev * np.exp(-0.5 * normalized_distance**2))


def legacy_unbounded_qpc_potential(section: str, s_nm: float) -> float:
    """Return the original unbounded local-saddle expression for comparison only."""

    if section == "L":
        return float(V0_L - Ux_L * (s_nm - s0_L) ** 2 + 0.5 * h_bar * np.sqrt(2 * Uy_L / m))
    if section == "U":
        return float(V0_U - Ux_U * (s_nm - s0_U) ** 2 + 0.5 * h_bar * np.sqrt(2 * Uy_U / m))
    if section == "D":
        return 0.0
    if section == "R":
        return float(V0_R - Ux_R * (s_nm - s0_R) ** 2 + 0.5 * h_bar * np.sqrt(2 * Uy_R / m))
    raise ValueError(f"Unknown section {section!r}")


def legacy_localized_qpc_potential(section: str, s_nm: float) -> float:
    """Return a finite-width version of the legacy QPC idea.

    The original formula was a local saddle-point approximation. Applied over
    the full arm/lead length it inevitably turned into a very deep negative
    parabola far from the constriction. Here we keep the legacy center height,
    but we localize the constriction to a finite region and force the tails to
    relax back to zero instead of to `-infinity`.
    """

    barrier_height_mev = section_legacy_barrier_height_mev(section)
    if barrier_height_mev <= 0.0:
        return 0.0

    curvature = max(section_longitudinal_curvature(section), 1e-12)
    half_width_nm = LEGACY_LOCALIZATION_WIDTH_SCALE * np.sqrt(barrier_height_mev / curvature)
    center_nm = section_qpc_center_nm(section)
    shape = max(0.0, 1.0 - ((s_nm - center_nm) / half_width_nm) ** 2)
    return float(barrier_height_mev * shape**2)


def V(section: str, i: int) -> float:
    """Return the active potential sampled at node `i` of the requested section.

    This dispatcher makes the current transport experiment explicit. The same
    solver can therefore be run with:

    - no barrier,
    - a smooth calibrated Gaussian barrier,
    - a corrected finite-width version of the legacy QPC, or
    - the original unbounded local-saddle expression for comparison only.
    """

    s_nm = section_position_nm(section, i)

    if POTENTIAL_MODEL == "none":
        return 0.0
    if POTENTIAL_MODEL == "gaussian_qpc":
        return gaussian_qpc_potential(section, s_nm)
    if POTENTIAL_MODEL == "legacy_localized_qpc":
        return legacy_localized_qpc_potential(section, s_nm)
    if POTENTIAL_MODEL == "legacy_unbounded_qpc":
        return legacy_unbounded_qpc_potential(section, s_nm)
    raise ValueError(f"Unknown potential model {POTENTIAL_MODEL!r}")


def summarize_potential_by_section() -> dict[str, dict[str, float]]:
    """Return min/max/center potential values for each section."""

    summary = {}
    for section, section_size in get_section_layout():
        values = np.array([V(section, i) for i in range(section_size)], dtype=float)
        center_index = section_size // 2
        summary[section] = {
            "min_mev": float(values.min()),
            "max_mev": float(values.max()),
            "center_mev": float(values[center_index]),
        }
    return summary


def build_site_potential(layout: SingleRingLayout | None = None) -> np.ndarray:
    """Return the physical on-site potential for every unique graph node."""

    layout = layout or build_single_ring_layout()
    potential = np.zeros(layout.unique_site_count, dtype=float)

    # Left lead nodes are stored from the external boundary towards the junction,
    # while the legacy potential helper expects index 0 at the junction.
    for local_index, site in enumerate(layout.left_lead_sites[:-1]):
        potential[site] = V("L", N_l - 1 - local_index)

    for local_index, site in enumerate(layout.right_lead_sites[1:], start=1):
        potential[site] = V("R", local_index)

    for local_index, site in enumerate(layout.upper_internal_sites, start=1):
        potential[site] = V("U", local_index)

    for local_index, site in enumerate(layout.lower_internal_sites, start=1):
        potential[site] = V("D", local_index)

    potential[layout.left_junction_site] = np.mean((V("L", 0), V("U", 0), V("D", 0)))
    potential[layout.right_junction_site] = np.mean((V("R", 0), V("U", N_R - 1), V("D", N_R - 1)))

    return potential


def linearize_site_values_for_plot(site_values: np.ndarray, layout: SingleRingLayout | None = None) -> np.ndarray:
    """Duplicate shared junction values so the branched graph can be plotted linearly."""

    layout = layout or build_single_ring_layout()
    return np.asarray(site_values[layout.plot_sites], dtype=float)


def build_potential_profile():
    """Assemble the plot-ready potential profile in the legacy L -> U -> D -> R order."""

    layout = build_single_ring_layout()
    node_indices = np.arange(layout.plot_sites.size)
    potentials = linearize_site_values_for_plot(build_site_potential(layout), layout)
    return node_indices, potentials, list(layout.plot_section_labels)


def build_single_ring_hamiltonian(layout: SingleRingLayout | None = None) -> sp.csr_matrix:
    """Build the Hermitian graph Hamiltonian for the spinor single-ring device.

    The Hamiltonian uses a nearest-neighbour finite-difference graph model:

    - each edge contributes `+t` to the diagonal of both endpoints,
    - each edge contributes `-t * exp(i phi) * U_spin` to the forward hop,
    - the reverse hop is the Hermitian adjoint automatically.

    This makes `H = H†` by construction, which is the key ingredient required
    for a norm-preserving Crank-Nicolson step.
    """

    layout = layout or build_single_ring_layout()
    hamiltonian = sp.lil_matrix((layout.spinor_size, layout.spinor_size), dtype=complex)
    spin_identity = np.eye(2, dtype=complex)
    spin_rotation_ring = U_rashba(phi_so_link)
    site_potential = build_site_potential(layout)

    def add_spin_block(site_row: int, site_col: int, block: np.ndarray) -> None:
        row_slice = _spin_slice(site_row)
        col_slice = _spin_slice(site_col)
        for local_row in range(2):
            for local_col in range(2):
                value = block[local_row, local_col]
                if _is_close_to_zero(value):
                    continue
                hamiltonian[row_slice.start + local_row, col_slice.start + local_col] += value

    def add_site_energy(site: int, energy_mev: float) -> None:
        if _is_close_to_zero(energy_mev):
            return
        add_spin_block(site, site, energy_mev * spin_identity)

    def add_link(
        site_left: int,
        site_right: int,
        hopping_mev: float,
        phase: float,
        spin_matrix: np.ndarray,
    ) -> None:
        """Add one Hermitian hopping term plus the matching kinetic diagonals."""

        add_site_energy(site_left, hopping_mev)
        add_site_energy(site_right, hopping_mev)

        forward_block = -hopping_mev * np.exp(1j * phase) * spin_matrix
        reverse_block = forward_block.conj().T
        add_spin_block(site_left, site_right, forward_block)
        add_spin_block(site_right, site_left, reverse_block)

    # Physical on-site potential.
    for site, potential_mev in enumerate(site_potential):
        add_site_energy(site, potential_mev)

    # Lead chains are spin diagonal.
    for site_left, site_right in zip(layout.left_lead_sites[:-1], layout.left_lead_sites[1:]):
        add_link(site_left, site_right, t_lead, phi_L, spin_identity)

    for site_left, site_right in zip(layout.right_lead_sites[:-1], layout.right_lead_sites[1:]):
        add_link(site_left, site_right, t_lead, phi_R, spin_identity)

    # Ring branches carry both the AB phase and the Rashba rotation.
    for site_left, site_right in zip(layout.upper_arm_sites[:-1], layout.upper_arm_sites[1:]):
        add_link(site_left, site_right, t_ring, phi_U, spin_rotation_ring)

    for site_left, site_right in zip(layout.lower_arm_sites[:-1], layout.lower_arm_sites[1:]):
        add_link(site_left, site_right, t_ring, phi_D, spin_rotation_ring)

    return hamiltonian.tocsr()


def matrix_A_B_generator_single_ring(requested_ring_points: int, k: float):
    """Build sparse Crank-Nicolson matrices for the stable graph Hamiltonian.

    Parameters
    ----------
    requested_ring_points:
        Kept for compatibility with the original `main.py`. The function now
        validates that the caller is using the same discretization as this
        module, because the graph layout is defined globally.
    k:
        Retained for signature compatibility. The matrix pair no longer needs an
        outgoing-wave boundary condition, so `k` only matters for the initial
        packet created in `main.py`.
    """

    if requested_ring_points != N_R:
        raise ValueError(
            f"Requested N_R={requested_ring_points} but tools.py is configured for N_R={N_R}. "
            "Update the module constants first so the graph geometry remains consistent."
        )

    _ = k
    layout = build_single_ring_layout()
    hamiltonian = build_single_ring_hamiltonian(layout)
    identity = sp.identity(layout.spinor_size, format="csr", dtype=complex)
    cn_prefactor = 1j * dt / (2 * h_bar)

    A = (identity + cn_prefactor * hamiltonian).tocsr()
    B = (identity - cn_prefactor * hamiltonian).tocsr()
    return A, B, layout.spinor_size
