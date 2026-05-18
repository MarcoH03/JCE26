"""Gaussian QPC sweep plots for the JCE26 quantum ring.

Produces two families of 3-D surface plots:

Family A — fixed Gaussian height, sweep width on one axis and one of the
           ring parameters (U_leads, U_U, phi_so) on the other axis.

Family B — fixed Gaussian width, sweep height on one axis and one of the
           ring parameters on the other axis.

In every plot the lead QPCs (L = R) use the Gaussian model and the upper-arm
QPC uses the legacy_localized_qpc model parameterized by U_U (Ux_U = Uy_U).
The lower arm D has no QPC throughout.

Sweep variables on the second axis (one plot per variable, per family):
  1. U_leads  [meV/nm²]  — curvature of the lead QPCs (kept equal L=R)
  2. U_U      [meV/nm²]  — curvature of the upper-arm QPC
  3. phi_so              — Rashba phase k_so·R

Run:
    python gaussian_qpc_plots.py              # all plots, default resolution
    python gaussian_qpc_plots.py --n-grid 5  # coarse test
    python gaussian_qpc_plots.py --plots A1 B2  # specific subsets
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

import tools as t
from conductance import run_single_conductance

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "gaussian_qpc_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Shared simulation settings
# ---------------------------------------------------------------------------
FERMI_ENERGY_MEV       = 4.19
TOTAL_TIME_PS          = 13.5
PACKET_CENTER_FRACTION = 0.8
PACKET_WIDTH_NM        = 150.0
CMAP                   = "viridis"

# ---------------------------------------------------------------------------
# phi_so conversion
# ---------------------------------------------------------------------------
BASE_PARAMS = t.PhysicsParams(
    potential_model="gaussian_qpc",
    gaussian_qpc_heights_mev={"L": 0.0, "U": 0.0, "D": 0.0, "R": 0.0},
    gaussian_qpc_widths_nm={"L": 120.0, "U": 90.0, "D": 120.0, "R": 120.0},
)


def _alpha_from_phi_so(phi_so: float, p: t.PhysicsParams = BASE_PARAMS) -> float:
    return phi_so * t.h_bar**2 / (p.m * p.R)


# ---------------------------------------------------------------------------
# Parameter constructor
#
# Lead QPCs (L = R): Gaussian model, height = gauss_height_mev, width = gauss_width_nm
# Upper-arm QPC:     legacy_localized model,  Ux_U = Uy_U = U_U
# The potential_model field selects which model is used for LEADS only;
# the upper arm is always legacy_localized regardless.  We achieve this by
# building the potential vector in a mixed way via build_site_potential, but
# since tools.py uses the same model for all sections, we need to set the
# Gaussian height for "U" = 0 and rely on the legacy parameters for Ux_U/Uy_U.
#
# Concretely: potential_model = "gaussian_qpc" for L and R
#             but for U we set height=0 so gaussian_qpc returns 0 there,
#             then we ALSO set Ux_U and Uy_U so that a second call with
#             "legacy_localized_qpc" would give the right U potential.
# PROBLEM: tools.py uses a single model selector — it can't mix models per section.
#
# SOLUTION: use "gaussian_qpc" as the model, set nonzero heights for L and R,
#           and zero height for U and D.  Then add the upper-arm barrier by
#           ALSO setting the gaussian height for U to a reasonable value derived
#           from U_U (we convert U_U to an equivalent Gaussian height using the
#           zero-point energy: h_eff = hbar*sqrt(2*U_U/m)/2).
#           Width for U arm is kept at a fixed 90 nm (can be changed).
# ---------------------------------------------------------------------------

def _U_to_equiv_gaussian_height(U_val: float, p: t.PhysicsParams) -> float:
    """Convert a QPC curvature U [meV/nm^2] to an equivalent Gaussian peak height.

    We use the zero-point energy of the transverse mode as the effective barrier
    height, matching the legacy_localized_qpc convention:
        h_eff = hbar * sqrt(2 * U / m) / 2
    This is the threshold energy at which the analytical T(E_g) = 0.5.
    """
    if U_val <= 0.0:
        return 0.0
    return 0.5 * t.h_bar * np.sqrt(2 * U_val / p.m)


def make_params_gaussian(
    gauss_height_mev_leads: float,
    gauss_width_nm_leads: float,
    U_U: float,
    phi_so: float,
    p_base: t.PhysicsParams = BASE_PARAMS,
) -> t.PhysicsParams:
    """Build PhysicsParams for one Gaussian-lead-QPC operating point.

    Lead QPCs (L = R): Gaussian with specified height and width.
    Upper-arm QPC: equivalent Gaussian derived from U_U via zero-point energy.
    Lower arm: no QPC (height = 0).
    """
    alpha = _alpha_from_phi_so(phi_so, p_base)
    h_U   = _U_to_equiv_gaussian_height(U_U, p_base)

    return p_base.with_changes(
        alpha=alpha,
        potential_model="gaussian_qpc",
        gaussian_qpc_heights_mev={
            "L": gauss_height_mev_leads,
            "U": h_U,
            "D": 0.0,
            "R": gauss_height_mev_leads,   # L = R always
        },
        gaussian_qpc_widths_nm={
            "L": gauss_width_nm_leads,
            "U": 90.0,   # fixed upper-arm width
            "D": 120.0,
            "R": gauss_width_nm_leads,     # L = R always
        },
        # Keep legacy parameters in sync for bookkeeping
        Ux_U=U_U,
        Uy_U=U_U,
    )


def make_params_gaussian_Uleads(
    gauss_height_mev_leads: float,
    gauss_width_nm_leads: float,
    U_leads: float,
    phi_so: float,
    p_base: t.PhysicsParams = BASE_PARAMS,
) -> t.PhysicsParams:
    """Lead QPC: Gaussian (height + width).  Lead curvature also stored in legacy fields.
    Upper-arm QPC: fixed U_U = 0.1 meV/nm^2 (low barrier, mostly transparent)."""
    U_U_fixed = 0.1
    alpha  = _alpha_from_phi_so(phi_so, p_base)
    h_U    = _U_to_equiv_gaussian_height(U_U_fixed, p_base)

    # Lead Gaussian height is the swept variable; U_leads controls the legacy fields
    # for bookkeeping but the actual barrier is the Gaussian.
    h_L = gauss_height_mev_leads
    w_L = gauss_width_nm_leads

    return p_base.with_changes(
        alpha=alpha,
        potential_model="gaussian_qpc",
        gaussian_qpc_heights_mev={"L": h_L, "U": h_U, "D": 0.0, "R": h_L},
        gaussian_qpc_widths_nm={"L": w_L, "U": 90.0, "D": 120.0, "R": w_L},
        Ux_L=U_leads, Uy_L=U_leads,
        Ux_R=U_leads, Uy_R=U_leads,
        Ux_U=U_U_fixed, Uy_U=U_U_fixed,
    )


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def compute_G(p: t.PhysicsParams, verbose: bool = False) -> float:
    result = run_single_conductance(
        p,
        fermi_energy_mev=FERMI_ENERGY_MEV,
        total_time_ps=TOTAL_TIME_PS,
        packet_center_fraction=PACKET_CENTER_FRACTION,
        packet_width_nm=PACKET_WIDTH_NM,
        keep_time_series=False,
        verbose=verbose,
    )
    return result.G_over_G0


# ---------------------------------------------------------------------------
# Plot helper: 3-D surface
# ---------------------------------------------------------------------------

def _surface3d(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    G_grid: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: Path,
) -> None:
    X, Y = np.meshgrid(x_vals, y_vals)
    fig  = plt.figure(figsize=(10, 7))
    ax   = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, G_grid, cmap=CMAP, vmin=0.0, vmax=1.0,
                           linewidth=0, antialiased=True, alpha=0.92)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, pad=0.1)
    cbar.set_label(r"$G / G_0$", fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
    ax.set_zlabel(r"$G / G_0$", fontsize=11, labelpad=6)
    ax.set_zlim(0.0, 1.0)
    ax.set_title(title, fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _run_grid(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    make_p,           # callable(x_val, y_val) -> PhysicsParams
    label: str,
) -> np.ndarray:
    nx, ny = len(x_vals), len(y_vals)
    G_grid = np.zeros((ny, nx))
    total  = nx * ny
    for j, yv in enumerate(y_vals):
        for i, xv in enumerate(x_vals):
            print(f"  {label} [{j*nx+i+1}/{total}]  x={xv:.4g}  y={yv:.4g}")
            p = make_p(xv, yv)
            G_grid[j, i] = compute_G(p)
    return G_grid


# ---------------------------------------------------------------------------
# Default grid sizes
# ---------------------------------------------------------------------------
N_GRID = 10   # points per axis (total runs = N_GRID^2 per plot)


# ============================================================================
# FAMILY A — fixed Gaussian height, sweep width vs ring parameter
# ============================================================================
#
# Plot A1: G vs (Gaussian width, U_leads)   height fixed
# Plot A2: G vs (Gaussian width, U_U)       height fixed
# Plot A3: G vs (Gaussian width, phi_so)    height fixed
#
# Fixed values chosen so the barrier is visible but not opaque:
FIXED_GAUSS_HEIGHT = 5.0   # meV  (> E_F = 4.19, gives partial transmission)
FIXED_U_U_FOR_A    = 0.1   # meV/nm^2  (for plots A1, A3)
FIXED_PHI_SO_FOR_A = 24.05
FIXED_U_LEADS_FOR_A2 = 2.0  # meV/nm^2


def plot_A1(n: int = N_GRID) -> None:
    """G vs (Gaussian width of lead QPCs, U_leads legacy curvature)
    Height fixed.  phi_so fixed.  U_U fixed."""
    HEIGHT   = FIXED_GAUSS_HEIGHT
    PHI_SO   = FIXED_PHI_SO_FOR_A
    U_U_FIX  = FIXED_U_U_FOR_A

    # ---- sweep ranges ----
    width_vals  = np.linspace(20.0, 200.0, n)    # nm
    Ulead_vals  = np.linspace(0.01,   6.0, n)    # meV/nm^2
    # ----------------------

    print(f"\n=== Plot A1: G vs (Gauss width, U_leads)  height={HEIGHT}, phi_so={PHI_SO} ===")

    def make_p(width, U_leads):
        h_U = _U_to_equiv_gaussian_height(U_U_FIX, BASE_PARAMS)
        return BASE_PARAMS.with_changes(
            alpha=_alpha_from_phi_so(PHI_SO),
            potential_model="gaussian_qpc",
            gaussian_qpc_heights_mev={"L": HEIGHT, "U": h_U, "D": 0.0, "R": HEIGHT},
            gaussian_qpc_widths_nm={"L": width, "U": 90.0, "D": 120.0, "R": width},
            Ux_L=U_leads, Uy_L=U_leads, Ux_R=U_leads, Uy_R=U_leads,
            Ux_U=U_U_FIX, Uy_U=U_U_FIX,
        )

    G = _run_grid(width_vals, Ulead_vals, make_p, "A1")
    np.savez_compressed(OUTPUT_DIR / "A1_data.npz",
                        gauss_width_nm=width_vals, U_leads=Ulead_vals, G_grid=G,
                        gauss_height=np.array(HEIGHT), phi_so=np.array(PHI_SO))
    _surface3d(width_vals, Ulead_vals, G,
               xlabel=r"Gaussian width $\sigma$ (nm)",
               ylabel=r"$U_{leads}$ (meV/nm²)",
               title=rf"$G/G_0$: Gaussian lead QPC  (h={HEIGHT} meV, $\phi_{{so}}$={PHI_SO})",
               save_path=OUTPUT_DIR / "A1_G_vs_width_Uleads.png")


def plot_A2(n: int = N_GRID) -> None:
    """G vs (Gaussian width, U_U)  Height and phi_so fixed, U_leads fixed."""
    HEIGHT      = FIXED_GAUSS_HEIGHT
    PHI_SO      = FIXED_PHI_SO_FOR_A
    U_LEADS_FIX = FIXED_U_LEADS_FOR_A2

    # ---- sweep ranges ----
    width_vals = np.linspace(20.0, 200.0, n)   # nm
    U_U_vals   = np.linspace(0.01,   6.0, n)   # meV/nm^2
    # ----------------------

    print(f"\n=== Plot A2: G vs (Gauss width, U_U)  height={HEIGHT}, phi_so={PHI_SO} ===")

    def make_p(width, U_U):
        h_U = _U_to_equiv_gaussian_height(U_U, BASE_PARAMS)
        return BASE_PARAMS.with_changes(
            alpha=_alpha_from_phi_so(PHI_SO),
            potential_model="gaussian_qpc",
            gaussian_qpc_heights_mev={"L": HEIGHT, "U": h_U, "D": 0.0, "R": HEIGHT},
            gaussian_qpc_widths_nm={"L": width, "U": 90.0, "D": 120.0, "R": width},
            Ux_L=U_LEADS_FIX, Uy_L=U_LEADS_FIX,
            Ux_R=U_LEADS_FIX, Uy_R=U_LEADS_FIX,
            Ux_U=U_U, Uy_U=U_U,
        )

    G = _run_grid(width_vals, U_U_vals, make_p, "A2")
    np.savez_compressed(OUTPUT_DIR / "A2_data.npz",
                        gauss_width_nm=width_vals, U_U=U_U_vals, G_grid=G,
                        gauss_height=np.array(HEIGHT), phi_so=np.array(PHI_SO))
    _surface3d(width_vals, U_U_vals, G,
               xlabel=r"Gaussian width $\sigma$ (nm)",
               ylabel=r"$U_U$ (meV/nm²)",
               title=rf"$G/G_0$: Gaussian lead QPC  (h={HEIGHT} meV, $\phi_{{so}}$={PHI_SO})",
               save_path=OUTPUT_DIR / "A2_G_vs_width_UU.png")


def plot_A3(n: int = N_GRID) -> None:
    """G vs (Gaussian width, phi_so)  Height, U_leads, U_U fixed."""
    HEIGHT      = FIXED_GAUSS_HEIGHT
    U_U_FIX     = FIXED_U_U_FOR_A
    U_LEADS_FIX = FIXED_U_LEADS_FOR_A2

    # ---- sweep ranges ----
    width_vals  = np.linspace(20.0, 200.0, n)
    phi_so_vals = np.linspace(0.5,   30.0, n)
    # ----------------------

    print(f"\n=== Plot A3: G vs (Gauss width, phi_so)  height={HEIGHT} ===")

    def make_p(width, phi_so):
        h_U = _U_to_equiv_gaussian_height(U_U_FIX, BASE_PARAMS)
        return BASE_PARAMS.with_changes(
            alpha=_alpha_from_phi_so(phi_so),
            potential_model="gaussian_qpc",
            gaussian_qpc_heights_mev={"L": HEIGHT, "U": h_U, "D": 0.0, "R": HEIGHT},
            gaussian_qpc_widths_nm={"L": width, "U": 90.0, "D": 120.0, "R": width},
            Ux_L=U_LEADS_FIX, Uy_L=U_LEADS_FIX,
            Ux_R=U_LEADS_FIX, Uy_R=U_LEADS_FIX,
            Ux_U=U_U_FIX, Uy_U=U_U_FIX,
        )

    G = _run_grid(width_vals, phi_so_vals, make_p, "A3")
    np.savez_compressed(OUTPUT_DIR / "A3_data.npz",
                        gauss_width_nm=width_vals, phi_so=phi_so_vals, G_grid=G,
                        gauss_height=np.array(HEIGHT))
    _surface3d(width_vals, phi_so_vals, G,
               xlabel=r"Gaussian width $\sigma$ (nm)",
               ylabel=r"$\phi_{so}$",
               title=rf"$G/G_0$: Gaussian lead QPC  (h={HEIGHT} meV)",
               save_path=OUTPUT_DIR / "A3_G_vs_width_phi_so.png")


# ============================================================================
# FAMILY B — fixed Gaussian width, sweep height vs ring parameter
# ============================================================================
#
# Plot B1: G vs (Gaussian height, U_leads)   width fixed
# Plot B2: G vs (Gaussian height, U_U)       width fixed
# Plot B3: G vs (Gaussian height, phi_so)    width fixed

FIXED_GAUSS_WIDTH  = 80.0   # nm  (well-localized QPC)
FIXED_U_U_FOR_B    = 0.1    # meV/nm^2
FIXED_U_LEADS_FOR_B2 = 2.0  # meV/nm^2


def plot_B1(n: int = N_GRID) -> None:
    """G vs (Gaussian height, U_leads)  Width and phi_so fixed, U_U fixed."""
    WIDTH    = FIXED_GAUSS_WIDTH
    PHI_SO   = FIXED_PHI_SO_FOR_A
    U_U_FIX  = FIXED_U_U_FOR_B

    # ---- sweep ranges ----
    height_vals = np.linspace(0.5, 10.0, n)    # meV
    Ulead_vals  = np.linspace(0.01,  6.0, n)   # meV/nm^2
    # ----------------------

    print(f"\n=== Plot B1: G vs (Gauss height, U_leads)  width={WIDTH} nm ===")

    def make_p(height, U_leads):
        h_U = _U_to_equiv_gaussian_height(U_U_FIX, BASE_PARAMS)
        return BASE_PARAMS.with_changes(
            alpha=_alpha_from_phi_so(PHI_SO),
            potential_model="gaussian_qpc",
            gaussian_qpc_heights_mev={"L": height, "U": h_U, "D": 0.0, "R": height},
            gaussian_qpc_widths_nm={"L": WIDTH, "U": 90.0, "D": 120.0, "R": WIDTH},
            Ux_L=U_leads, Uy_L=U_leads, Ux_R=U_leads, Uy_R=U_leads,
            Ux_U=U_U_FIX, Uy_U=U_U_FIX,
        )

    G = _run_grid(height_vals, Ulead_vals, make_p, "B1")
    np.savez_compressed(OUTPUT_DIR / "B1_data.npz",
                        gauss_height_mev=height_vals, U_leads=Ulead_vals, G_grid=G,
                        gauss_width=np.array(WIDTH), phi_so=np.array(PHI_SO))
    _surface3d(height_vals, Ulead_vals, G,
               xlabel=r"Gaussian height $h$ (meV)",
               ylabel=r"$U_{leads}$ (meV/nm²)",
               title=rf"$G/G_0$: Gaussian lead QPC  ($\sigma$={WIDTH} nm, $\phi_{{so}}$={PHI_SO})",
               save_path=OUTPUT_DIR / "B1_G_vs_height_Uleads.png")


def plot_B2(n: int = N_GRID) -> None:
    """G vs (Gaussian height, U_U)  Width, phi_so, U_leads fixed."""
    WIDTH       = FIXED_GAUSS_WIDTH
    PHI_SO      = FIXED_PHI_SO_FOR_A
    U_LEADS_FIX = FIXED_U_LEADS_FOR_B2

    # ---- sweep ranges ----
    height_vals = np.linspace(0.5, 10.0, n)
    U_U_vals    = np.linspace(0.01,  6.0, n)
    # ----------------------

    print(f"\n=== Plot B2: G vs (Gauss height, U_U)  width={WIDTH} nm ===")

    def make_p(height, U_U):
        h_U = _U_to_equiv_gaussian_height(U_U, BASE_PARAMS)
        return BASE_PARAMS.with_changes(
            alpha=_alpha_from_phi_so(PHI_SO),
            potential_model="gaussian_qpc",
            gaussian_qpc_heights_mev={"L": height, "U": h_U, "D": 0.0, "R": height},
            gaussian_qpc_widths_nm={"L": WIDTH, "U": 90.0, "D": 120.0, "R": WIDTH},
            Ux_L=U_LEADS_FIX, Uy_L=U_LEADS_FIX,
            Ux_R=U_LEADS_FIX, Uy_R=U_LEADS_FIX,
            Ux_U=U_U, Uy_U=U_U,
        )

    G = _run_grid(height_vals, U_U_vals, make_p, "B2")
    np.savez_compressed(OUTPUT_DIR / "B2_data.npz",
                        gauss_height_mev=height_vals, U_U=U_U_vals, G_grid=G,
                        gauss_width=np.array(WIDTH), phi_so=np.array(PHI_SO))
    _surface3d(height_vals, U_U_vals, G,
               xlabel=r"Gaussian height $h$ (meV)",
               ylabel=r"$U_U$ (meV/nm²)",
               title=rf"$G/G_0$: Gaussian lead QPC  ($\sigma$={WIDTH} nm, $\phi_{{so}}$={PHI_SO})",
               save_path=OUTPUT_DIR / "B2_G_vs_height_UU.png")


def plot_B3(n: int = N_GRID) -> None:
    """G vs (Gaussian height, phi_so)  Width, U_leads, U_U fixed."""
    WIDTH       = FIXED_GAUSS_WIDTH
    U_U_FIX     = FIXED_U_U_FOR_B
    U_LEADS_FIX = FIXED_U_LEADS_FOR_B2

    # ---- sweep ranges ----
    height_vals = np.linspace(0.5, 10.0, n)
    phi_so_vals = np.linspace(0.5,  30.0, n)
    # ----------------------

    print(f"\n=== Plot B3: G vs (Gauss height, phi_so)  width={WIDTH} nm ===")

    def make_p(height, phi_so):
        h_U = _U_to_equiv_gaussian_height(U_U_FIX, BASE_PARAMS)
        return BASE_PARAMS.with_changes(
            alpha=_alpha_from_phi_so(phi_so),
            potential_model="gaussian_qpc",
            gaussian_qpc_heights_mev={"L": height, "U": h_U, "D": 0.0, "R": height},
            gaussian_qpc_widths_nm={"L": WIDTH, "U": 90.0, "D": 120.0, "R": WIDTH},
            Ux_L=U_LEADS_FIX, Uy_L=U_LEADS_FIX,
            Ux_R=U_LEADS_FIX, Uy_R=U_LEADS_FIX,
            Ux_U=U_U_FIX, Uy_U=U_U_FIX,
        )

    G = _run_grid(height_vals, phi_so_vals, make_p, "B3")
    np.savez_compressed(OUTPUT_DIR / "B3_data.npz",
                        gauss_height_mev=height_vals, phi_so=phi_so_vals, G_grid=G,
                        gauss_width=np.array(WIDTH))
    _surface3d(height_vals, phi_so_vals, G,
               xlabel=r"Gaussian height $h$ (meV)",
               ylabel=r"$\phi_{so}$",
               title=rf"$G/G_0$: Gaussian lead QPC  ($\sigma$={WIDTH} nm)",
               save_path=OUTPUT_DIR / "B3_G_vs_height_phi_so.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    VALID_PLOTS = ["A1", "A2", "A3", "B1", "B2", "B3"]
    parser = argparse.ArgumentParser(
        description="Gaussian QPC sweep plots for the JCE26 ring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Plots
-----
  A1  G vs (Gauss width, U_leads)   — height fixed
  A2  G vs (Gauss width, U_U)       — height fixed
  A3  G vs (Gauss width, phi_so)    — height fixed
  B1  G vs (Gauss height, U_leads)  — width fixed
  B2  G vs (Gauss height, U_U)      — width fixed
  B3  G vs (Gauss height, phi_so)   — width fixed

To change sweep ranges, edit the np.linspace calls inside each plot function.
    step = (stop - start) / (n_points - 1)

Examples
--------
  python gaussian_qpc_plots.py                        # all 6 plots
  python gaussian_qpc_plots.py --plots A1 A3 B3       # subset
  python gaussian_qpc_plots.py --n-grid 4             # coarse test
  python gaussian_qpc_plots.py --n-grid 20            # publication quality
        """,
    )
    parser.add_argument("--plots", nargs="+", default=VALID_PLOTS, choices=VALID_PLOTS,
                        metavar="ID", help="Which plots to run (default: all).")
    parser.add_argument("--n-grid", type=int, default=N_GRID,
                        help=f"Points per axis (default {N_GRID}). "
                             "Total simulations = n_grid² per plot.")
    args = parser.parse_args()

    dispatch = {
        "A1": plot_A1, "A2": plot_A2, "A3": plot_A3,
        "B1": plot_B1, "B2": plot_B2, "B3": plot_B3,
    }

    wall_t0 = time.perf_counter()
    for name in args.plots:
        dispatch[name](args.n_grid)

    elapsed = time.perf_counter() - wall_t0
    print(f"\nAll done in {elapsed:.1f} s  ({elapsed/60:.1f} min).")
    print(f"Results in: {OUTPUT_DIR.resolve()}")
