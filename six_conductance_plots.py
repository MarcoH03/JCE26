"""Six conductance plots for the JCE26 quantum ring.

Physical conventions
--------------------
- **Confinement energy** of a QPC:  E_conf = ℏ √(2 U / m)
  where U stands for either Ux or Uy (they are always equal in this script).
  Inverting:  U = m (E_conf / ℏ)² / 2

- **phi_so** (total Rashba phase around one arm):
      phi_so = k_so · R,   k_so = m α / ℏ²
  Inverting for alpha:  α = phi_so · ℏ² / (m · R)

- Lead QPCs are **always identical and symmetric** (L = R).
  The helper ``make_params`` enforces this: a single ``E_conf_leads`` value
  sets Ux_L = Uy_L = Ux_R = Uy_R = U(E_conf_leads).

- The upper-arm QPC sets its own ``E_conf_U`` → Ux_U = Uy_U = U(E_conf_U).

- The lower arm D has no QPC (Ux_D = Uy_D = 0 always).

Plots produced
--------------
1. G/G₀ vs (E_conf_leads, E_conf_U)  [2-D heatmap]  phi_so = 24.05
2. G/G₀ vs E_conf_leads              [1-D line]     E_conf_U = 0.1 meV, phi_so = 24.05
3. G/G₀ vs (E_conf_leads, phi_so)    [2-D heatmap]  E_conf_U = 2.78 meV
4. G/G₀ vs E_conf_leads              [1-D line]     E_conf_U = 2.78 meV, phi_so = 1
5. G/G₀ vs phi_so                    [1-D line]     E_conf_leads = 3.08 meV, E_conf_U = 0.2 meV
6. G/G₀ vs phi_so                    [1-D line]     E_conf_leads = 6.0  meV, E_conf_U = 0.2 meV

All results are also saved as .npz files next to this script.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import tools as t
from conductance import run_single_conductance, G0_SIEMENS


# ---------------------------------------------------------------------------
# Unit-conversion helpers
# ---------------------------------------------------------------------------

def _U_from_Econf(E_conf_mev: float, p_base: t.PhysicsParams) -> float:
    """Curvature parameter U [meV/nm²] from confinement energy [meV]."""
    return p_base.m * (E_conf_mev / t.h_bar) ** 2 / 2.0


def _alpha_from_phi_so(phi_so: float, p_base: t.PhysicsParams) -> float:
    """Rashba strength α [meV·nm] from phi_so = k_so · R."""
    return phi_so * t.h_bar ** 2 / (p_base.m * p_base.R)


def _phi_so_from_alpha(alpha: float, p_base: t.PhysicsParams) -> float:
    """phi_so from α."""
    return p_base.m * alpha * p_base.R / t.h_bar ** 2


# ---------------------------------------------------------------------------
# Canonical parameter constructor
# ---------------------------------------------------------------------------

def make_params(
    E_conf_leads: float,
    E_conf_U: float,
    phi_so: float,
    p_base: t.PhysicsParams | None = None,
) -> t.PhysicsParams:
    """Build a PhysicsParams for one (E_conf_leads, E_conf_U, phi_so) point.

    Lead QPCs are symmetric (L = R) and isotropic (Ux = Uy).
    Upper-arm QPC is isotropic (Ux_U = Uy_U).
    Lower arm D has no QPC (hardcoded to zero inside tools.py).
    All V0 offsets are set to zero; confinement energy fully controls the barrier.
    """
    if p_base is None:
        p_base = BASE_PARAMS

    U_leads = _U_from_Econf(E_conf_leads, p_base)
    U_U     = _U_from_Econf(E_conf_U,     p_base)
    alpha   = _alpha_from_phi_so(phi_so,  p_base)

    return p_base.with_changes(
        alpha = alpha,
        Ux_L  = U_leads,
        Uy_L  = U_leads,
        Ux_R  = U_leads,   # right lead mirrors left lead exactly
        Uy_R  = U_leads,
        Ux_U  = U_U,
        Uy_U  = U_U,
        V0_L  = 0.0,
        V0_U  = 0.0,
        V0_R  = 0.0,
    )


# ---------------------------------------------------------------------------
# Base parameter set shared by all plots
# ---------------------------------------------------------------------------

BASE_PARAMS = t.PhysicsParams(
    potential_model = "legacy_localized_qpc",
    # Gaussian heights not used; legacy model is active
    gaussian_qpc_heights_mev = {"L": 0.0, "U": 0.0, "D": 0.0, "R": 0.0},
)

# Verify the phi_so = 24.05 alpha
_alpha_check = _alpha_from_phi_so(24.05, BASE_PARAMS)
_phi_so_check = _phi_so_from_alpha(_alpha_check, BASE_PARAMS)
print(f"[check] phi_so=24.05 → α={_alpha_check:.4f} meV·nm → phi_so back={_phi_so_check:.4f}")


# ---------------------------------------------------------------------------
# Shared simulation settings
# ---------------------------------------------------------------------------

FERMI_ENERGY_MEV       = 4.19
TOTAL_TIME_PS          = 13.5
PACKET_CENTER_FRACTION = 0.8
PACKET_WIDTH_NM        = 150.0

OUTPUT_DIR = Path(__file__).parent / "conductance_plots"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Core runner: single (E_conf_leads, E_conf_U, phi_so) point
# ---------------------------------------------------------------------------

def compute_G(
    E_conf_leads: float,
    E_conf_U: float,
    phi_so: float,
    p_base: t.PhysicsParams = BASE_PARAMS,
    verbose: bool = False,
) -> float:
    """Return G/G₀ for one parameter combination."""
    p = make_params(E_conf_leads, E_conf_U, phi_so, p_base)
    result = run_single_conductance(
        p,
        fermi_energy_mev       = FERMI_ENERGY_MEV,
        total_time_ps          = TOTAL_TIME_PS,
        packet_center_fraction = PACKET_CENTER_FRACTION,
        packet_width_nm        = PACKET_WIDTH_NM,
        keep_time_series       = False,
        verbose                = verbose,
    )
    return result.G_over_G0


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

CMAP = "viridis"

def _heatmap(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    G_grid: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: Path,
) -> None:
    """Save a 2-D heatmap of G/G₀."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(
        x_vals, y_vals, G_grid,
        cmap=CMAP, shading="nearest",
        vmin=0.0, vmax=1.0,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$G / G_0$", fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _lineplot(
    x_vals: np.ndarray,
    G_vals: np.ndarray,
    xlabel: str,
    title: str,
    save_path: Path,
    extra_label: str = "",
) -> None:
    """Save a 1-D conductance line plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_vals, G_vals, "o-", color="tab:blue", linewidth=1.8,
            markersize=5, label=extra_label if extra_label else r"$G/G_0$")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(r"$G / G_0$", fontsize=12)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    if extra_label:
        ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Grid resolution  (reduce for quick tests; increase for publication)
# ---------------------------------------------------------------------------
# Each full-simulation run takes O(10–60 s depending on hardware.
# A 10×10 grid = 100 runs; a 20×20 = 400 runs.
# Override these at the top of __main__ if you want finer resolution.

N_LEADS  = 10   # number of E_conf_leads points
N_U      = 10   # number of E_conf_U  (or phi_so) points on the second axis
N_LINE   = 15   # number of points for 1-D line plots


# ---------------------------------------------------------------------------
# Plot 1: G vs (E_conf_leads, E_conf_U)  — 2-D heatmap
#         phi_so = 24.05  (fixed)
# ---------------------------------------------------------------------------

def plot1(n_leads: int = N_LEADS, n_U: int = N_U) -> None:
    PHI_SO    = 24.05
    E_L_vals  = np.linspace(0.05, 8.0, n_leads)   # meV
    E_U_vals  = np.linspace(0.05, 8.0, n_U)        # meV

    print(f"\n=== Plot 1: G vs (E_conf_leads, E_conf_U)  phi_so={PHI_SO} ===")
    G_grid = np.zeros((n_U, n_leads))

    for j, E_U in enumerate(E_U_vals):
        for i, E_L in enumerate(E_L_vals):
            print(f"  [{j*n_leads+i+1}/{n_leads*n_U}] E_L={E_L:.2f}  E_U={E_U:.2f}")
            G_grid[j, i] = compute_G(E_L, E_U, PHI_SO)

    np.savez_compressed(
        OUTPUT_DIR / "plot1_data.npz",
        E_conf_leads=E_L_vals, E_conf_U=E_U_vals,
        G_grid=G_grid, phi_so=np.array(PHI_SO),
    )
    _heatmap(
        E_L_vals, E_U_vals, G_grid,
        xlabel=r"$E_{conf}^{leads}$ (meV)",
        ylabel=r"$E_{conf}^{U}$ (meV)",
        title=rf"$G/G_0$ vs confinement energies  ($\phi_{{so}}={PHI_SO}$)",
        save_path=OUTPUT_DIR / "plot1_G_vs_Econf_leads_Econf_U.png",
    )


# ---------------------------------------------------------------------------
# Plot 2: G vs E_conf_leads  — 1-D line
#         E_conf_U = 0.1 meV, phi_so = 24.05  (fixed)
# ---------------------------------------------------------------------------

def plot2(n_line: int = N_LINE) -> None:
    PHI_SO   = 24.05
    E_CONF_U = 0.1   # meV
    E_L_vals = np.linspace(0.05, 8.0, n_line)

    print(f"\n=== Plot 2: G vs E_conf_leads  E_conf_U={E_CONF_U} meV, phi_so={PHI_SO} ===")
    G_vals = np.array([
        compute_G(E_L, E_CONF_U, PHI_SO)
        for E_L in E_L_vals
    ])

    np.savez_compressed(
        OUTPUT_DIR / "plot2_data.npz",
        E_conf_leads=E_L_vals, G=G_vals,
        E_conf_U=np.array(E_CONF_U), phi_so=np.array(PHI_SO),
    )
    _lineplot(
        E_L_vals, G_vals,
        xlabel=r"$E_{conf}^{leads}$ (meV)",
        title=(rf"$G/G_0$ vs lead confinement  "
               rf"($E_{{conf}}^U={E_CONF_U}$ meV, $\phi_{{so}}={PHI_SO}$)"),
        save_path=OUTPUT_DIR / "plot2_G_vs_Econf_leads.png",
    )


# ---------------------------------------------------------------------------
# Plot 3: G vs (E_conf_leads, phi_so)  — 2-D heatmap
#         E_conf_U = 2.78 meV  (fixed)
# ---------------------------------------------------------------------------

def plot3(n_leads: int = N_LEADS, n_phi: int = N_U) -> None:
    E_CONF_U    = 2.78   # meV
    E_L_vals    = np.linspace(0.05, 8.0, n_leads)
    phi_so_vals = np.linspace(0.5, 30.0, n_phi)

    print(f"\n=== Plot 3: G vs (E_conf_leads, phi_so)  E_conf_U={E_CONF_U} meV ===")
    G_grid = np.zeros((n_phi, n_leads))

    for j, phi_so in enumerate(phi_so_vals):
        for i, E_L in enumerate(E_L_vals):
            print(f"  [{j*n_leads+i+1}/{n_leads*n_phi}] E_L={E_L:.2f}  phi_so={phi_so:.2f}")
            G_grid[j, i] = compute_G(E_L, E_CONF_U, phi_so)

    np.savez_compressed(
        OUTPUT_DIR / "plot3_data.npz",
        E_conf_leads=E_L_vals, phi_so=phi_so_vals,
        G_grid=G_grid, E_conf_U=np.array(E_CONF_U),
    )
    _heatmap(
        E_L_vals, phi_so_vals, G_grid,
        xlabel=r"$E_{conf}^{leads}$ (meV)",
        ylabel=r"$\phi_{so}$",
        title=rf"$G/G_0$ vs lead confinement and $\phi_{{so}}$  ($E_{{conf}}^U={E_CONF_U}$ meV)",
        save_path=OUTPUT_DIR / "plot3_G_vs_Econf_leads_phi_so.png",
    )


# ---------------------------------------------------------------------------
# Plot 4: G vs E_conf_leads  — 1-D line
#         E_conf_U = 2.78 meV, phi_so = 1  (fixed)
# ---------------------------------------------------------------------------

def plot4(n_line: int = N_LINE) -> None:
    PHI_SO   = 1.0
    E_CONF_U = 2.78   # meV
    E_L_vals = np.linspace(0.05, 8.0, n_line)

    print(f"\n=== Plot 4: G vs E_conf_leads  E_conf_U={E_CONF_U} meV, phi_so={PHI_SO} ===")
    G_vals = np.array([
        compute_G(E_L, E_CONF_U, PHI_SO)
        for E_L in E_L_vals
    ])

    np.savez_compressed(
        OUTPUT_DIR / "plot4_data.npz",
        E_conf_leads=E_L_vals, G=G_vals,
        E_conf_U=np.array(E_CONF_U), phi_so=np.array(PHI_SO),
    )
    _lineplot(
        E_L_vals, G_vals,
        xlabel=r"$E_{conf}^{leads}$ (meV)",
        title=(rf"$G/G_0$ vs lead confinement  "
               rf"($E_{{conf}}^U={E_CONF_U}$ meV, $\phi_{{so}}={PHI_SO}$)"),
        save_path=OUTPUT_DIR / "plot4_G_vs_Econf_leads.png",
    )


# ---------------------------------------------------------------------------
# Plot 5: G vs phi_so  — 1-D line
#         E_conf_leads = 3.08 meV, E_conf_U = 0.2 meV
# ---------------------------------------------------------------------------

def plot5(n_line: int = N_LINE) -> None:
    E_CONF_LEADS = 3.08   # meV
    E_CONF_U     = 0.2    # meV
    phi_so_vals  = np.linspace(0.5, 30.0, n_line)

    print(f"\n=== Plot 5: G vs phi_so  E_conf_leads={E_CONF_LEADS} meV, E_conf_U={E_CONF_U} meV ===")
    G_vals = np.array([
        compute_G(E_CONF_LEADS, E_CONF_U, phi_so)
        for phi_so in phi_so_vals
    ])

    np.savez_compressed(
        OUTPUT_DIR / "plot5_data.npz",
        phi_so=phi_so_vals, G=G_vals,
        E_conf_leads=np.array(E_CONF_LEADS), E_conf_U=np.array(E_CONF_U),
    )
    _lineplot(
        phi_so_vals, G_vals,
        xlabel=r"$\phi_{so}$",
        title=(rf"$G/G_0$ vs $\phi_{{so}}$  "
               rf"($E_{{conf}}^{{leads}}={E_CONF_LEADS}$ meV, $E_{{conf}}^U={E_CONF_U}$ meV)"),
        save_path=OUTPUT_DIR / "plot5_G_vs_phi_so.png",
    )


# ---------------------------------------------------------------------------
# Plot 6: G vs phi_so  — 1-D line
#         E_conf_leads = 6.0 meV, E_conf_U = 0.2 meV
# ---------------------------------------------------------------------------

def plot6(n_line: int = N_LINE) -> None:
    E_CONF_LEADS = 6.0    # meV
    E_CONF_U     = 0.2    # meV
    phi_so_vals  = np.linspace(0.5, 30.0, n_line)

    print(f"\n=== Plot 6: G vs phi_so  E_conf_leads={E_CONF_LEADS} meV, E_conf_U={E_CONF_U} meV ===")
    G_vals = np.array([
        compute_G(E_CONF_LEADS, E_CONF_U, phi_so)
        for phi_so in phi_so_vals
    ])

    np.savez_compressed(
        OUTPUT_DIR / "plot6_data.npz",
        phi_so=phi_so_vals, G=G_vals,
        E_conf_leads=np.array(E_CONF_LEADS), E_conf_U=np.array(E_CONF_U),
    )
    _lineplot(
        phi_so_vals, G_vals,
        xlabel=r"$\phi_{so}$",
        title=(rf"$G/G_0$ vs $\phi_{{so}}$  "
               rf"($E_{{conf}}^{{leads}}={E_CONF_LEADS}$ meV, $E_{{conf}}^U={E_CONF_U}$ meV)"),
        save_path=OUTPUT_DIR / "plot6_G_vs_phi_so.png",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute and save the six conductance plots for the JCE26 ring."
    )
    parser.add_argument(
        "--plots", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6],
        metavar="N",
        help="Which plots to run (e.g. --plots 2 4 5). Default: all six.",
    )
    parser.add_argument(
        "--n-grid", type=int, default=10,
        help="Number of points per axis for 2-D heatmaps (default 10).",
    )
    parser.add_argument(
        "--n-line", type=int, default=15,
        help="Number of points for 1-D line plots (default 15).",
    )
    args = parser.parse_args()

    N_G = args.n_grid
    N_L = args.n_line

    wall_t0 = time.perf_counter()
    dispatch = {
        1: lambda: plot1(N_G, N_G),
        2: lambda: plot2(N_L),
        3: lambda: plot3(N_G, N_G),
        4: lambda: plot4(N_L),
        5: lambda: plot5(N_L),
        6: lambda: plot6(N_L),
    }
    for n in sorted(set(args.plots)):
        if n in dispatch:
            dispatch[n]()
        else:
            print(f"Warning: plot {n} is not defined (valid: 1–6).")

    print(f"\nAll done in {time.perf_counter()-wall_t0:.1f} s.")
    print(f"Results saved in: {OUTPUT_DIR.resolve()}")
