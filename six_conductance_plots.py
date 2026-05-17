"""Six conductance plots for the JCE26 quantum ring.

Swept parameters
----------------
- ``U_leads``  [meV/nm²]  — curvature strength Ux = Uy for both lead QPCs (L = R always).
- ``U_U``      [meV/nm²]  — curvature strength Ux_U = Uy_U for the upper-arm QPC.
- ``phi_so``               — total Rashba phase k_so·R along one arm;
                             converted to alpha internally via  α = phi_so · ℏ² / (m · R).

The lower arm (D) never carries a QPC; its curvature is hardcoded to zero inside tools.py.

Plots produced
--------------
1. G/G₀ vs (U_leads, U_U)      [3-D surface]  phi_so = 24.05
2. G/G₀ vs U_leads             [1-D line]     U_U = 0.1 meV/nm², phi_so = 24.05
3. G/G₀ vs (U_leads, phi_so)   [3-D surface]  U_U = 2.78 meV/nm²
4. G/G₀ vs U_leads             [1-D line]     U_U = 2.78 meV/nm², phi_so = 1
5. G/G₀ vs phi_so              [1-D line]     U_leads = 3.08 meV/nm², U_U = 0.2 meV/nm²
6. G/G₀ vs phi_so              [1-D line]     U_leads = 6.0  meV/nm², U_U = 0.2 meV/nm²

How to change sweep ranges and step size
-----------------------------------------
Every plot function has a clearly marked block that looks like:

    # ---- sweep range: edit start, stop, n_points here ----
    U_L_vals = np.linspace(start, stop, n_points)
    # -------------------------------------------------------

- ``start``   : first value of the parameter
- ``stop``    : last value of the parameter
- ``n_points``: total number of evenly-spaced samples (inclusive of both ends)
                step between consecutive values = (stop - start) / (n_points - 1)

You can also pass --n-grid and --n-line from the command line to override the
default point counts for all plots at once (useful for quick tests):

    python six_conductance_plots.py --n-grid 4 --n-line 6    # coarse, fast
    python six_conductance_plots.py --n-grid 25 --n-line 30  # fine, slow

All raw data is saved as .npz files so you can replot without re-running.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 — registers 3-D projection

import tools as t
from conductance import run_single_conductance


# ---------------------------------------------------------------------------
# phi_so <-> alpha  (U values are passed directly — no conversion needed)
# ---------------------------------------------------------------------------

def _alpha_from_phi_so(phi_so: float, p: t.PhysicsParams) -> float:
    """α [meV·nm] from phi_so = m α R / ℏ²."""
    return phi_so * t.h_bar ** 2 / (p.m * p.R)


def _phi_so_from_alpha(alpha: float, p: t.PhysicsParams) -> float:
    return p.m * alpha * p.R / t.h_bar ** 2


# ---------------------------------------------------------------------------
# Base parameter set
# ---------------------------------------------------------------------------

BASE_PARAMS = t.PhysicsParams(
    potential_model="legacy_localized_qpc",
    gaussian_qpc_heights_mev={"L": 0.0, "U": 0.0, "D": 0.0, "R": 0.0},
)

_a24 = _alpha_from_phi_so(24.05, BASE_PARAMS)
print(f"[check] phi_so=24.05 -> alpha={_a24:.4f} meV*nm "
      f"-> phi_so back={_phi_so_from_alpha(_a24, BASE_PARAMS):.4f}")
_a1 = _alpha_from_phi_so(1.0, BASE_PARAMS)
print(f"[check] phi_so=1     -> alpha={_a1:.4f} meV*nm")


# ---------------------------------------------------------------------------
# Parameter constructor  (U values passed directly in meV/nm²)
# ---------------------------------------------------------------------------

def make_params(
    U_leads: float,
    U_U: float,
    phi_so: float,
    p_base: t.PhysicsParams | None = None,
) -> t.PhysicsParams:
    """Return a PhysicsParams for one (U_leads, U_U, phi_so) operating point.

    Parameters
    ----------
    U_leads : float
        QPC curvature [meV/nm²] applied identically to both lead QPCs:
        Ux_L = Uy_L = Ux_R = Uy_R = U_leads.
    U_U : float
        QPC curvature [meV/nm²] for the upper ring arm: Ux_U = Uy_U = U_U.
    phi_so : float
        Total Rashba phase along one arm (dimensionless). Converted to alpha.
    """
    if p_base is None:
        p_base = BASE_PARAMS
    return p_base.with_changes(
        alpha=_alpha_from_phi_so(phi_so, p_base),
        Ux_L=U_leads,
        Uy_L=U_leads,
        Ux_R=U_leads,    # right lead always mirrors left lead
        Uy_R=U_leads,
        Ux_U=U_U,
        Uy_U=U_U,
        V0_L=0.0,
        V0_U=0.0,
        V0_R=0.0,
    )


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
# Core runner
# ---------------------------------------------------------------------------

def compute_G(U_leads: float, U_U: float, phi_so: float, verbose: bool = False) -> float:
    """Return G/G0 for one (U_leads, U_U, phi_so) point."""
    p = make_params(U_leads, U_U, phi_so)
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
# Plotting helpers
# ---------------------------------------------------------------------------

CMAP = "viridis"


def _surface3d(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    G_grid: np.ndarray,      # shape (len(y_vals), len(x_vals))
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: Path,
) -> None:
    """3-D surface plot of G/G0. Colour is mapped to the G axis using viridis."""
    X, Y = np.meshgrid(x_vals, y_vals)

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, G_grid,
        cmap=CMAP,
        vmin=0.0, vmax=1.0,
        linewidth=0,
        antialiased=True,
        alpha=0.92,
    )

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


def _lineplot(
    x_vals: np.ndarray,
    G_vals: np.ndarray,
    xlabel: str,
    title: str,
    save_path: Path,
) -> None:
    """1-D conductance line plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_vals, G_vals, "o-", color="tab:blue", linewidth=1.8, markersize=5)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(r"$G / G_0$", fontsize=12)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Default resolution (overridable via CLI --n-grid / --n-line)
# ---------------------------------------------------------------------------

N_GRID = 20   # points per axis for 3-D surface plots  (total runs = N_GRID²)
N_LINE = 50   # points for 1-D line plots


# ---------------------------------------------------------------------------
# Plot 1 — 3-D surface: G vs (U_leads, U_U),  phi_so = 24.05
# ---------------------------------------------------------------------------

def plot1(n_leads: int = N_GRID, n_U: int = N_GRID) -> None:
    PHI_SO = 24.05

    # ---- sweep range: edit start, stop, n_points here ----
    U_L_vals = np.linspace(0.0, 3.0, n_leads)   # [meV/nm²]
    U_U_vals = np.linspace(0.0, 3.0, n_U)        # [meV/nm²]
    # -------------------------------------------------------

    _print_sweep_info("Plot 1", {"U_leads": U_L_vals, "U_U": U_U_vals},
                      fixed={"phi_so": PHI_SO})

    G_grid = np.zeros((n_U, n_leads))
    total  = n_leads * n_U
    for j, U_U in enumerate(U_U_vals):
        for i, U_L in enumerate(U_L_vals):
            print(f"  [{j*n_leads+i+1}/{total}]  U_leads={U_L:.4f}  U_U={U_U:.4f}")
            G_grid[j, i] = compute_G(U_L, U_U, PHI_SO)

    np.savez_compressed(
        OUTPUT_DIR / "plot1_data.npz",
        U_leads=U_L_vals, U_U=U_U_vals, G_grid=G_grid, phi_so=np.array(PHI_SO),
    )
    _surface3d(
        U_L_vals, U_U_vals, G_grid,
        xlabel=r"$U_{leads}$ (meV/nm$^2$)",
        ylabel=r"$U_U$ (meV/nm$^2$)",
        title=rf"$G/G_0$ vs QPC curvatures  ($\phi_{{so}} = {PHI_SO}$)",
        save_path=OUTPUT_DIR / "plot1_G_vs_Uleads_UU.png",
    )


# ---------------------------------------------------------------------------
# Plot 2 — 1-D line: G vs U_leads,  U_U = 0.1, phi_so = 24.05
# ---------------------------------------------------------------------------

def plot2(n_line: int = N_LINE) -> None:
    PHI_SO  = 24.05
    U_U_FIX = 0.1    # meV/nm²

    # ---- sweep range ----
    U_L_vals = np.linspace(1, 4.0, n_line)   # [meV/nm²]
    # ---------------------

    _print_sweep_info("Plot 2", {"U_leads": U_L_vals},
                      fixed={"U_U": U_U_FIX, "phi_so": PHI_SO})

    G_vals = np.array([compute_G(U_L, U_U_FIX, PHI_SO) for U_L in U_L_vals])

    np.savez_compressed(
        OUTPUT_DIR / "plot2_data.npz",
        U_leads=U_L_vals, G=G_vals,
        U_U=np.array(U_U_FIX), phi_so=np.array(PHI_SO),
    )
    _lineplot(
        U_L_vals, G_vals,
        xlabel=r"$U_{leads}$ (meV/nm$^2$)",
        title=(rf"$G/G_0$ vs lead QPC curvature  "
               rf"($U_U = {U_U_FIX}\ \mathrm{{meV/nm^2}},\ \phi_{{so}} = {PHI_SO}$)"),
        save_path=OUTPUT_DIR / "plot2_G_vs_Uleads.png",
    )


# ---------------------------------------------------------------------------
# Plot 3 — 3-D surface: G vs (U_leads, phi_so),  U_U = 2.78
# ---------------------------------------------------------------------------

def plot3(n_leads: int = N_GRID, n_phi: int = N_GRID) -> None:
    U_U_FIX = 2.78   # meV/nm²

    # ---- sweep range ----
    U_L_vals    = np.linspace(1, 6.0,  n_leads)
    phi_so_vals = np.linspace(0.0, 2.0, n_phi)
    # ---------------------

    _print_sweep_info("Plot 3", {"U_leads": U_L_vals, "phi_so": phi_so_vals},
                      fixed={"U_U": U_U_FIX})

    G_grid = np.zeros((n_phi, n_leads))
    total  = n_leads * n_phi
    for j, phi_so in enumerate(phi_so_vals):
        for i, U_L in enumerate(U_L_vals):
            print(f"  [{j*n_leads+i+1}/{total}]  U_leads={U_L:.4f}  phi_so={phi_so:.3f}")
            G_grid[j, i] = compute_G(U_L, U_U_FIX, phi_so)

    np.savez_compressed(
        OUTPUT_DIR / "plot3_data.npz",
        U_leads=U_L_vals, phi_so=phi_so_vals,
        G_grid=G_grid, U_U=np.array(U_U_FIX),
    )
    _surface3d(
        U_L_vals, phi_so_vals, G_grid,
        xlabel=r"$U_{leads}$ (meV/nm$^2$)",
        ylabel=r"$\phi_{so}$",
        title=rf"$G/G_0$ vs lead curvature and $\phi_{{so}}$  ($U_U = {U_U_FIX}\ \mathrm{{meV/nm^2}}$)",
        save_path=OUTPUT_DIR / "plot3_G_vs_Uleads_phi_so.png",
    )


# ---------------------------------------------------------------------------
# Plot 4 — 1-D line: G vs U_leads,  U_U = 2.78, phi_so = 1
# ---------------------------------------------------------------------------

def plot4(n_line: int = N_LINE) -> None:
    PHI_SO  = 1.0
    U_U_FIX = 2.78   # meV/nm²

    # ---- sweep range ----
    U_L_vals = np.linspace(0.0, 6.0, n_line)
    # ---------------------

    _print_sweep_info("Plot 4", {"U_leads": U_L_vals},
                      fixed={"U_U": U_U_FIX, "phi_so": PHI_SO})

    G_vals = np.array([compute_G(U_L, U_U_FIX, PHI_SO) for U_L in U_L_vals])

    np.savez_compressed(
        OUTPUT_DIR / "plot4_data.npz",
        U_leads=U_L_vals, G=G_vals,
        U_U=np.array(U_U_FIX), phi_so=np.array(PHI_SO),
    )
    _lineplot(
        U_L_vals, G_vals,
        xlabel=r"$U_{leads}$ (meV/nm$^2$)",
        title=(rf"$G/G_0$ vs lead QPC curvature  "
               rf"($U_U = {U_U_FIX}\ \mathrm{{meV/nm^2}},\ \phi_{{so}} = {PHI_SO}$)"),
        save_path=OUTPUT_DIR / "plot4_G_vs_Uleads.png",
    )


# ---------------------------------------------------------------------------
# Plot 5 — 1-D line: G vs phi_so,  U_leads = 3.08, U_U = 0.2
# ---------------------------------------------------------------------------

def plot5(n_line: int = N_LINE) -> None:
    U_L_FIX = 3.08   # meV/nm²
    U_U_FIX = 0.2    # meV/nm²

    # ---- sweep range ----
    phi_so_vals = np.linspace(-6.0, 6.0, n_line)
    # ---------------------

    _print_sweep_info("Plot 5", {"phi_so": phi_so_vals},
                      fixed={"U_leads": U_L_FIX, "U_U": U_U_FIX})

    G_vals = np.array([compute_G(U_L_FIX, U_U_FIX, phi_so) for phi_so in phi_so_vals])

    np.savez_compressed(
        OUTPUT_DIR / "plot5_data.npz",
        phi_so=phi_so_vals, G=G_vals,
        U_leads=np.array(U_L_FIX), U_U=np.array(U_U_FIX),
    )
    _lineplot(
        phi_so_vals, G_vals,
        xlabel=r"$\phi_{so}$",
        title=(rf"$G/G_0$ vs $\phi_{{so}}$  "
               rf"($U_{{leads}} = {U_L_FIX}\ \mathrm{{meV/nm^2}},\ "
               rf"U_U = {U_U_FIX}\ \mathrm{{meV/nm^2}}$)"),
        save_path=OUTPUT_DIR / "plot5_G_vs_phi_so.png",
    )


# ---------------------------------------------------------------------------
# Plot 6 — 1-D line: G vs phi_so,  U_leads = 6.0, U_U = 0.2
# ---------------------------------------------------------------------------

def plot6(n_line: int = N_LINE) -> None:
    U_L_FIX = 6.0    # meV/nm²
    U_U_FIX = 0.2    # meV/nm²

    # ---- sweep range ----
    phi_so_vals = np.linspace(-6.0, 6.0, n_line)
    # ---------------------

    _print_sweep_info("Plot 6", {"phi_so": phi_so_vals},
                      fixed={"U_leads": U_L_FIX, "U_U": U_U_FIX})

    G_vals = np.array([compute_G(U_L_FIX, U_U_FIX, phi_so) for phi_so in phi_so_vals])

    np.savez_compressed(
        OUTPUT_DIR / "plot6_data.npz",
        phi_so=phi_so_vals, G=G_vals,
        U_leads=np.array(U_L_FIX), U_U=np.array(U_U_FIX),
    )
    _lineplot(
        phi_so_vals, G_vals,
        xlabel=r"$\phi_{so}$",
        title=(rf"$G/G_0$ vs $\phi_{{so}}$  "
               rf"($U_{{leads}} = {U_L_FIX}\ \mathrm{{meV/nm^2}},\ "
               rf"U_U = {U_U_FIX}\ \mathrm{{meV/nm^2}}$)"),
        save_path=OUTPUT_DIR / "plot6_G_vs_phi_so.png",
    )


# ---------------------------------------------------------------------------
# Utility: print sweep info before running
# ---------------------------------------------------------------------------

def _print_sweep_info(
    plot_name: str,
    swept: dict[str, np.ndarray],
    fixed: dict[str, float],
) -> None:
    print(f"\n=== {plot_name} ===")
    for name, vals in swept.items():
        n  = len(vals)
        step = (vals[-1] - vals[0]) / (n - 1) if n > 1 else 0.0
        print(f"  sweep  {name}: {vals[0]:.4g} ... {vals[-1]:.4g}  "
              f"({n} pts, step={step:.4g})")
    for name, val in fixed.items():
        print(f"  fixed  {name} = {val}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute and save the six conductance plots for the JCE26 ring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
To change the range or step of a specific plot's sweep, open this file and edit
the np.linspace call inside that plot's function (plot1 … plot6):

    U_L_vals = np.linspace(start, stop, n_points)
    step = (stop - start) / (n_points - 1)

Examples
--------
  python six_conductance_plots.py                         # all 6, default res
  python six_conductance_plots.py --plots 2 4 5 6         # only 1-D lines
  python six_conductance_plots.py --plots 1 3             # only 3-D surfaces
  python six_conductance_plots.py --n-grid 4 --n-line 5   # coarse (for testing)
  python six_conductance_plots.py --n-grid 25 --n-line 30 # fine (for publication)
        """,
    )
    parser.add_argument(
        "--plots", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6], metavar="N",
        help="Which plots to run (default: all six).",
    )
    parser.add_argument(
        "--n-grid", type=int, default=N_GRID,
        help=f"Points per axis for 3-D surface plots (default {N_GRID}). "
             "Total simulations = n_grid².",
    )
    parser.add_argument(
        "--n-line", type=int, default=N_LINE,
        help=f"Points for 1-D line plots (default {N_LINE}).",
    )
    args = parser.parse_args()

    wall_t0 = time.perf_counter()
    dispatch = {
        1: lambda: plot1(args.n_grid, args.n_grid),
        2: lambda: plot2(args.n_line),
        3: lambda: plot3(args.n_grid, args.n_grid),
        4: lambda: plot4(args.n_line),
        5: lambda: plot5(args.n_line),
        6: lambda: plot6(args.n_line),
    }
    for n in sorted(set(args.plots)):
        if n in dispatch:
            dispatch[n]()
        else:
            print(f"Warning: plot {n} is not in 1–6, skipping.")

    elapsed = time.perf_counter() - wall_t0
    print(f"\nAll done in {elapsed:.1f} s  ({elapsed/60:.1f} min).")
    print(f"Results in: {OUTPUT_DIR.resolve()}")
    # At the end of your script, after all plots are done
    os.system('afplay /System/Library/Sounds/Glass.aiff')
