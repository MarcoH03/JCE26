#!/usr/bin/env python3
"""Animate 3D conductance surface G/G₀ vs (U_leads, U_U) with φ_so as time.

This script creates an animated 3D surface plot where:
    - x-axis: lead QPC curvature U_leads [meV/nm²] (same for L and R)
    - y-axis: upper‑arm QPC curvature U_U [meV/nm²]
    - z-axis: Landauer conductance G/G₀
    - frames: different values of the Rashba phase φ_so

The animation reveals how the transport landscape evolves with spin‑orbit
coupling, helping to identify parameter combinations that reproduce
desired results.

Usage
-----
    python animate_conductance_vs_Uleads_UU_phi_so.py

Options
-------
    --n-leads INT     number of U_leads points (default 15)
    --n-uu INT        number of U_U points (default 15)
    --n-phi INT       number of φ_so frames (default 12)
    --U-leads-min F   min U_leads (default 0.0)
    --U-leads-max F   max U_leads (default 6.0)
    --U-U-min F       min U_U (default 0.0)
    --U-U-max F       max U_U (default 6.0)
    --phi-min F       min φ_so (default 0.0)
    --phi-max F       max φ_so (default 2*np.pi)
    --recompute       force recomputation even if cached data exists
    --save-anim FILE  save animation as GIF or MP4 (e.g., animation.gif)
    --fps INT         frames per second for saved animation (default 5)

Examples
--------
    # Quick test (coarse grid)
    python animate_conductance_vs_Uleads_UU_phi_so.py --n-leads 5 --n-uu 5 --n-phi 6

    # Publication quality (requires many simulations)
    python animate_conductance_vs_Uleads_UU_phi_so.py --n-leads 30 --n-uu 30 --n-phi 36 --save-anim animation.mp4 --fps 10

    # Use specific φ_so range to zoom in
    python animate_conductance_vs_Uleads_UU_phi_so.py --phi-min 1.0 --phi-max 2.5
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

import tools as t
from conductance import run_single_conductance
from conductance import run_cap_conductance


# ---------------------------------------------------------------------------
# Helper: parameter construction
# ---------------------------------------------------------------------------

def _alpha_from_phi_so(phi_so: float, p_base: t.PhysicsParams) -> float:
    """Convert φ_so to Rashba strength α [meV·nm]."""
    return phi_so * t.h_bar**2 / (p_base.m * p_base.R)


def make_params(
    U_leads: float,
    U_U: float,
    phi_so: float,
    p_base: t.PhysicsParams | None = None,
) -> t.PhysicsParams:
    """Return PhysicsParams for a single (U_leads, U_U, φ_so) point."""
    if p_base is None:
        p_base = t.default_params()
    return p_base.with_changes(
        alpha=_alpha_from_phi_so(phi_so, p_base),
        potential_model="legacy_localized_qpc",   # use the well‑behaved QPC model
        Ux_L=U_leads,
        Uy_L=U_leads,
        Ux_R=U_leads,
        Uy_R=U_leads,
        Ux_U=U_U,
        Uy_U=U_U,
        V0_L=0.0,
        V0_U=0.0,
        V0_R=0.0,
    )


# ---------------------------------------------------------------------------
# Simulation settings (matching six_conductance_plots.py)
# ---------------------------------------------------------------------------

FERMI_ENERGY_MEV = 4.19
TOTAL_TIME_PS = 30.5
PACKET_CENTER_FRACTION = 0.8
PACKET_WIDTH_NM = 250.0

BASE_PARAMS = t.default_params()


def compute_G(U_leads: float, U_U: float, phi_so: float, verbose: bool = False) -> float:
    """Return G/G0 for a single point."""
    p = make_params(U_leads, U_U, phi_so, BASE_PARAMS)
    result = run_cap_conductance(
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
# Data generation and caching
# ---------------------------------------------------------------------------

def generate_data(
    U_leads_vals: np.ndarray,
    U_U_vals: np.ndarray,
    phi_vals: np.ndarray,
    cache_file: Path,
    recompute: bool = False,
) -> np.ndarray:
    """Return a 3D array G[i_phi, i_UU, i_Ulead] with cached I/O.

    If cache_file exists and recompute is False, load and return.
    Otherwise compute all points and save.
    """
    if not recompute and cache_file.exists():
        print(f"Loading cached data from {cache_file} ...")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        return data

    n_phi = len(phi_vals)
    n_uu = len(U_U_vals)
    n_lead = len(U_leads_vals)
    G_grid = np.zeros((n_phi, n_uu, n_lead))

    total_points = n_phi * n_uu * n_lead
    current = 0
    print(f"Computing {total_points} conductance points ...")
    start_time = time.perf_counter()

    for i_phi, phi in enumerate(phi_vals):
        for j_uu, uu in enumerate(U_U_vals):
            for k_lead, ulead in enumerate(U_leads_vals):
                current += 1
                # Print progress every 5% or every 100 points
                if current % max(1, total_points // 20) == 0 or current <= 100:
                    pct = 100.0 * current / total_points
                    print(f"  [{current:5d}/{total_points:5d}] φ={phi:.3f}  U_U={uu:.4f}  U_leads={ulead:.4f}  ({pct:.1f}%)")
                G_grid[i_phi, j_uu, k_lead] = compute_G(ulead, uu, phi, verbose=False)

    elapsed = time.perf_counter() - start_time
    print(f"Computation finished in {elapsed:.1f} s ({elapsed/60:.1f} min).")

    # Save to cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(G_grid, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cached data saved to {cache_file}")

    return G_grid


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def animate_conductance_surface(
    U_leads_vals: np.ndarray,
    U_U_vals: np.ndarray,
    phi_vals: np.ndarray,
    G_grid: np.ndarray,
    save_path: Path | None = None,
    fps: int = 5,
) -> None:
    """Create and display (or save) an animation of the 3D surface."""
    n_phi = len(phi_vals)
    X, Y = np.meshgrid(U_leads_vals, U_U_vals)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Common axis limits
    z_min, z_max = 0.0, 1.0

    # Title placeholder
    title_text = ax.set_title(f"φ_so = {phi_vals[0]:.3f}")

    # Plot first frame
    Z0 = G_grid[0, :, :]  # shape (n_uu, n_lead)
    surf = ax.plot_surface(
        X, Y, Z0, cmap="viridis", vmin=z_min, vmax=z_max,
        linewidth=0, antialiased=True, alpha=0.92
    )
    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, pad=0.1)
    cbar.set_label(r"$G / G_0$", fontsize=11)

    ax.set_xlabel(r"$U_{leads}$ (meV/nm$^2$)", fontsize=10, labelpad=8)
    ax.set_ylabel(r"$U_U$ (meV/nm$^2$)", fontsize=10, labelpad=8)
    ax.set_zlabel(r"$G / G_0$", fontsize=10, labelpad=6)
    ax.set_zlim(z_min, z_max)

    def update(frame: int) -> list:
        """Update the surface data for a new φ_so value."""
        phi = phi_vals[frame]
        Z = G_grid[frame, :, :]   # (n_uu, n_lead)

        # Remove old surface and add new one
        nonlocal surf
        surf.remove()
        surf = ax.plot_surface(
            X, Y, Z, cmap="viridis", vmin=z_min, vmax=z_max,
            linewidth=0, antialiased=True, alpha=0.92
        )
        title_text.set_text(f"φ_so = {phi:.3f}")
        return [surf, title_text]

    anim = FuncAnimation(fig, update, frames=n_phi, interval=1000.0 / fps, blit=False)

    if save_path is not None:
        # Determine writer based on file extension
        if save_path.suffix.lower() == ".gif":
            writer = PillowWriter(fps=fps)
        else:
            # For mp4, requires ffmpeg; fallback to Pillow if not available
            try:
                from matplotlib.animation import FFMpegWriter
                writer = FFMpegWriter(fps=fps)
            except ImportError:
                print("FFmpeg not found, saving as GIF instead.")
                writer = PillowWriter(fps=fps)
                save_path = save_path.with_suffix(".gif")
        print(f"Saving animation to {save_path} ...")
        anim.save(save_path, writer=writer)
        print("Animation saved.")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate 3D conductance surface vs (U_leads, U_U) with φ_so as time."
    )
    parser.add_argument("--n-leads", type=int, default=15,
                        help="Number of U_leads points (default 15)")
    parser.add_argument("--n-uu", type=int, default=15,
                        help="Number of U_U points (default 15)")
    parser.add_argument("--n-phi", type=int, default=12,
                        help="Number of φ_so frames (default 12)")

    parser.add_argument("--U-leads-min", type=float, default=0.0,
                        help="Minimum U_leads value (meV/nm², default 0.0)")
    parser.add_argument("--U-leads-max", type=float, default=6.0,
                        help="Maximum U_leads value (meV/nm², default 6.0)")
    parser.add_argument("--U-U-min", type=float, default=0.0,
                        help="Minimum U_U value (meV/nm², default 0.0)")
    parser.add_argument("--U-U-max", type=float, default=6.0,
                        help="Maximum U_U value (meV/nm², default 6.0)")
    parser.add_argument("--phi-min", type=float, default=0.0,
                        help="Minimum φ_so (default 0.0)")
    parser.add_argument("--phi-max", type=float, default=2 * np.pi,
                        help="Maximum φ_so (default 2π)")

    parser.add_argument("--recompute", action="store_true",
                        help="Recompute all points even if cached data exists")
    parser.add_argument("--save-anim", type=str, default=None,
                        help="Save animation to file (e.g., animation.gif or video.mp4)")
    parser.add_argument("--fps", type=int, default=5,
                        help="Frames per second for saved animation (default 5)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Build parameter vectors
    U_leads_vals = np.linspace(args.U_leads_min, args.U_leads_max, args.n_leads)
    U_U_vals     = np.linspace(args.U_U_min, args.U_U_max, args.n_uu)
    phi_vals     = np.linspace(args.phi_min, args.phi_max, args.n_phi)

    print("=== Animation parameters ===")
    print(f"U_leads:  {U_leads_vals[0]:.3f} ... {U_leads_vals[-1]:.3f}  ({args.n_leads} pts)")
    print(f"U_U:      {U_U_vals[0]:.3f} ... {U_U_vals[-1]:.3f}  ({args.n_uu} pts)")
    print(f"φ_so:     {phi_vals[0]:.3f} ... {phi_vals[-1]:.3f}  ({args.n_phi} frames)")
    print(f"Total simulations: {args.n_leads * args.n_uu * args.n_phi}")

    # Cache file name includes grid parameters to avoid collisions
    cache_name = f"conductance_3d_Uleads_{args.n_leads}_UU_{args.n_uu}_phi_{args.n_phi}.pkl"
    cache_path = Path(__file__).parent / "conductance_plots" / "anim_cache" / cache_name

    # Compute or load data
    G_grid = generate_data(U_leads_vals, U_U_vals, phi_vals, cache_path, args.recompute)

    import os    # At the end of your script, after all plots are done
    os.system('afplay /System/Library/Sounds/Glass.aiff')
    # Animate
    save_path = Path(args.save_anim) if args.save_anim else None
    animate_conductance_surface(
        U_leads_vals, U_U_vals, phi_vals, G_grid,
        save_path=save_path, fps=args.fps
    )


if __name__ == "__main__":
    main()
