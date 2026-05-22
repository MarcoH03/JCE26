#!/usr/bin/env python3
"""Aharonov-Bohm and Aharonov-Casher interference using the JCE26 Crank-Nicolson simulator.

This script demonstrates conductance oscillations in a clean ring (V=0) as a function of
magnetic flux (AB) and Rashba strength (AC). It reuses the CAP-based transmission
calculation from conductance.py.
"""

import numpy as np
import matplotlib.pyplot as plt

import tools as t
from conductance import run_cap_conductance, sweep_conductance, plot_sweep_conductance, save_sweep_results_npz


def make_clean_ring_params() -> t.PhysicsParams:
    """Return a PhysicsParams with zero potential everywhere."""
    params = t.default_params()
    # Turn off all QPC potentials
    params = params.with_changes(
        potential_model="none",
        V0_L=0.0, Ux_L=0.0, Uy_L=0.0,
        V0_U=0.0, Ux_U=0.0, Uy_U=0.0,
        V0_R=0.0, Ux_R=0.0, Uy_R=0.0,
        gaussian_qpc_heights_mev={"L":0.0, "U":0.0, "D":0.0, "R":0.0}
    )
    return params


def sweep_ab_conductance(base_params: t.PhysicsParams, flux_values: np.ndarray,
                         total_time_ps: float = 30.0, **cap_kwargs):
    """Sweep magnetic flux Phi (in units of Phi0) and return conductance results."""
    results = []
    for phi in flux_values:
        p = base_params.with_changes(Phi=float(phi))
        cr = run_cap_conductance(p, fermi_energy_mev=4.19,
                                 total_time_ps=total_time_ps,
                                 **cap_kwargs)
        results.append(cr)
    return results


def sweep_ac_conductance(base_params: t.PhysicsParams, alpha_values: np.ndarray,
                         total_time_ps: float = 30.0, **cap_kwargs):
    """Sweep Rashba strength alpha (meV·nm) and return conductance results."""
    results = []
    for alpha in alpha_values:
        p = base_params.with_changes(alpha=float(alpha))
        cr = run_cap_conductance(p, fermi_energy_mev=4.19,
                                 total_time_ps=total_time_ps,
                                 **cap_kwargs)
        results.append(cr)
    return results


def plot_ab_oscillations(flux_values, conductances, save_path=None):
    """Plot G/G0 vs magnetic flux."""
    plt.figure(figsize=(8,5))
    plt.plot(flux_values, conductances, 'o-', color='navy', linewidth=1.5, markersize=4)
    plt.xlabel(r'Magnetic flux $\Phi / \Phi_0$')
    plt.ylabel(r'Conductance $G/G_0$')
    plt.title('Aharonov–Bohm oscillations (clean ring, $V=0$)')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    if save_path:
        plt.savefig(save_path, dpi=180)
    plt.show()


def plot_ac_oscillations(alpha_values, conductances, save_path=None):
    """Plot G/G0 vs Rashba strength."""
    plt.figure(figsize=(8,5))
    plt.plot(alpha_values, conductances, 'o-', color='darkred', linewidth=1.5, markersize=4)
    plt.xlabel(r'Rashba strength $\alpha$ (meV·nm)')
    plt.ylabel(r'Conductance $G/G_0$')
    plt.title('Aharonov–Casher oscillations (clean ring, $V=0$, $\Phi=0$)')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    if save_path:
        plt.savefig(save_path, dpi=180)
    plt.show()
    
def alpha_to_ac_phase(alpha, p):
    """Convierte α (meV·nm) a fase AC en radianes."""
    m = p.m  # masa efectiva en meV·ps²/nm²
    Lring = p.L_ring  # nm
    hbar = 0.658212  # meV·ps
    return (2.0 * m * Lring / hbar**2) * alpha





if __name__ == "__main__":
    # Use CAP with moderate strength; ensure total time is long enough for absorption
    cap_params = {
        "cap_strength": 2.0,      # meV
        "cap_fraction": 0.25,
        "cap_order": 3,
        "keep_time_series": False,
        "verbose": True
    }

    # -------------------- AB sweep --------------------
    print("\n=== Aharonov–Bohm sweep ===")
    base_clean = make_clean_ring_params()
    flux_sweep = np.linspace(0, 2.0, 21)   # 0 to 2 flux quanta
    # ab_results = sweep_ab_conductance(base_clean, flux_sweep, total_time_ps=30.0, **cap_params)
    # ab_T = [res.T for res in ab_results]

    # plot_ab_oscillations(flux_sweep, ab_T, save_path="AB_oscillations.png")
    # # Optionally save data
    # np.savez_compressed("AB_sweep.npz", flux=flux_sweep, T=ab_T)

    # -------------------- AC sweep --------------------
    print("\n=== Aharonov–Casher sweep ===")
    # Reset flux to zero
    base_clean_flux0 = base_clean.with_changes(Phi=0.0)
    alpha_sweep = np.linspace(0, 60, 31)   # 0 to 60 meV·nm
    ac_results = sweep_ac_conductance(base_clean_flux0, alpha_sweep, total_time_ps=30.0, **cap_params)
    ac_T = [res.T for res in ac_results]
    
    # Crear un objeto PhysicsParams base para obtener m y L_ring
    p_base = make_clean_ring_params()

    # Calcular fase AC para cada alpha
    ac_phase_vals = alpha_to_ac_phase(alpha_sweep, p_base)

    plot_ac_oscillations(ac_phase_vals, ac_T, save_path="AC_oscillations.png")
    np.savez_compressed("AC_sweep.npz", alpha=alpha_sweep, T=ac_T)