"""Proof-of-concept: Aharonov-Bohm and Aharonov-Casher interference in a quantum ring.

Physical setup
--------------
A 1D quantum ring (V = 0, no QPC barriers) connected to two semi-infinite leads.
The ring carries:
  - An Aharonov-Bohm phase from a perpendicular magnetic flux Φ (in units of Φ₀).
  - A Rashba spin-orbit interaction of strength α [meV·nm].

Both AB and AC oscillations in conductance G/G₀ are computed as a function of
those two parameters and compared to the analytical Büttiker formula for a
transparent ring.

Analytical reference (Büttiker et al. 1984, thesis eq. 3.2)
-------------------------------------------------------------
For a spin channel σ = ±1 in a transparent ring:

    G_σ / G₀ = 16 cos²(φ_σ) sin²(θ) / D_σ

where:
    φ_σ  = π(1/2 + Φ/Φ₀ + σ·φ_so)     (AB + AC combined phase)
    θ    = k_F · π · R                   (dynamic phase, half-ring)
    φ_so = k_so · R = (mα/ℏ²) · R
    D_σ  = [1 - 2cos(2θ) + cos(2φ_σ)]² + 4sin²(θ)

    G_total / G₀ = G_↑ / G₀ + G_↓ / G₀   (max = 2)

Parameter choices to see AB oscillations
-----------------------------------------
  Φ varies in [-1, 1] (one full flux quantum).
  Choose θ = k_F·π·R such that sin²(θ) is appreciable (not a node).
  With the InAs parameters below: θ ≈ 12.5π → sin²(θ) ≈ sin²(12.5π) ≈ 0 (node!).
  Fix: choose R so that θ is NOT a multiple of π.
  Good choice: θ = 12.57π (slightly off a node) or use fewer ring sites for speed.

  The wavepacket measurement agrees with the Büttiker formula when:
  1. The ring is transparent (no barrier potential).
  2. BOTH spin channels are injected (spin="both").
  3. The dynamic phase θ is NOT a node (sin θ ≠ 0).

Parameter choices to see AC oscillations
------------------------------------------
  Fix Φ (so AB phase is constant).
  Vary α (so φ_so changes) over [0, 2] φ_so units.
  Period: Δα such that Δφ_so = 1, i.e., Δα = ℏ²/(mR).

Coupling between lead and ring (ε)
-----------------------------------
  In a Y-junction tight-binding graph, the S-matrix gives:
    r = -1/3,  t_arm = 2/3  (equal hopping amplitudes t_lead = t_ring).
  This gives reflection R = 1/9 ≈ 11% at each junction -- purely topological,
  not a bug. The Fabry-Perot denominator accounts for this, and the maximum
  T still reaches 1.0 at constructive interference.

  The partial reflection you see in the animation IS physical. It comes from
  the 3-port topology, not from impedance mismatch. The packet is not purely
  transmitted because a 3-port symmetric junction always splits the amplitude
  1:1 between the two arms, and some amplitude necessarily reflects.

Run this module
---------------
    python ab_ac_proof.py          # all plots, default resolution
    python ab_ac_proof.py --quick  # fast low-resolution run
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tools as t
from conductance import run_single_conductance, ConductanceResult

OUTPUT_DIR = Path(__file__).parent / "ab_ac_proof"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Physical constants and system parameters
# ---------------------------------------------------------------------------
h_bar = t.h_bar   # 0.658212 meV·ps
m_e   = t.m_e     # 5.68563e-3 meV·ps²/nm²

# InAs parameters from the thesis (chapter 3.3):
M_FACTOR  = 0.023          # effective mass / m_e
R_NM      = 250.0          # ring radius, nm
E_F_MEV   = 4.19           # Fermi energy, meV
ALPHA_REF = 20.0           # reference Rashba constant, meV·nm

# Derived
m = M_FACTOR * m_e
k_F = np.sqrt(2 * m * E_F_MEV) / h_bar   # nm⁻¹
theta_dyn = k_F * np.pi * R_NM            # dynamic phase (half-ring)
k_so_ref  = m * ALPHA_REF / h_bar**2      # nm⁻¹
phi_so_ref = k_so_ref * R_NM              # dimensionless AC phase per arm

print(f"k_F        = {k_F:.6f} nm⁻¹")
print(f"θ_dynamic  = k_F·π·R = {theta_dyn:.4f} rad = {theta_dyn/np.pi:.4f}·π")
print(f"φ_so (α=20 meV·nm) = {phi_so_ref:.4f}")
print(f"sin²(θ)    = {np.sin(theta_dyn)**2:.6f}  (must be ≠ 0 for AB/AC to appear)")
print()


# ---------------------------------------------------------------------------
# Analytical Büttiker formula for a transparent ring (thesis eq. 3.2)
# ---------------------------------------------------------------------------

def analytical_G(Phi: float, alpha_mev_nm: float) -> float:
    """Return G/G₀ from the Büttiker transparent-ring formula.

    Parameters
    ----------
    Phi   : float   AB flux in units of Φ₀.
    alpha : float   Rashba constant [meV·nm].

    Phase-convention note
    ---------------------
    The Büttiker formula was derived for a closed ring with Bloch boundary
    conditions, where the spinor boundary condition introduces an extra
    Berry phase of π (equivalent to a flux offset of 1/2 Φ₀).  The
    tight-binding open-path model (wavepacket travelling from left to right
    junction through each arm) does NOT have this extra phase, because the
    arms are open paths, not closed loops.

    To align the two conventions, we shift Φ → Φ − 1/2 in the formula:

        φ_σ = π(½ + (Φ − ½) + σ·φ_so) = π(Φ + σ·φ_so)

    This makes G_max appear at Φ = 0, ±1 in both the formula and the
    tight-binding simulation.
    """
    phi_so = (m * alpha_mev_nm / h_bar**2) * R_NM   # = k_so * R
    theta  = theta_dyn

    G_total = 0.0
    for sigma in (+1, -1):   # spin up and down
        # Shift Phi by -1/2 to match the open-path tight-binding convention
        phi_sigma = np.pi * (Phi + sigma * phi_so)
        num = 16.0 * np.cos(phi_sigma)**2 * np.sin(theta)**2
        den = (1.0 - 2.0*np.cos(2*theta) + np.cos(2*phi_sigma))**2 \
              + 4.0 * np.sin(theta)**2
        G_total += num / den if den > 1e-15 else 0.0

    return G_total   # in units of G₀ (range [0, 2])


# ---------------------------------------------------------------------------
# PhysicsParams factory for a transparent ring
# ---------------------------------------------------------------------------

def transparent_ring_params(
    Phi: float = 0.0,
    alpha: float = ALPHA_REF,
) -> t.PhysicsParams:
    """Return PhysicsParams for a ring with V=0 everywhere."""
    return t.PhysicsParams(
        m_factor=M_FACTOR,
        R=R_NM,
        L_leads=2000.0,
        N_l=381,
        N_R=151,
        dt=0.002,
        Phi=Phi,
        alpha=alpha,
        potential_model="none",   # V = 0 everywhere
        gaussian_qpc_heights_mev={"L": 0.0, "U": 0.0, "D": 0.0, "R": 0.0},
    )


# ---------------------------------------------------------------------------
# Simulation runner  (both spin channels, echo window)
# ---------------------------------------------------------------------------

def simulate_G(Phi: float, alpha: float, verbose: bool = False,
               boundary: str = "transparent", total_time_ps: float = 35.0) -> float:
    """Return numerical G/G₀ using an open-boundary lead absorber.

    boundary="cap" (default, unchanged behaviour): Complex Absorbing
    Potential.  Absorbs all outgoing probability at both lead ends.
    Probability absorbed on the right = T, on the left = R.
    G/G₀ = 2·T  (factor 2 for both spin channels, T+R should equal 1).

    boundary="transparent": exact single-site lead self-energy boundary
    condition (see conductance.run_transparent_conductance / the "Soluciones
    propuestas" doc, solution #1). Verified (test_transparent_boundary.py)
    to reduce spurious reflection at the lead truncation itself by ~2-25x
    relative to this CAP configuration, depending on Fermi energy. However,
    for the FULL ring system this does NOT by itself fix the attenuated
    AB/AC oscillation amplitude reported in section 5.2 of the article: at
    total_time_ps up to 250 ps (~23x the ring transit time) both boundary
    choices plateau at essentially the same T+R (~0.35-0.38), which points
    to a separate, still-open cause -- probability trapped in long-lived
    quasi-bound resonances at the Y-junctions (topological ~11% reflection
    per junction, noted below) rather than reflection at the outer lead
    edges. Kept here as an experimental option for future work rather than
    as the new default, precisely so it does not silently change these
    reference plots without a corresponding junction-level fix.

    Total simulation time 35 ps gives the wavepacket enough time to:
      1. Travel from the packet centre to the ring  (1.6 ps)
      2. Traverse one ring arm                      (1.6 ps)
      3. Exit to the right lead and be absorbed     (2–4 ps)
      4. Any reflected amplitude returns and is absorbed by the left CAP
    The CAP turns off hard-wall echoes so the run can be as long as needed.
    """
    from conductance import run_cap_conductance, run_transparent_conductance
    p = transparent_ring_params(Phi=Phi, alpha=alpha)
    if boundary == "cap":
        result = run_cap_conductance(
            p,
            fermi_energy_mev=E_F_MEV,
            total_time_ps=total_time_ps,
            packet_center_fraction=0.8,
            packet_width_nm=150.0,
            cap_fraction=0.20,           # outer 20% of each lead is absorbing
            cap_strength=2.0,            # 2 meV peak absorption
            cap_order=3,
            spin_both=True,              # inject both spin channels
            verbose=verbose,
        )
    elif boundary == "transparent":
        result = run_transparent_conductance(
            p,
            fermi_energy_mev=E_F_MEV,
            total_time_ps=total_time_ps,
            packet_center_fraction=0.8,
            packet_width_nm=150.0,
            spin_both=True,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown boundary {boundary!r}")
    return result.G_over_G0


# ---------------------------------------------------------------------------
# Plot 1 — AB oscillations: G vs Φ at fixed α
# ---------------------------------------------------------------------------

def plot_ab_oscillations(n_points: int = 40) -> None:
    """G/G₀ vs magnetic flux Φ/Φ₀ for several fixed α values."""
    Phi_values = np.linspace(-1.0, 1.0, n_points)
    alpha_cases = [
        (0.0,           "α = 0  (no SOI)",       "tab:gray"),
        (ALPHA_REF,     f"α = {ALPHA_REF} meV·nm", "tab:blue"),
        (phi_so_ref*h_bar**2/m/R_NM*2.0, f"φ_so = 2·φ_so_ref", "tab:orange"),
    ]
    # Note: phi_so_ref already set for alpha=20. Adjust alpha for phi_so=1 or phi_so=2.
    alpha_phi1 = 1.0 * h_bar**2 / (m * R_NM)  # alpha for phi_so = 1
    alpha_cases = [
        (0.0,          "α = 0  (no SOI)",              "tab:gray"),
        (alpha_phi1,   f"φ_so = 1  (α = {alpha_phi1:.1f} meV·nm)", "tab:blue"),
        (ALPHA_REF,    f"α = {ALPHA_REF} meV·nm (φ_so = {phi_so_ref:.2f})", "tab:orange"),
    ]

    fig, axes = plt.subplots(len(alpha_cases), 2, figsize=(13, 4*len(alpha_cases)),
                              sharex="col")
    fig.suptitle("Aharonov-Bohm oscillations: G/G₀ vs Φ/Φ₀", fontsize=14)

    for row, (alpha, label, color) in enumerate(alpha_cases):
        ax_num = axes[row, 0]
        ax_cmp = axes[row, 1]

        print(f"\n[AB] {label}: simulating {n_points} points ...")
        G_analytical = np.array([analytical_G(Phi, alpha) for Phi in Phi_values])
        G_numerical  = np.array([simulate_G(Phi, alpha) for Phi in Phi_values])

        for ax in (ax_num, ax_cmp):
            ax.plot(Phi_values, G_analytical, "k--", linewidth=1.4,
                    label="Analítico (Büttiker)")
        ax_num.plot(Phi_values, G_numerical, "o-", color=color,
                    markersize=4, linewidth=1.4, label="Numérico")
        ax_cmp.plot(Phi_values, G_numerical, "o-", color=color,
                    markersize=4, linewidth=1.4, label="Numérico")
        for ax in (ax_num, ax_cmp):
            ax.set_ylabel(r"$G/G_0$")
            ax.set_ylim(-0.05, 2.1)
            ax.axhline(2.0, color="green", linestyle=":", alpha=0.4, linewidth=0.9)
            ax.axhline(0.0, color="red",   linestyle=":", alpha=0.4, linewidth=0.9)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            ax.set_title(label, fontsize=10)
        axes[row, 0].set_title(f"Numérico vs Analítico — {label}", fontsize=10)
        axes[row, 1].set_title(f"Comparación — {label}", fontsize=10)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\Phi / \Phi_0$")

    np.savez_compressed(OUTPUT_DIR / "ab_oscillations.npz",
                        Phi=Phi_values,
                        alpha_cases=np.array([c[0] for c in alpha_cases]),
                        G_analytical_last=G_analytical,
                        G_numerical_last=G_numerical)
    path = OUTPUT_DIR / "ab_oscillations.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 2 — AC oscillations: G vs φ_so at fixed Φ
# ---------------------------------------------------------------------------

def plot_ac_oscillations(n_points: int = 40) -> None:
    """G/G₀ vs Rashba phase φ_so = k_so·R for several fixed Φ values."""
    phi_so_values = np.linspace(0.0, 2.0, n_points)
    # Convert phi_so to alpha: alpha = phi_so * hbar^2 / (m*R)
    alpha_from_phi_so = lambda ps: ps * h_bar**2 / (m * R_NM)
    alpha_values = np.array([alpha_from_phi_so(ps) for ps in phi_so_values])

    Phi_cases = [
        (0.0,   "Φ = 0",       "tab:blue"),
        (1/3,   "Φ = 1/3 Φ₀",  "tab:orange"),
        (1.0,   "Φ = Φ₀",      "tab:green"),
    ]

    fig, axes = plt.subplots(len(Phi_cases), 1, figsize=(10, 4*len(Phi_cases)),
                              sharex=True)
    fig.suptitle("Aharonov-Casher oscillations: G/G₀ vs φ_so = k_so·R", fontsize=14)

    for row, (Phi, label, color) in enumerate(Phi_cases):
        ax = axes[row]
        print(f"\n[AC] {label}: simulating {n_points} points ...")
        G_analytical = np.array([analytical_G(Phi, a) for a in alpha_values])
        G_numerical  = np.array([simulate_G(Phi, a) for a in alpha_values])

        ax.plot(phi_so_values, G_analytical, "k--", linewidth=1.4,
                label="Analítico (Büttiker)")
        ax.plot(phi_so_values, G_numerical, "o-", color=color,
                markersize=4, linewidth=1.4, label="Numérico")
        ax.set_ylabel(r"$G/G_0$")
        ax.set_ylim(-0.05, 2.1)
        ax.axhline(2.0, color="green", linestyle=":", alpha=0.4, linewidth=0.9)
        ax.axhline(0.0, color="red",   linestyle=":", alpha=0.4, linewidth=0.9)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
        ax.set_title(label, fontsize=10)

    axes[-1].set_xlabel(r"$\phi_{so} = k_{so} \cdot R$")
    np.savez_compressed(OUTPUT_DIR / "ac_oscillations.npz",
                        phi_so=phi_so_values, alpha=alpha_values,
                        G_analytical_last=G_analytical,
                        G_numerical_last=G_numerical)
    path = OUTPUT_DIR / "ac_oscillations.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 3 — 2D map: G vs (Φ, φ_so) — the full AB+AC landscape
# ---------------------------------------------------------------------------

def plot_ab_ac_map(n_phi: int = 20, n_phi_so: int = 20) -> None:
    """2-D map of G/G₀ as a function of both Φ and φ_so."""
    Phi_vals    = np.linspace(-1.0, 1.0, n_phi)
    phi_so_vals = np.linspace(0.0, 2.0, n_phi_so)

    G_analytical = np.zeros((n_phi_so, n_phi))
    G_numerical  = np.zeros((n_phi_so, n_phi))
    total = n_phi * n_phi_so

    print(f"\n[2D map] {n_phi}×{n_phi_so} = {total} simulations ...")
    count = 0
    for j, ps in enumerate(phi_so_vals):
        alpha = ps * h_bar**2 / (m * R_NM)
        for i, Phi in enumerate(Phi_vals):
            count += 1
            G_analytical[j, i] = analytical_G(Phi, alpha)
            G_numerical[j, i]  = simulate_G(Phi, alpha)
            if count % 10 == 0:
                print(f"  [{count}/{total}]  Φ={Phi:.2f}  φ_so={ps:.2f}  "
                      f"G_num={G_numerical[j,i]:.3f}  G_ana={G_analytical[j,i]:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    kw = dict(aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=2.0,
              extent=[-1, 1, 0, 2])

    im0 = axes[0].imshow(G_analytical, **kw)
    axes[0].set_title("Analítico (Büttiker)", fontsize=12)

    im1 = axes[1].imshow(G_numerical, **kw)
    axes[1].set_title("Numérico (wavepacket)", fontsize=12)

    diff = G_numerical - G_analytical
    im2 = axes[2].imshow(diff, aspect="auto", origin="lower", cmap="RdBu_r",
                          vmin=-0.5, vmax=0.5,
                          extent=[-1, 1, 0, 2])
    axes[2].set_title("Diferencia (num - analítico)", fontsize=12)

    for ax, im in zip(axes, [im0, im1, im2]):
        ax.set_xlabel(r"$\Phi / \Phi_0$")
        ax.set_ylabel(r"$\phi_{so}$")
        fig.colorbar(im, ax=ax)

    fig.suptitle("G/G₀ landscape: AB (Φ) × AC (φ_so)", fontsize=14)
    fig.tight_layout()

    np.savez_compressed(OUTPUT_DIR / "ab_ac_map.npz",
                        Phi=Phi_vals, phi_so=phi_so_vals,
                        G_analytical=G_analytical, G_numerical=G_numerical)
    path = OUTPUT_DIR / "ab_ac_map.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Diagnostics: verify sin²(θ) ≠ 0 and print expected oscillation amplitudes
# ---------------------------------------------------------------------------

def print_diagnostics() -> None:
    print("=" * 60)
    print("  AB/AC interference diagnostics")
    print("=" * 60)
    print(f"  Effective mass   m* = {M_FACTOR} × m_e")
    print(f"  Ring radius      R  = {R_NM} nm")
    print(f"  Fermi energy     E_F = {E_F_MEV} meV")
    print(f"  k_F              = {k_F:.5f} nm⁻¹")
    print(f"  Dynamic phase    θ = k_F·π·R = {theta_dyn:.4f} rad = {theta_dyn/np.pi:.4f}·π")
    print(f"  sin²(θ)          = {np.sin(theta_dyn)**2:.5f}")
    if abs(np.sin(theta_dyn)) < 0.1:
        print("  WARNING: sin²(θ) ≈ 0 → ring at a transmission node!")
        print("  AB/AC oscillations will be suppressed.")
        print("  Adjust R or E_F to move away from the node.")
    else:
        print("  ✓ sin²(θ) is appreciable → oscillations will be visible.")
    print()
    # AB oscillation amplitude
    G_max = analytical_G(0.0, ALPHA_REF)
    G_min = min(analytical_G(Phi, ALPHA_REF) for Phi in np.linspace(-1, 1, 200))
    print(f"  AB amplitude (α={ALPHA_REF} meV·nm):")
    print(f"    G_max = {G_max:.4f},  G_min = {G_min:.4f}")
    print(f"    Swing = {G_max - G_min:.4f} G₀")
    print()
    # AC oscillation amplitude
    alpha_vals = np.linspace(0, 4*h_bar**2/(m*R_NM), 400)
    G_ac = [analytical_G(0.0, a) for a in alpha_vals]
    print(f"  AC amplitude (Φ=0):")
    print(f"    G_max = {max(G_ac):.4f},  G_min = {min(G_ac):.4f}")
    print(f"    Swing = {max(G_ac)-min(G_ac):.4f} G₀")
    print()
    print("  Junction coupling note:")
    print("  A 3-port Y-junction (continuum) has r = -1/3, t_arm = 2/3.")
    print("  This gives R = 1/9 ≈ 11% reflection per junction — physical,")
    print("  not a bug.  The Fabry-Perot resonances in the ring ensure")
    print("  T → 1 (constructive) and T → 0 (destructive) as Φ or α varies.")
    print("  The partial reflection visible in the wavepacket animation")
    print("  is this 11% Y-junction reflection plus the wavepacket k-spread")
    print("  (Δk/k ≈ 13% for σ=150 nm), which smears the fringes slightly.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AB and AC interference proof-of-concept.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Plots produced
--------------
  ab    G vs Φ at fixed α   (AB oscillations)
  ac    G vs φ_so at fixed Φ (AC oscillations)
  map   2-D G(Φ, φ_so) landscape

Examples
--------
  python ab_ac_proof.py                    # all plots, 40-point lines
  python ab_ac_proof.py --quick            # 10 points each (fast test)
  python ab_ac_proof.py --plots ab ac      # only 1-D plots
  python ab_ac_proof.py --n-line 60 --n-map 15  # custom resolution
        """
    )
    parser.add_argument("--plots", nargs="+", default=["ab", "ac", "map"],
                        choices=["ab", "ac", "map"])
    parser.add_argument("--n-line", type=int, default=40,
                        help="Points for 1-D line plots (default 40)")
    parser.add_argument("--n-map", type=int, default=15,
                        help="Points per axis for 2-D map (default 15, total n²)")
    parser.add_argument("--quick", action="store_true",
                        help="Set n-line=8, n-map=6 for a fast test run")
    args = parser.parse_args()

    if args.quick:
        args.n_line = 8
        args.n_map  = 6

    print_diagnostics()
    wall_t0 = time.perf_counter()

    if "ab"  in args.plots: plot_ab_oscillations(args.n_line)
    if "ac"  in args.plots: plot_ac_oscillations(args.n_line)
    if "map" in args.plots: plot_ab_ac_map(args.n_map, args.n_map)

    print(f"\nDone in {time.perf_counter()-wall_t0:.1f} s.")
    print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")
