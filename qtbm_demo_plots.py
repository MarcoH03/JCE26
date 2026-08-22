"""Generates the comparison figures referenced in README_QTBM.md.

Produces qtbm_ab_ac_comparison.png: AB and AC oscillations from QTBM
(exact, stationary) at two ring discretizations (N_R=151, the repo's
current default, and N_R=601, finer) overlaid on the closed-form Buttiker
analytical formula (thesis Ec. 3.2 / article Ec. 2-25/4-26), plus a panel
showing convergence of T(Phi=0) and T(Phi=0.5) toward the analytical
values as N_R increases.
"""
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qtbm


def main():
    t0 = time.perf_counter()
    n_points = 41

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # --- Panel (0,0): AB oscillations, N_R=151 vs N_R=601 vs analytic -------
    ax = axes[0, 0]
    Phi_values = np.linspace(-1.0, 1.0, n_points)
    G_analytic = np.array([qtbm.analytical_G_transparent(Phi, qtbm.ALPHA_REF) for Phi in Phi_values])
    ax.plot(Phi_values, G_analytic, "k--", linewidth=1.8, label="Analitico (Buttiker, Ec. 3.2 tesis JJ)")

    for N_R, color in [(151, "tab:red"), (601, "tab:blue")]:
        print(f"AB sweep, N_R={N_R} ...")
        sweep = qtbm.sweep_ab(alpha=qtbm.ALPHA_REF, n_points=n_points,
                              params_fn=lambda Phi, alpha, N_R=N_R: qtbm.transparent_ring_params(
                                  Phi=Phi, alpha=alpha, N_R=N_R, junction_correction=False))
        ax.plot(sweep["Phi"], sweep["T_total"], "o-", color=color, markersize=3,
                linewidth=1.2, label=f"QTBM, N_R={N_R} (junction_correction=False)")
    ax.set_xlabel(r"$\Phi/\Phi_0$"); ax.set_ylabel(r"$G/G_0$")
    ax.set_title("Oscilaciones Aharonov-Bohm (anillo transparente, alpha=20 meV*nm)")
    ax.set_ylim(-0.05, 2.1); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # --- Panel (0,1): AC oscillations, same comparison -----------------------
    ax = axes[0, 1]
    phi_so_values = np.linspace(0.0, 2.0, n_points)
    m = qtbm.M_FACTOR * __import__("tools").m_e
    alpha_from_phi_so = phi_so_values * __import__("tools").h_bar**2 / (m * qtbm.R_NM)
    G_analytic_ac = np.array([qtbm.analytical_G_transparent(0.0, a) for a in alpha_from_phi_so])
    ax.plot(phi_so_values, G_analytic_ac, "k--", linewidth=1.8, label="Analitico (Buttiker)")

    for N_R, color in [(151, "tab:red"), (601, "tab:blue")]:
        print(f"AC sweep, N_R={N_R} ...")
        sweep = qtbm.sweep_ac(Phi=0.0, n_points=n_points,
                              params_fn=lambda Phi, alpha, N_R=N_R: qtbm.transparent_ring_params(
                                  Phi=Phi, alpha=alpha, N_R=N_R, junction_correction=False))
        ax.plot(sweep["phi_so"], sweep["T_total"], "o-", color=color, markersize=3,
                linewidth=1.2, label=f"QTBM, N_R={N_R}")
    ax.set_xlabel(r"$\phi_{so}$"); ax.set_ylabel(r"$G/G_0$")
    ax.set_title("Oscilaciones Aharonov-Casher (anillo transparente, Phi=0)")
    ax.set_ylim(-0.05, 2.1); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # --- Panel (1,0): convergence of T vs N_R at two Phi values --------------
    ax = axes[1, 0]
    N_R_values = [151, 251, 401, 601, 901, 1501]
    T_at_0 = []
    T_at_half = []
    for N_R in N_R_values:
        print(f"Convergence check, N_R={N_R} ...")
        p0 = qtbm.transparent_ring_params(Phi=0.0, alpha=qtbm.ALPHA_REF, N_R=N_R, junction_correction=False)
        p5 = qtbm.transparent_ring_params(Phi=0.5, alpha=qtbm.ALPHA_REF, N_R=N_R, junction_correction=False)
        T_at_0.append(qtbm.qtbm_conductance(p0, qtbm.E_F_MEV).T)
        T_at_half.append(qtbm.qtbm_conductance(p5, qtbm.E_F_MEV).T)
    ax.axhline(qtbm.analytical_G_transparent(0.0, qtbm.ALPHA_REF), color="tab:orange",
              linestyle="--", linewidth=1.2, label="Analitico, Phi=0")
    ax.axhline(qtbm.analytical_G_transparent(0.5, qtbm.ALPHA_REF), color="tab:green",
              linestyle="--", linewidth=1.2, label="Analitico, Phi=0.5")
    ax.plot(N_R_values, T_at_0, "o-", color="tab:orange", label="QTBM, Phi=0")
    ax.plot(N_R_values, T_at_half, "o-", color="tab:green", label="QTBM, Phi=0.5")
    ax.set_xlabel(r"$N_R$ (nodos por brazo del anillo)"); ax.set_ylabel(r"$G/G_0$")
    ax.set_title("Convergencia hacia el limite continuo al refinar N_R")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # --- Panel (1,1): text summary -------------------------------------------
    ax = axes[1, 1]
    ax.axis("off")
    summary = (
        "Hallazgos clave (ver README_QTBM.md):\n\n"
        "1) QTBM conserva T+R=1 a precision de maquina\n"
        "   (~1e-13) en todos los puntos -- validacion\n"
        "   interna fuerte, independiente de la formula\n"
        "   analitica.\n\n"
        "2) junction_correction=True (default del repo)\n"
        "   EMPEORA el acuerdo con la formula analitica\n"
        "   en Phi=0, alpha=0: T=0.13 vs T=1.28 sin la\n"
        "   correccion (analitico predice 1.67).\n\n"
        "3) El desacuerdo dominante no es de frontera ni\n"
        "   de union: es de discretizacion del anillo.\n"
        "   Con N_R=151 (default) el numero de onda del\n"
        "   anillo se aleja notablemente del limite\n"
        "   continuo que asume la formula analitica.\n"
        "   Al refinar N_R el ajuste mejora sistemati-\n"
        "   camente (panel inferior izquierdo)."
    )
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, va="top", ha="left",
           fontsize=9.5, family="monospace")

    fig.suptitle("QTBM vs formula analitica de Buttiker -- anillo cuantico transparente", fontsize=13)
    fig.tight_layout()
    fig.savefig("qtbm_ab_ac_comparison.png", dpi=160)
    print(f"\nSaved: qtbm_ab_ac_comparison.png  ({time.perf_counter()-t0:.1f} s)")


if __name__ == "__main__":
    main()
