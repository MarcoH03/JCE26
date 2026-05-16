"""Run the spinor JCE26 simulation and persist each execution under RESULTADOS/.

This version keeps the script focused on three explicit responsibilities:

1. Build the stable Crank-Nicolson propagator from the Hermitian graph
   Hamiltonian defined in ``tools.py``.
2. Evolve a physically interpretable incoming packet from the left lead.
3. Save both local-density plots and probability-conservation diagnostics so it
   is easy to separate genuine transport from numerical artifacts.

Refactored to use the explicit ``PhysicsParams`` API in tools.py so that the
same simulation logic can be driven by ``conductance.py`` for parameter sweeps
without touching module-level state.
"""

from __future__ import annotations

from datetime import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import results_manager as rm
import tools as t


# ---------------------------------------------------------------------------
# Run-level configuration  (single-shot simulation settings)
# ---------------------------------------------------------------------------

FERMI_ENERGY_MEV = 4.19
TOTAL_TIME_PS    = 13.5
SNAPSHOT_COUNT   = 8
MIN_PLOT_DENSITY_MAX      = 1.0
MAX_PLOT_DENSITY          = 1e6
INITIAL_PACKET_CENTER_FRACTION = 0.8
INITIAL_PACKET_WIDTH_NM        = 150.0


# ---------------------------------------------------------------------------
# Physics helpers (now take p explicitly)
# ---------------------------------------------------------------------------

def compute_wave_number(p: t.PhysicsParams, fermi_energy_mev: float) -> float:
    """Convert the chosen Fermi energy into the incoming lead wave-number."""
    return np.sqrt(2 * p.m * fermi_energy_mev) / t.h_bar


def build_initial_wavefunction(
    p: t.PhysicsParams,
    layout: t.SingleRingLayout,
    k: float,
) -> np.ndarray:
    """Create a packet that starts in the left lead and moves towards the ring.

    The packet is intentionally left unnormalized because that matches the
    original workflow. What matters for the validation is not whether the
    initial probability equals one, but whether the total discrete probability
    stays constant relative to that initial value.
    """
    psi_initial = np.zeros(layout.spinor_size, dtype=complex)
    left_positions_nm = np.arange(layout.left_lead_sites.size, dtype=float) * p.delta_x
    packet_center_nm  = INITIAL_PACKET_CENTER_FRACTION * p.L_leads
    packet_width_nm   = max(3.0 * p.delta_x, INITIAL_PACKET_WIDTH_NM)

    for site, x_nm in zip(layout.left_lead_sites, left_positions_nm):
        amplitude = (np.exp(-0.5 * ((x_nm - packet_center_nm) / packet_width_nm) ** 2)
                     * np.exp(1j * k * x_nm))
        psi_initial[2 * site]     = amplitude
        psi_initial[2 * site + 1] = 0.0

    return psi_initial


def propagate_wavefunction(
    A: sp.spmatrix,
    B: sp.spmatrix,
    psi_initial: np.ndarray,
    time_steps: int,
) -> np.ndarray:
    """Advance the spinor state for ``time_steps`` using a cached sparse factorization."""
    solver      = spla.factorized(A.tocsc())
    psi_history = np.empty((time_steps + 1, psi_initial.size), dtype=complex)
    psi_history[0] = psi_initial

    psi_old = psi_initial
    for step in range(time_steps):
        rhs     = B @ psi_old
        psi_new = solver(rhs)
        psi_history[step + 1] = psi_new
        psi_old = psi_new

    return psi_history


def build_site_density_history(psi_history: np.ndarray, site_count: int) -> np.ndarray:
    """Project the spinor history into total local density per physical site."""
    spinor_by_site = psi_history.reshape(psi_history.shape[0], site_count, 2)
    return np.sum(np.abs(spinor_by_site) ** 2, axis=2, dtype=float)


def build_plot_density_history(
    site_density_history: np.ndarray,
    layout: t.SingleRingLayout,
) -> np.ndarray:
    """Duplicate shared junction nodes so the branched graph can be plotted linearly."""
    return site_density_history[:, layout.plot_sites]


def build_probability_history(site_density_history: np.ndarray) -> np.ndarray:
    """Return the conserved probability of the discrete graph model."""
    return np.sum(site_density_history, axis=1, dtype=float)


def build_snapshot_indices(frame_count: int, snapshot_count: int) -> np.ndarray:
    """Select evenly spaced frames, always including the first and last."""
    if frame_count <= snapshot_count:
        return np.arange(frame_count)
    return np.unique(np.linspace(0, frame_count - 1, snapshot_count, dtype=int))


def summarize_probability_regions(
    final_site_density: np.ndarray,
    layout: t.SingleRingLayout,
) -> dict[str, float]:
    """Partition the final probability into branches and shared junction nodes."""
    return {
        "left_lead":           float(np.sum(final_site_density[layout.left_lead_sites[:-1]])),
        "left_junction":       float(final_site_density[layout.left_junction_site]),
        "upper_arm_internal":  float(np.sum(final_site_density[layout.upper_internal_sites])),
        "lower_arm_internal":  float(np.sum(final_site_density[layout.lower_internal_sites])),
        "right_junction":      float(final_site_density[layout.right_junction_site]),
        "right_lead":          float(np.sum(final_site_density[layout.right_lead_sites[1:]])),
    }


def build_region_probability_history(
    site_density_history: np.ndarray,
    layout: t.SingleRingLayout,
) -> dict[str, np.ndarray]:
    """Track how probability moves between source lead, ring, and drain lead."""
    left_lead        = np.sum(site_density_history[:, layout.left_lead_sites[:-1]], axis=1, dtype=float)
    left_junction    = site_density_history[:, layout.left_junction_site]
    upper_arm        = np.sum(site_density_history[:, layout.upper_internal_sites], axis=1, dtype=float)
    lower_arm        = np.sum(site_density_history[:, layout.lower_internal_sites], axis=1, dtype=float)
    right_junction   = site_density_history[:, layout.right_junction_site]
    right_lead       = np.sum(site_density_history[:, layout.right_lead_sites[1:]], axis=1, dtype=float)
    ring_total       = left_junction + upper_arm + lower_arm + right_junction

    return {
        "left_lead":           left_lead,
        "left_junction":       left_junction,
        "upper_arm_internal":  upper_arm,
        "lower_arm_internal":  lower_arm,
        "right_junction":      right_junction,
        "right_lead":          right_lead,
        "ring_total":          ring_total,
    }


def estimate_reflection_horizons_ps(p: t.PhysicsParams, k: float) -> dict[str, float]:
    """Estimate when external hard-wall reflections first return to the device."""
    lead_velocity_nm_ps = t.lead_group_velocity(p, k)
    packet_center_nm    = INITIAL_PACKET_CENTER_FRACTION * p.L_leads
    distance_to_left    = packet_center_nm
    distance_to_junction = p.L_leads - packet_center_nm
    time_to_right_junction_ps = (distance_to_junction + p.L_ring) / lead_velocity_nm_ps

    return {
        "lead_velocity_nm_ps": lead_velocity_nm_ps,
        "left_reflection_return_to_junction_ps": (distance_to_left + p.L_leads) / lead_velocity_nm_ps,
        "right_reflection_return_to_junction_ps": (
            time_to_right_junction_ps + 2.0 * p.L_leads / lead_velocity_nm_ps
        ),
        "packet_center_to_left_junction_ps":  distance_to_junction / lead_velocity_nm_ps,
        "packet_center_to_right_junction_ps": time_to_right_junction_ps,
    }


# ---------------------------------------------------------------------------
# Plotting helpers  (unchanged in substance)
# ---------------------------------------------------------------------------

def backend_supports_show() -> bool:
    return "agg" not in plt.get_backend().lower()


def section_boundaries(section_labels: list[str] | tuple[str, ...]) -> list[float]:
    boundaries = []
    previous   = section_labels[0]
    for index, label in enumerate(section_labels[1:], start=1):
        if label != previous:
            boundaries.append(index - 0.5)
            previous = label
    return boundaries


def add_section_guides(axis, section_labels: list[str] | tuple[str, ...]) -> None:
    for boundary in section_boundaries(section_labels):
        axis.axvline(boundary, color="white", alpha=0.15, linewidth=0.8, linestyle="--")


def sanitize_density_for_plotting(density_history: np.ndarray) -> tuple[np.ndarray, float, int]:
    finite_mask        = np.isfinite(density_history)
    nonfinite_entries  = int(density_history.size - np.count_nonzero(finite_mask))

    if np.any(finite_mask):
        plot_cap = float(np.quantile(density_history[finite_mask], 0.995))
        plot_cap = max(plot_cap, MIN_PLOT_DENSITY_MAX)
    else:
        plot_cap = MIN_PLOT_DENSITY_MAX

    plot_cap  = min(plot_cap, MAX_PLOT_DENSITY)
    sanitized = np.nan_to_num(density_history, nan=0.0, posinf=plot_cap, neginf=0.0)
    sanitized = np.clip(sanitized, 0.0, plot_cap)
    return sanitized, plot_cap, nonfinite_entries


def save_density_snapshots(density_history, snapshot_indices, section_labels,
                           y_limit_max, output_path) -> None:
    x_nodes = np.arange(density_history.shape[1])
    figure, axes = plt.subplots(snapshot_indices.size, 1,
                                figsize=(11, 2.5 * snapshot_indices.size), sharex=True)
    if snapshot_indices.size == 1:
        axes = [axes]
    for axis, frame in zip(axes, snapshot_indices):
        axis.plot(x_nodes, density_history[frame], color="tab:blue", linewidth=1.2)
        axis.set_ylabel(r"$|\psi|^2$")
        axis.set_ylim(0.0, y_limit_max)
        axis.set_title(f"Densidad local en el paso temporal {frame}")
        axis.grid(True, alpha=0.25)
        add_section_guides(axis, section_labels)
    axes[-1].set_xlabel("Nodo linealizado L -> U -> D -> R")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_density_heatmap(density_history, section_labels, output_path) -> None:
    figure, axis = plt.subplots(figsize=(12, 6))
    image = axis.imshow(density_history, aspect="auto", origin="lower", cmap="inferno",
                        extent=[0, density_history.shape[1] - 1,
                                0, density_history.shape[0] - 1])
    axis.set_xlabel("Nodo linealizado L -> U -> D -> R")
    axis.set_ylabel("Paso temporal")
    axis.set_title(r"Historia completa de $|\psi|^2$")
    add_section_guides(axis, section_labels)
    figure.colorbar(image, ax=axis, label=r"$|\psi|^2$")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_potential_profile(node_indices, potential_profile, section_labels, output_path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(node_indices, potential_profile, color="tab:orange", linewidth=1.5, label="V")
    axis.set_xlabel("Nodo linealizado L -> U -> D -> R")
    axis.set_ylabel("Potencial (meV)")
    axis.set_title("Perfil de potencial")
    axis.grid(True, alpha=0.25)
    add_section_guides(axis, section_labels)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_initial_final_density(density_history, section_labels, y_limit_max, output_path) -> None:
    x_nodes = np.arange(density_history.shape[1])
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(x_nodes, density_history[0],  label="Inicial", linewidth=1.2)
    axis.plot(x_nodes, density_history[-1], label="Final",   linewidth=1.2)
    axis.set_xlabel("Nodo linealizado L -> U -> D -> R")
    axis.set_ylabel(r"$|\psi|^2$")
    axis.set_title("Comparación entre densidad inicial y final")
    axis.set_ylim(0.0, y_limit_max)
    axis.grid(True, alpha=0.25)
    add_section_guides(axis, section_labels)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_probability_history(time_axis_ps, probability_history,
                              relative_probability_history, output_path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(time_axis_ps, probability_history, color="tab:green", linewidth=1.2)
    axes[0].set_ylabel("Probabilidad")
    axes[0].set_title("Probabilidad discreta total")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(time_axis_ps, relative_probability_history, color="tab:red", linewidth=1.2)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
    axes[1].set_xlabel("Tiempo (ps)")
    axes[1].set_ylabel("P/P0")
    axes[1].set_title("Conservación relativa de la norma discreta")
    axes[1].grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_region_probability_history(time_axis_ps, region_history, output_path) -> None:
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.plot(time_axis_ps, region_history["left_lead"],  label="Lead izquierdo",      linewidth=1.2)
    axis.plot(time_axis_ps, region_history["ring_total"], label="Región de dispersión", linewidth=1.2)
    axis.plot(time_axis_ps, region_history["right_lead"], label="Lead derecho",         linewidth=1.2)
    axis.set_xlabel("Tiempo (ps)")
    axis.set_ylabel("Probabilidad discreta")
    axis.set_title("Evolución regional de la probabilidad")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def animate_density_history(density_history, section_labels,
                             interval_ms=100, y_limit_max=None, save_path=None) -> None:
    from matplotlib import animation

    x_nodes = np.arange(density_history.shape[1])
    if y_limit_max is None:
        y_limit_max = max(1.0, float(np.nanmax(density_history)) * 1.05)

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.set_xlim(0, density_history.shape[1] - 1)
    axis.set_ylim(0.0, y_limit_max)
    axis.set_xlabel("Nodo linealizado L -> U -> D -> R")
    axis.set_ylabel(r"$|\psi|^2$")
    axis.grid(True, alpha=0.25)
    add_section_guides(axis, section_labels)

    line,      = axis.plot([], [], color="tab:blue", linewidth=1.5)
    frame_text = axis.text(0.98, 0.95, "", transform=axis.transAxes,
                           ha="right", va="top", color="white", fontsize=10)

    def init():
        line.set_data([], [])
        frame_text.set_text("")
        return line, frame_text

    def update(frame):
        line.set_data(x_nodes, density_history[frame])
        frame_text.set_text(f"Paso temporal {frame}")
        return line, frame_text

    anim = animation.FuncAnimation(figure, update, frames=density_history.shape[0],
                                   init_func=init, blit=True, interval=interval_ms)
    if save_path is not None:
        writer = animation.PillowWriter(fps=max(1, int(round(1000.0 / interval_ms))))
        anim.save(save_path, writer=writer)

    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute the simulation, archive the run and print the key diagnostics."""

    # --- Build the parameter set for this run --------------------------------
    # Change any field here; all downstream calls use p explicitly.
    p = t.default_params()

    layout            = t.build_single_ring_layout(p)
    run_directories   = rm.create_run_directories()
    execution_started = datetime.now().astimezone()
    wall_start        = time.perf_counter()

    k                    = compute_wave_number(p, FERMI_ENERGY_MEV)
    phase_around_ring    = k * (2 * np.pi * p.R)
    lead_velocity_nm_ps  = t.lead_group_velocity(p, k)
    reflection_horizons  = estimate_reflection_horizons_ps(p, k)
    potential_summary    = t.summarize_potential_by_section(p, layout)
    time_steps           = t.time_steps_for_duration(p, TOTAL_TIME_PS)

    print(f"theta = {phase_around_ring:.2f} rad")
    print(f"k = {k:.4f} nm^-1")
    print(f"v_group = {lead_velocity_nm_ps:.2f} nm/ps")
    print(f"potential_model = {p.potential_model}")

    A, B, size = t.build_cn_matrices(p, layout)
    hamiltonian = t.build_single_ring_hamiltonian(p, layout)

    psi_initial = build_initial_wavefunction(p, layout, k)
    psi_history = propagate_wavefunction(A, B, psi_initial, time_steps)

    site_density_history      = build_site_density_history(psi_history, layout.unique_site_count)
    plot_density_history      = build_plot_density_history(site_density_history, layout)
    probability_history       = build_probability_history(site_density_history)
    region_probability_history = build_region_probability_history(site_density_history, layout)
    relative_probability_history = probability_history / probability_history[0]
    peak_density_history      = np.max(site_density_history, axis=1)

    density_history_plot, density_plot_cap, nonfinite_density_entries = \
        sanitize_density_for_plotting(plot_density_history)

    node_indices, potential_profile, section_labels = t.build_potential_profile(p, layout)
    snapshot_indices       = build_snapshot_indices(density_history_plot.shape[0], SNAPSHOT_COUNT)
    final_probability_regions = summarize_probability_regions(site_density_history[-1], layout)
    plot_y_limit           = max(MIN_PLOT_DENSITY_MAX, density_plot_cap)
    time_axis_ps           = np.arange(time_steps + 1, dtype=float) * p.dt

    # --- Plots ---------------------------------------------------------------
    animate_density_history(plot_density_history, section_labels,
                            interval_ms=10, y_limit_max=plot_y_limit)

    save_initial_final_density(density_history_plot, section_labels, plot_y_limit,
                               run_directories.images / "density_initial_final.png")
    save_density_snapshots(density_history_plot, snapshot_indices, section_labels,
                           plot_y_limit, run_directories.images / "density_snapshots.png")
    save_density_heatmap(density_history_plot, section_labels,
                         run_directories.images / "density_heatmap.png")
    save_potential_profile(node_indices, potential_profile, section_labels,
                           run_directories.images / "potential_profile.png")
    save_probability_history(time_axis_ps, probability_history,
                             relative_probability_history,
                             run_directories.images / "probability_history.png")
    save_region_probability_history(time_axis_ps, region_probability_history,
                                    run_directories.images / "region_probabilities.png")

    # --- Diagnostics / metadata ----------------------------------------------
    execution_finished = datetime.now().astimezone()
    wall_seconds       = time.perf_counter() - wall_start

    hermiticity_residual = hamiltonian - hamiltonian.getH()
    hermiticity_max_abs  = (float(np.max(np.abs(hermiticity_residual.data)))
                            if hermiticity_residual.nnz else 0.0)
    probability_drift         = float(probability_history[-1] - probability_history[0])
    relative_probability_drift = float(relative_probability_history[-1] - 1.0)
    right_lead_fraction_final = float(region_probability_history["right_lead"][-1] / probability_history[-1])
    left_lead_fraction_final  = float(region_probability_history["left_lead"][-1]  / probability_history[-1])
    ring_fraction_final       = float(region_probability_history["ring_total"][-1] / probability_history[-1])

    parameters = p.to_dict()
    parameters.update({
        "fermi_energy_mev":        FERMI_ENERGY_MEV,
        "k_nm_inverse":            float(k),
        "phase_around_ring_rad":   float(phase_around_ring),
        "lead_group_velocity_nm_ps": float(lead_velocity_nm_ps),
        "total_time_ps":           TOTAL_TIME_PS,
        "time_steps":              time_steps,
    })

    matrix_summary = {
        "size": size,
        "A_shape": list(A.shape),
        "B_shape": list(B.shape),
        "H_shape": list(hamiltonian.shape),
        "A_nnz":   int(A.nnz),
        "B_nnz":   int(B.nnz),
        "H_nnz":   int(hamiltonian.nnz),
        "A_density": float(A.nnz / (A.shape[0] * A.shape[1])),
        "B_density": float(B.nnz / (B.shape[0] * B.shape[1])),
        "H_density": float(hamiltonian.nnz / (hamiltonian.shape[0] * hamiltonian.shape[1])),
        "hamiltonian_hermiticity_max_abs": hermiticity_max_abs,
    }

    run_manifest = {
        "run_id":           run_directories.run_id,
        "run_number":       run_directories.run_number,
        "timestamp_local":  run_directories.timestamp,
        "started_at_local": execution_started.isoformat(),
        "finished_at_local": execution_finished.isoformat(),
        "wall_time_seconds": wall_seconds,
        "results_root":     str(run_directories.run_root),
        "images_dir":       str(run_directories.images),
        "calculations_dir": str(run_directories.calculations),
        "metadata_dir":     str(run_directories.metadata),
        "files": {
            "density_initial_final": "imagenes/density_initial_final.png",
            "density_snapshots":     "imagenes/density_snapshots.png",
            "density_heatmap":       "imagenes/density_heatmap.png",
            "potential_profile":     "imagenes/potential_profile.png",
            "probability_history":   "imagenes/probability_history.png",
            "region_probabilities":  "imagenes/region_probabilities.png",
            "density_history":       "calculos/density_history.npz",
            "matrix_summary":        "calculos/matrix_summary.json",
            "parameters":            "metadata/parameters.json",
            "run_manifest":          "metadata/run_manifest.json",
            "execution_log":         "metadata/execution.log",
        },
        "reflection_horizons_ps":    reflection_horizons,
        "potential_model":           p.potential_model,
        "potential_section_summary": potential_summary,
        "final_probability_regions": final_probability_regions,
        "density_plot_cap":          density_plot_cap,
        "nonfinite_density_entries": nonfinite_density_entries,
        "probability_definition":    "sum of site occupations on the orthonormal graph basis",
        "probability": {
            "initial":        float(probability_history[0]),
            "final":          float(probability_history[-1]),
            "minimum":        float(probability_history.min()),
            "maximum":        float(probability_history.max()),
            "absolute_drift": probability_drift,
            "relative_drift": relative_probability_drift,
        },
        "asymptotic_transport": {
            "final_left_lead_fraction":  left_lead_fraction_final,
            "final_ring_fraction":       ring_fraction_final,
            "final_right_lead_fraction": right_lead_fraction_final,
        },
        "peak_density": {
            "initial": float(peak_density_history[0]),
            "final":   float(peak_density_history[-1]),
            "maximum": float(peak_density_history.max()),
        },
    }

    np.savez_compressed(
        run_directories.calculations / "density_history.npz",
        site_density_history=site_density_history,
        plot_density_history=plot_density_history,
        density_history_plot=density_history_plot,
        probability_history=probability_history,
        relative_probability_history=relative_probability_history,
        left_lead_probability_history=region_probability_history["left_lead"],
        ring_probability_history=region_probability_history["ring_total"],
        right_lead_probability_history=region_probability_history["right_lead"],
        peak_density_history=peak_density_history,
        node_indices=node_indices,
        potential_profile=potential_profile,
        section_labels=np.array(section_labels),
        snapshot_indices=snapshot_indices,
        snapshot_density=plot_density_history[snapshot_indices],
        psi_initial=psi_history[0],
        psi_final=psi_history[-1],
        time_axis_ps=time_axis_ps,
        integration_weights_nm=layout.integration_weights_nm,
        plot_sites=layout.plot_sites,
        left_lead_sites=layout.left_lead_sites,
        upper_internal_sites=layout.upper_internal_sites,
        lower_internal_sites=layout.lower_internal_sites,
        right_lead_sites=layout.right_lead_sites,
    )

    rm.write_json(run_directories.calculations / "matrix_summary.json", matrix_summary)
    rm.write_json(run_directories.metadata / "parameters.json", parameters)
    rm.write_json(run_directories.metadata / "run_manifest.json", run_manifest)
    rm.write_text(
        run_directories.metadata / "execution.log",
        [
            f"run_id: {run_directories.run_id}",
            f"started_at: {execution_started.isoformat()}",
            f"finished_at: {execution_finished.isoformat()}",
            f"wall_time_seconds: {wall_seconds:.6f}",
            f"k_nm_inverse: {k:.6f}",
            f"phase_around_ring_rad: {phase_around_ring:.6f}",
            f"lead_group_velocity_nm_ps: {lead_velocity_nm_ps:.6f}",
            f"total_time_ps: {TOTAL_TIME_PS:.6f}",
            f"time_steps: {time_steps}",
            f"potential_model: {p.potential_model}",
            f"alpha_mev_nm: {p.alpha:.6f}",
            f"left_reflection_return_to_junction_ps: "
            f"{reflection_horizons['left_reflection_return_to_junction_ps']:.6f}",
            f"right_reflection_return_to_junction_ps: "
            f"{reflection_horizons['right_reflection_return_to_junction_ps']:.6f}",
            f"results_dir: {run_directories.run_root}",
            f"probability_definition: discrete graph norm",
            f"probability_initial: {probability_history[0]:.12f}",
            f"probability_final: {probability_history[-1]:.12f}",
            f"probability_relative_final: {relative_probability_history[-1]:.12f}",
            f"final_left_lead_fraction: {left_lead_fraction_final:.12f}",
            f"final_ring_fraction: {ring_fraction_final:.12f}",
            f"final_right_lead_fraction: {right_lead_fraction_final:.12f}",
            f"peak_density_max: {peak_density_history.max():.12f}",
        ],
    )

    print(f"Resultados guardados en: {run_directories.run_root}")
    print(f"Tiempo de ejecución: {wall_seconds:.2f} s")
    print(f"Probabilidad discreta final P/P0 = {relative_probability_history[-1]:.8f}")
    print(f"Drift relativo de probabilidad = {relative_probability_drift:+.3e}")
    print(f"Fracción final lead izquierdo = {left_lead_fraction_final:.6f}")
    print(f"Fracción final región de dispersión = {ring_fraction_final:.6f}")
    print(f"Fracción final lead derecho = {right_lead_fraction_final:.6f}")
    print(f"Máximo local de |psi|^2 observado = {peak_density_history.max():.6f}")

    if nonfinite_density_entries:
        print(
            "Advertencia: la corrida produjo densidades no finitas; se guardaron los arrays crudos "
            "y las imágenes usaron clipping diagnóstico."
        )

    if backend_supports_show():
        plt.show()


if __name__ == "__main__":
    main()
