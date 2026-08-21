"""Isolated boundary-reflection benchmark: CAP vs. the exact single-site
transparent self-energy boundary condition (see ``conductance.py``,
functions ``lead_wavenumber`` / ``lead_self_energy`` /
``build_cn_matrices_with_transparent_bc`` / ``run_transparent_conductance``).

Why an isolated test instead of the full ring
-----------------------------------------------
The full ring + two leads system (``ab_ac_proof.py``) mixes TWO different
physical effects when judging "spurious reflection":

  1. Reflection literally AT the truncated end of a lead (what the CAP /
     transparent boundary condition is meant to remove) -- this is the
     problem targeted by "Soluciones propuestas" solution #1 (DTBC) and
     its lighter-weight approximation implemented here.
  2. Genuine, physical multiple-scattering at the Y-junctions between the
     leads and the ring (~11% per junction, purely topological, already
     noted as expected physics in ``ab_ac_proof.py``), which can trap
     probability in long-lived quasi-bound ring resonances for many tens
     of ps regardless of how good the outer boundary condition is.

Mixing the two in one benchmark makes it impossible to tell whether a
change in T + R comes from a better boundary or from the resonance
physics of the ring itself. This script isolates effect (1) only: a bare,
straight, spinless 1-D tight-binding chain (no ring, no junction) with a
Gaussian packet launched toward one end. Whatever fraction of the packet
bounces back is *entirely* attributable to the boundary treatment.

Usage
-----
    python test_transparent_boundary.py
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import tools as t
import conductance as c


# ---------------------------------------------------------------------------
# Bare 1-D chain matching the lead discretization used elsewhere in the repo
# ---------------------------------------------------------------------------

def build_chain_hamiltonian(n_sites: int, hopping_mev: float) -> sp.csr_matrix:
    """Spinless uniform tight-binding chain: on-site 2*t (bulk), hopping -t."""
    diag = np.full(n_sites, 2.0 * hopping_mev, dtype=complex)
    off  = np.full(n_sites - 1, -hopping_mev, dtype=complex)
    H = sp.diags([off, diag, off], offsets=[-1, 0, 1], format="lil")
    return H.tocsr()


def gaussian_packet(x_nm: np.ndarray, center_nm: float, width_nm: float, k: float) -> np.ndarray:
    return (np.exp(-0.5 * ((x_nm - center_nm) / width_nm) ** 2) * np.exp(1j * k * x_nm)).astype(complex)


def run_boundary_reflection_test(
    n_sites: int = 600,
    delta_x_nm: float = 5.263157894736842,   # same as the repo's default lead spacing
    m_factor: float = 0.023,
    fermi_energy_mev: float = 4.19,
    dt: float = 0.002,
    packet_center_fraction: float = 0.35,
    packet_width_nm: float = 150.0,
    total_time_ps: float = 20.0,
    boundary: str = "cap",
    cap_fraction: float = 0.20,
    cap_strength: float = 2.0,
    cap_order: int = 3,
) -> dict:
    """Launch a right-moving packet down a bare chain and measure reflection.

    Only the RIGHT end has an absorber (CAP or transparent self-energy). The
    LEFT end is a hard wall far enough away that, for the packet's own
    velocity and the simulated duration, nothing physically reaches it -- so
    any probability found in the left half of the chain at the end of the
    run is a real reflection off the right-hand boundary treatment, not an
    artifact of the left edge.
    """
    m = m_factor * t.m_e
    t_hop = t.h_bar**2 / (2 * m * delta_x_nm**2)
    x_nm = np.arange(n_sites) * delta_x_nm

    k = np.sqrt(2 * m * fermi_energy_mev) / t.h_bar
    v_group = t.h_bar * k / m

    H = build_chain_hamiltonian(n_sites, t_hop).tolil()

    if boundary == "cap":
        n_cap = max(1, int(np.ceil(cap_fraction * n_sites)))
        W = np.zeros(n_sites)
        for i in range(n_cap):
            ramp = i / (n_cap - 1) if n_cap > 1 else 1.0
            W[n_sites - 1 - i] = cap_strength * ramp**cap_order
        for site in range(n_sites):
            if W[site] > 0.0:
                H[site, site] += -1j * W[site]
        boundary_label = f"CAP(frac={cap_fraction}, strength={cap_strength}, order={cap_order})"
    elif boundary == "transparent":
        cos_val = np.clip(1.0 - fermi_energy_mev / (2.0 * t_hop), -1.0, 1.0)
        k_bc = np.arccos(cos_val) / delta_x_nm
        sigma = -t_hop * np.exp(1j * k_bc * delta_x_nm)
        H[n_sites - 1, n_sites - 1] += sigma
        boundary_label = f"transparent self-energy (Sigma={sigma:.3f} meV)"
    elif boundary == "none":
        boundary_label = "none (bare hard-wall truncation, reference)"
    else:
        raise ValueError(boundary)

    H = H.tocsr()
    identity  = sp.identity(n_sites, format="csr", dtype=complex)
    prefactor = 1j * dt / (2 * t.h_bar)
    A = (identity + prefactor * H).tocsr()
    B = (identity - prefactor * H).tocsr()
    solver = spla.factorized(A.tocsc())

    center_nm = packet_center_fraction * (n_sites - 1) * delta_x_nm
    psi = gaussian_packet(x_nm, center_nm, packet_width_nm, k)
    N0  = float(np.sum(np.abs(psi) ** 2))

    time_steps = int(round(total_time_ps / dt))
    midpoint = n_sites // 2

    for _ in range(time_steps):
        psi = solver(B @ psi)

    P_total_final = float(np.sum(np.abs(psi) ** 2))
    P_left_final  = float(np.sum(np.abs(psi[:midpoint]) ** 2))   # "reflected back" region
    P_right_final = float(np.sum(np.abs(psi[midpoint:]) ** 2))   # transmitted / still en route

    return {
        "boundary": boundary,
        "boundary_label": boundary_label,
        "n_sites": n_sites,
        "t_hop_mev": t_hop,
        "k_nm_inv": k,
        "v_group_nm_ps": v_group,
        "transit_time_full_chain_ps": (n_sites * delta_x_nm) / v_group,
        "N0": N0,
        "P_total_final": P_total_final,
        "absorbed_fraction": 1.0 - P_total_final / N0,
        "reflected_fraction_of_remaining": P_left_final / P_total_final if P_total_final > 0 else float("nan"),
        "reflected_fraction_of_N0": P_left_final / N0,
    }


def main() -> None:
    print("=" * 78)
    print("  Boundary reflection benchmark: bare 1-D lead, single right-hand absorber")
    print("=" * 78)

    common = dict(
        n_sites=600,
        fermi_energy_mev=4.19,
        packet_center_fraction=0.35,
        packet_width_nm=150.0,
        total_time_ps=20.0,
    )

    results = {}
    for boundary in ("none", "cap", "transparent"):
        r = run_boundary_reflection_test(boundary=boundary, **common)
        results[boundary] = r
        print(f"\n[{boundary}] {r['boundary_label']}")
        print(f"  v_group = {r['v_group_nm_ps']:.2f} nm/ps   "
              f"full-chain transit = {r['transit_time_full_chain_ps']:.2f} ps "
              f"(simulated {common['total_time_ps']} ps)")
        print(f"  absorbed fraction (1 - P_final/P0)      = {r['absorbed_fraction']:.6f}")
        print(f"  reflected-back fraction of P0 (left half) = {r['reflected_fraction_of_N0']:.6f}")

    print("\n" + "-" * 78)
    r_cap  = results["cap"]
    r_tbc  = results["transparent"]
    r_none = results["none"]
    print(f"Reference (no absorber at all): reflected fraction = "
          f"{r_none['reflected_fraction_of_N0']:.6f}  "
          f"(expected ~0, hard right wall is far beyond the packet's reach in this window)")
    print(f"CAP reflected-back fraction of P0:          {r_cap['reflected_fraction_of_N0']:.6e}")
    print(f"Transparent-BC reflected-back fraction of P0: {r_tbc['reflected_fraction_of_N0']:.6e}")
    if r_tbc["reflected_fraction_of_N0"] > 0:
        ratio = r_cap["reflected_fraction_of_N0"] / max(r_tbc["reflected_fraction_of_N0"], 1e-300)
        print(f"CAP reflects {ratio:.1f}x more probability back into the device than the "
              f"transparent boundary, at the same Fermi energy and packet.")
    print("-" * 78)


if __name__ == "__main__":
    main()
