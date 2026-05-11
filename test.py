import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------------------------------------------------------------
# Ring scattering model with Aharonov-Bohm and Aharonov-Casher phases
# -----------------------------------------------------------------------------------

# Physical parameters
m = 1.0                          # effective mass of the particle
R = 250.0                        # ring radius in arbitrary length units
N_l = 10                         # number of discretization points per lead
N_R = 120                        # number of discretization points per ring arm

def V(x):
    """Potential energy function for the system. Default: zero everywhere."""
    return 0.0


def build_index(N_R):
    """Build an index map for lead and ring variables."""
    idx = {}
    counter = 0

    for i in range(N_l):
        idx[f"L{i}"] = counter
        counter += 1

    for i in range(N_R):
        idx[f"U{i}"] = counter
        counter += 1

    for i in range(N_R):
        idx[f"D{i}"] = counter
        counter += 1

    for i in range(N_l):
        idx[f"R{i}"] = counter
        counter += 1

    return idx, counter


def build_matrices(N_R, phi_AB=0.0, phi_AC=0.0):
    """Build sparse Crank-Nicolson matrices A and B with AB/AC phases in the ring."""
    idx, size = build_index(N_R)
    A = sp.lil_matrix((size, size), dtype=complex)
    B = sp.lil_matrix((size, size), dtype=complex)

    lambda_lead = 1j / (2.0 * m)
    lambda_ring = 1j / (2.0 * m * R**2)

    def diag_element(lam, x):
        return 1.0 + lam + 0.5j * V(x)

    def diag_element_B(lam, x):
        return 1.0 - lam - 0.5j * V(x)

    # Phases for the ring arms
    # AB flux through the ring and AC spin-orbit-like phase offset
    total_flux_phase = 2.0 * np.pi * phi_AB
    ac_phase = phi_AC
    phase_up = (total_flux_phase + ac_phase) / max(1, N_R)
    phase_down = (total_flux_phase - ac_phase) / max(1, N_R)

    def lead_offdiag(lam):
        return -lam / 2.0

    def ring_offdiag_up(lam):
        return -lam / 2.0 * np.exp(1j * phase_up)

    def ring_offdiag_up_back(lam):
        return -lam / 2.0 * np.exp(-1j * phase_up)

    def ring_offdiag_down(lam):
        return -lam / 2.0 * np.exp(1j * phase_down)

    def ring_offdiag_down_back(lam):
        return -lam / 2.0 * np.exp(-1j * phase_down)

    def lead_offdiag_B(lam):
        return +lam / 2.0

    def ring_offdiag_up_B(lam):
        return +lam / 2.0 * np.exp(1j * phase_up)

    def ring_offdiag_up_back_B(lam):
        return +lam / 2.0 * np.exp(-1j * phase_up)

    def ring_offdiag_down_B(lam):
        return +lam / 2.0 * np.exp(1j * phase_down)

    def ring_offdiag_down_back_B(lam):
        return +lam / 2.0 * np.exp(-1j * phase_down)

    row = 0

    # Left lead interior rows
    for i in range(1, N_l - 1):
        A[row, idx[f"L{i-1}"]] = lead_offdiag(lambda_lead)
        A[row, idx[f"L{i}"]] = diag_element(lambda_lead, i)
        A[row, idx[f"L{i+1}"]] = lead_offdiag(lambda_lead)

        B[row, idx[f"L{i-1}"]] = lead_offdiag_B(lambda_lead)
        B[row, idx[f"L{i}"]] = diag_element_B(lambda_lead, i)
        B[row, idx[f"L{i+1}"]] = lead_offdiag_B(lambda_lead)
        row += 1

    # Junction conditions at left ring entry
    A[row, idx["L0"]] = B[row, idx["L0"]] = 1.0
    A[row, idx["U0"]] = B[row, idx["U0"]] = -1.0
    row += 1

    A[row, idx["L0"]] = B[row, idx["L0"]] = 1.0
    A[row, idx["D0"]] = B[row, idx["D0"]] = -1.0
    row += 1

    # Current conservation at left junction
    A[row, idx["L1"]] = B[row, idx["L1"]] = -1.0
    A[row, idx["L0"]] = B[row, idx["L0"]] = 1.0
    A[row, idx["U1"]] = B[row, idx["U1"]] = 1.0
    A[row, idx["U0"]] = B[row, idx["U0"]] = -1.0
    A[row, idx["D1"]] = B[row, idx["D1"]] = 1.0
    A[row, idx["D0"]] = B[row, idx["D0"]] = -1.0
    row += 1

    # Upper arm interior
    for i in range(1, N_R - 1):
        A[row, idx[f"U{i-1}"]] = ring_offdiag_up_back(lambda_ring)
        A[row, idx[f"U{i}"]] = diag_element(lambda_ring, i)
        A[row, idx[f"U{i+1}"]] = ring_offdiag_up(lambda_ring)

        B[row, idx[f"U{i-1}"]] = ring_offdiag_up_back_B(lambda_ring)
        B[row, idx[f"U{i}"]] = diag_element_B(lambda_ring, i)
        B[row, idx[f"U{i+1}"]] = ring_offdiag_up_B(lambda_ring)
        row += 1

    # Lower arm interior
    for i in range(1, N_R - 1):
        A[row, idx[f"D{i-1}"]] = ring_offdiag_down_back(lambda_ring)
        A[row, idx[f"D{i}"]] = diag_element(lambda_ring, i)
        A[row, idx[f"D{i+1}"]] = ring_offdiag_down(lambda_ring)

        B[row, idx[f"D{i-1}"]] = ring_offdiag_down_back_B(lambda_ring)
        B[row, idx[f"D{i}"]] = diag_element_B(lambda_ring, i)
        B[row, idx[f"D{i+1}"]] = ring_offdiag_down_B(lambda_ring)
        row += 1

    # Dirichlet condition at left lead end (left boundary point remains fixed)
    A[row, idx[f"L{N_l-1}"]] = 1.0
    B[row, idx[f"L{N_l-1}"]] = 1.0
    row += 1

    # Junction conditions at right ring exit
    A[row, idx["R0"]] = B[row, idx["R0"]] = 1.0
    A[row, idx[f"U{N_R-1}"]] = B[row, idx[f"U{N_R-1}"]] = -1.0
    row += 1

    A[row, idx["R0"]] = B[row, idx["R0"]] = 1.0
    A[row, idx[f"D{N_R-1}"]] = B[row, idx[f"D{N_R-1}"]] = -1.0
    row += 1

    # Current conservation at right junction
    A[row, idx[f"U{N_R-1}"]] = B[row, idx[f"U{N_R-1}"]] = -1.0
    A[row, idx[f"U{N_R-2}"]] = B[row, idx[f"U{N_R-2}"]] = 1.0
    A[row, idx[f"D{N_R-1}"]] = B[row, idx[f"D{N_R-1}"]] = -1.0
    A[row, idx[f"D{N_R-2}"]] = B[row, idx[f"D{N_R-2}"]] = 1.0
    A[row, idx["R1"]] = B[row, idx["R1"]] = 1.0
    A[row, idx["R0"]] = B[row, idx["R0"]] = -1.0
    row += 1

    # Right lead interior rows
    for i in range(1, N_l - 1):
        A[row, idx[f"R{i-1}"]] = lead_offdiag(lambda_lead)
        A[row, idx[f"R{i}"]] = diag_element(lambda_lead, i)
        A[row, idx[f"R{i+1}"]] = lead_offdiag(lambda_lead)

        B[row, idx[f"R{i-1}"]] = lead_offdiag_B(lambda_lead)
        B[row, idx[f"R{i}"]] = diag_element_B(lambda_lead, i)
        B[row, idx[f"R{i+1}"]] = lead_offdiag_B(lambda_lead)
        row += 1

    # Dirichlet condition at right lead end
    A[row, idx[f"R{N_l-1}"]] = 1.0
    B[row, idx[f"R{N_l-1}"]] = 1.0
    row += 1

    if row != size:
        raise RuntimeError(f"Matrix build mismatch: expected {size} rows but filled {row}")

    return A.tocsc(), B.tocsc(), idx


def build_initial_wavefunction(idx):
    """Initialize the wavefunction with a left-lead Gaussian wave packet."""
    size = len(idx)
    psi = np.zeros(size, dtype=complex)
    k = 1.0
    for j in range(N_l):
        x = float(j)
        psi[idx[f"L{j}"]] = np.exp(-((x - 2.0) ** 2) / 4.0) * np.exp(1j * k * x)
    return psi


def solve_time_evolution(A, B, psi_init, n_steps, solver_name):
    """Solve the Crank-Nicolson time evolution with a selected sparse solver."""
    psi = psi_init.copy()
    psi_history = [psi.copy()]
    d = B.dot(psi)

    if solver_name == "factorized":
        solver = spla.factorized(A)
    elif solver_name == "dense":
        A_dense = A.toarray()
    elif solver_name in {"gmres", "bicgstab"}:
        preconditioner = None
        try:
            ilu = spla.spilu(A)
            preconditioner = spla.LinearOperator(A.shape, lambda x: ilu.solve(x))
        except Exception:
            preconditioner = None

    for step in range(n_steps):
        if solver_name == "spsolve":
            psi = spla.spsolve(A, d)
        elif solver_name == "factorized":
            psi = solver(d)
        elif solver_name == "gmres":
            psi, info = spla.gmres(A, d, rtol=1e-9, restart=50, maxiter=500, M=preconditioner)
            if info != 0:
                raise RuntimeError(f"GMRES failed to converge at step {step}, info={info}")
        elif solver_name == "bicgstab":
            psi, info = spla.bicgstab(A, d, tol=1e-9, maxiter=500, M=preconditioner)
            if info != 0:
                raise RuntimeError(f"BiCGSTAB failed to converge at step {step}, info={info}")
        elif solver_name == "dense":
            psi = np.linalg.solve(A_dense, d)
        else:
            raise ValueError(f"Unknown solver {solver_name}")

        psi_history.append(psi.copy())
        d = B.dot(psi)

    return np.vstack(psi_history)


def time_solver(A, B, psi_init, n_steps, solver_list):
    """Compare solver runtimes and return history for the fastest one."""
    timings = {}
    histories = {}

    for solver_name in solver_list:
        start = time.perf_counter()
        try:
            history = solve_time_evolution(A, B, psi_init, n_steps, solver_name)
            elapsed = time.perf_counter() - start
            timings[solver_name] = elapsed
            histories[solver_name] = history
            print(f"Solver {solver_name}: {elapsed:.4f} s")
        except Exception as exc:
            elapsed = time.perf_counter() - start
            timings[solver_name] = None
            print(f"Solver {solver_name} failed after {elapsed:.4f} s: {exc}")

    successful = {name: t for name, t in timings.items() if t is not None}
    if not successful:
        raise RuntimeError("All solver attempts failed.")
    fastest = min(successful, key=successful.get)
    print(f"Fastest solver: {fastest} ({successful[fastest]:.4f} s)")
    return fastest, timings, histories[fastest]


def render_wavefunction_2d(psi, idx, grid_size=300):
    """Render the discrete wavefunction on a 2D plane with leads and ring."""
    grid = np.zeros((grid_size, grid_size), dtype=float)
    x_grid = np.linspace(-3.0, 3.0, grid_size)
    y_grid = np.linspace(-3.0, 3.0, grid_size)

    def fill_point(x, y, value):
        ix = int(round((x + 3.0) / 6.0 * (grid_size - 1)))
        iy = int(round((y + 3.0) / 6.0 * (grid_size - 1)))
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                xi = np.clip(ix + dx, 0, grid_size - 1)
                yi = np.clip(iy + dy, 0, grid_size - 1)
                grid[yi, xi] = max(grid[yi, xi], value)

    for j in range(N_l):
        x = -2.0
        y = -1.5 + 3.0 * j / max(1, N_l - 1)
        fill_point(x, y, np.abs(psi[idx[f"L{j}"]]) ** 2)

    for j in range(N_l):
        x = 2.0
        y = -1.5 + 3.0 * j / max(1, N_l - 1)
        fill_point(x, y, np.abs(psi[idx[f"R{j}"]]) ** 2)

    for j in range(N_R):
        angle = 2.0 * np.pi * j / N_R
        x = np.cos(angle)
        y = np.sin(angle)
        fill_point(x, y, np.abs(psi[idx[f"U{j}"]]) ** 2)
        fill_point(x, y, np.abs(psi[idx[f"D{j}"]]) ** 2)

    return grid, x_grid, y_grid


def animate_wavefunction(history, idx):
    """Animate the time evolution of |psi|^2 on the ring+leads geometry."""
    fig, ax = plt.subplots(figsize=(6, 6))
    initial_grid, x_grid, y_grid = render_wavefunction_2d(history[0], idx)
    im = ax.imshow(
        initial_grid,
        origin="lower",
        extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
        cmap="inferno",
        vmin=0.0,
        vmax=np.max(np.abs(history) ** 2),
    )
    ax.set_title("|psi|^2 evolution on leads and ring")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label=r"$|\psi|^2$")

    def update(frame):
        grid, _, _ = render_wavefunction_2d(history[frame], idx)
        im.set_data(grid)
        ax.set_title(f"Time step {frame}")
        return [im]

    return FuncAnimation(fig, update, frames=len(history), interval=150, blit=True)


def main():
    print("Building transfer matrices with Aharonov-Bohm and Aharonov-Casher phases...")
    phi_AB = 0.25
    phi_AC = 0.15
    A, B, idx = build_matrices(N_R, phi_AB=phi_AB, phi_AC=phi_AC)
    print(f"Matrix size: {A.shape[0]}x{A.shape[1]}")

    psi_init = build_initial_wavefunction(idx)
    time_steps = 80
    solver_candidates = ["spsolve", "factorized", "gmres"]

    print("Comparing solver performance...")
    fastest_solver, timings, history = time_solver(A, B, psi_init, time_steps, solver_candidates)

    print("Plotting the final wavefunction and animation...")
    fig, ax = plt.subplots(figsize=(6, 6))
    grid, x_grid, y_grid = render_wavefunction_2d(history[-1], idx)
    im = ax.imshow(
        grid,
        origin="lower",
        extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
        cmap="plasma",
        vmin=0.0,
        vmax=np.max(grid),
    )
    ax.set_title("Final |psi|^2 on the ring and leads")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label=r"$|\psi|^2$")
    plt.tight_layout()
    plt.show()

    ani = animate_wavefunction(history, idx)
    plt.show()

    print("\nSummary of execution:")
    print("- Constructed a sparse Crank-Nicolson matrix for a ring plus two leads.")
    print("- Included Aharonov-Bohm flux and Aharonov-Casher-like phase shifts in ring hopping.")
    print("- Compared sparse solvers: spsolve, factorized direct solve, and GMRES.")
    print(f"- Fastest solver: {fastest_solver}.")
    print("- Generated a 2D heat map where the left/right leads are vertical lines and the ring is a circle.")
    print("- Outside the leads and ring the field is zero, while the ring and lead points show |psi|^2.")


if __name__ == "__main__":
    main()
