import tools as t
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

N_R = t.N_R #number of points per branch of the ring
N_l = t.N_l

Ef = 4.19 #Fermi Energy in meV
theta = 12.5*np.pi 

#k= np.sqrt(2*t.m*Ef)/t.h_bar #wavevector of the initial wavefunction in the left lead
#k = theta/(2*np.pi*t.R)
k = 100


theta = k * (2*np.pi*t.R)
print(f"theta = {theta:.2f} rad")
print(f"k = {k:.4f} nm^-1")

A, B, size = t.matrix_A_B_generator_single_ring(N_R, k)
A = sp.csr_matrix(A)
B = sp.csr_matrix(B)
time_steps = 10000



psi_array = np.empty(time_steps+1, dtype=object)

#The wavefunction at the current (known) time step
psi_old = np.zeros(size, dtype=complex)

for j in range(N_l):
    x = j

    idx_up = 2*j
    idx_down = 2*j + 1

    psi_old[idx_up] = np.exp(-(x-2)**2) * np.exp(1j*k*x)
    psi_old[idx_down] = 0
    
psi_array[0] = psi_old

d = B @ psi_old

for i in range(time_steps):
    psi_new = sp.linalg.spsolve(A, d)
    psi_array[i+1] = psi_new

    psi_old = psi_new
    d = B @ psi_old

#region plot_density
def extract_density(psi, N_l, N_R):
    """
    Convierte el vector interleaved en densidad física ordenada:
    L → U → D → R
    """

    def up(j): return 2*j
    def down(j): return 2*j + 1

    density = []

    # --- LEFT LEAD ---
    for j in range(N_l):
        density.append(
            np.abs(psi[up(j)])**2 + np.abs(psi[down(j)])**2
        )

    # --- UPPER ARM ---
    offset = N_l
    for j in range(N_R):
        idx = offset + j
        density.append(
            np.abs(psi[up(idx)])**2 + np.abs(psi[down(idx)])**2
        )

    # --- LOWER ARM ---
    offset = N_l + N_R
    for j in range(N_R):
        idx = offset + j
        density.append(
            np.abs(psi[up(idx)])**2 + np.abs(psi[down(idx)])**2
        )

    # --- RIGHT LEAD ---
    offset = N_l + 2*N_R
    for j in range(N_l):
        idx = offset + j
        density.append(
            np.abs(psi[up(idx)])**2 + np.abs(psi[down(idx)])**2
        )

    return np.array(density)

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

# número total de nodos físicos
N_total = N_l + 2*N_R + N_l

x = np.arange(N_total)

line, = ax.plot(x, np.zeros(N_total))

ax.set_ylim(0, 2)

def update(frame):
    psi = psi_array[frame]

    rho = extract_density(psi, N_l, N_R)

    line.set_ydata(rho)
    return (line,)

ani = FuncAnimation(fig, update, frames=len(psi_array), interval=10, blit=True)

plt.show()
#endregion plot_density

#region plot_V

# Define the sections and their respective N values
sections = [("L", N_l), ("U", N_R), ("D", N_R), ("R", N_l)]

# Initialize lists for x (cumulative indices) and V values
x_points = []
V_values = []

# Populate x_points and V_values by traversing each section
for section, N in sections:
    for i in range(N):
        x_points.append(len(x_points))  # Cumulative index (0, 1, 2, ..., size-1)
        V_values.append(t.V(section, i))  # Compute V for this section and i

# Convert to numpy arrays for plotting
x_points = np.array(x_points)
V_values = np.array(V_values)

# Plot V vs x
plt.figure(figsize=(10, 6))
plt.plot(x_points, V_values, label="Potential V")
plt.xlabel("Cumulative Index (across sections)")
plt.ylabel("Potential V")
plt.title("Potential V across the System")
plt.grid(True)
plt.legend()
plt.show()

#endregion plot_V

