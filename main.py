import tools as t
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from matplotlib.animation import FuncAnimation


N_R = t.N_R #number of points per branch of the ring
N_l = t.N_l

Ef = 4.19 #Fermi Energy in meV
theta = 12.5*np.pi 

#k= np.sqrt(2*t.m*Ef)/t.h_bar #wavevector of the initial wavefunction in the left lead
k = theta/(2*np.pi*t.R)
k=10

theta = k * (2*np.pi*t.R)
print(f"theta = {theta:.2f} rad")
print(f"k = {k:.4f} nm^-1")

time_steps = 10000

#region Solve QR
psi_up_array, psi_down_array = t.solve_QR(N_R, k, time_steps)
#endregion Solve QR

#region plot_density
t.plot_animate_psi_total(psi_up_array, psi_down_array, (2*N_l+2*N_R), k)
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

# for i in range(100):
#     k = i/100 + 100
#     psi_k_up_array, psi_k_down_array = t.solve_QR(N_R, k, time_steps)
#     t.plot_animate_psi_total(psi_k_up_array, psi_k_down_array, (2*N_l+2*N_R), k)
    

    
    

    

