import tools as t
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

N_R = 10 #number of points per branch of the ring
N_l = 10

A, B, size = t.matrix_A_B_generator_single_ring(N_R)
A = sp.csr_matrix(A)
B = sp.csr_matrix(B)
time_steps = 100


psi_array = np.empty(time_steps+1, dtype=object)

#The wavefunction at the current (known) time step
k=1
psi_old = np.zeros(size, dtype=complex)
for j in range(N_l):
    x = j
    psi_old[j] = np.exp(-(x-2)**2) * np.exp(1j*k*x)

psi_array[0] = psi_old

d = B @ psi_old

for i in range(time_steps):
    psi_new = sp.linalg.spsolve(A, d)
    psi_array[i+1] = psi_new

    psi_old = psi_new
    d = B @ psi_old


xpoints = np.array(list(range(size)))
ypoints = np.abs(psi_array[0])**2

plt.plot(xpoints, ypoints)
plt.show()

from matplotlib.animation import FuncAnimation

xpoints = np.arange(size)
fig, ax = plt.subplots()
line, = ax.plot(xpoints, np.abs(psi_array[0])**2)
ax.set_xlabel("index")
ax.set_ylabel("|psi|^2")

def update(frame):
    line.set_ydata(np.abs(psi_array[frame])**2)
    return (line,)

ani = FuncAnimation(fig, update, frames=len(psi_array), interval=200, blit=True)
plt.show()




