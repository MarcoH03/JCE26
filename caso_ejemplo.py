#Define the indexing system
#We assign a unique index to every point in the system.

import numpy as np

R=5
# number of points per segment
N = 5

# helper to assign indices
idx = {}

counter = 0

# Left lead (L4 → L0)
for j in range(N):
    idx[f"L{j}"] = counter
    counter += 1

# Upper arm (U0 → U4)
for j in range(N):
    idx[f"U{j}"] = counter
    counter += 1

# Lower arm (D0 → D4)
for j in range(N):
    idx[f"D{j}"] = counter
    counter += 1

# Right lead (R0 → R4)
for j in range(N):
    idx[f"R{j}"] = counter
    counter += 1

size = counter
print("Matrix size:", size)

#The wavefunction at the current (known) time step
k=1
psi_old = np.zeros(size, dtype=complex)
for j in range(N):
    x = j
    psi_old[idx[f"L{j}"]] = np.exp(-(x-2)**2) * np.exp(1j*k*x)


#Define coefficients (Crank–Nicolson)

# parameters
m = 1.0
lambda_lead = 1j / (2*m)
lambda_ring = 1j / (2*m*R**2)

def V(x):
    return 0.0  # you can change this later

def a(lam=lambda_ring): return -lam/2
def b(Vx, lam=lambda_ring): return 1 + lam + 1j*Vx/2
def c(lam=lambda_ring): return -lam/2

#Initialize matrix A

A = np.zeros((size, size), dtype=complex)

row = 0

#Left lead interior equations (L1, L2, L3)

for j in [1, 2, 3]:
    A[row, idx[f"L{j-1}"]] = a(lambda_lead)
    A[row, idx[f"L{j}"]]   = b(j, lambda_lead)
    A[row, idx[f"L{j+1}"]] = c(lambda_lead)
    row += 1

#Junction A equations

#(A1) L0 = U0
A[row, idx["L0"]] = 1
A[row, idx["U0"]] = -1
row += 1

#(A2) L0 = D0
A[row, idx["L0"]] = 1
A[row, idx["D0"]] = -1
row += 1

#(A3) Current conservation
#−(L1−L0)+(U1−U0)+(D1−D0)=0
A[row, idx["L1"]] = -1
A[row, idx["L0"]] = 1

A[row, idx["U1"]] = 1
A[row, idx["U0"]] = -1

A[row, idx["D1"]] = 1
A[row, idx["D0"]] = -1

row += 1

#Upper arm interior (U1, U2, U3)
for j in [1, 2, 3]:
    A[row, idx[f"U{j-1}"]] = a()
    A[row, idx[f"U{j}"]]   = b(j)
    A[row, idx[f"U{j+1}"]] = c()
    row += 1

#Lower arm interior (D1, D2, D3)

for j in [1, 2, 3]:
    A[row, idx[f"D{j-1}"]] = a()
    A[row, idx[f"D{j}"]]   = b(j)
    A[row, idx[f"D{j+1}"]] = c()
    row += 1


#Junction B equations
#(B1) R0 = U4

A[row, idx["R0"]] = 1
A[row, idx["U4"]] = -1
row += 1

#(B2) R0 = D4
A[row, idx["R0"]] = 1
A[row, idx["D4"]] = -1
row += 1

#(B3) Current conservation
#−(U4−U3)−(D4−D3)+(R1−R0)=0
A[row, idx["U4"]] = -1
A[row, idx["U3"]] = 1

A[row, idx["D4"]] = -1
A[row, idx["D3"]] = 1

A[row, idx["R1"]] = 1
A[row, idx["R0"]] = -1

row += 1

#Right lead interior (R1, R2, R3)
for j in [1, 2, 3]:
    A[row, idx[f"R{j-1}"]] = a(lambda_lead)
    A[row, idx[f"R{j}"]]   = b(j, lambda_lead)
    A[row, idx[f"R{j+1}"]] = c(lambda_lead)
    row += 1

#Done
print("Total rows filled:", row)

print(A)

#Build B step by step (same structure)
#We reuse the same indexing.
B = np.zeros((size, size), dtype=complex)
row = 0
for j in [1, 2, 3]:
    lam = lambda_lead
    
    B[row, idx[f"L{j-1}"]] = +lam/2
    B[row, idx[f"L{j}"]]   = 1 - lam - 1j*V(j)/2
    B[row, idx[f"L{j+1}"]] = +lam/2
    
    row += 1

#(A1) L0 = U0
B[row, idx["L0"]] = 1
B[row, idx["U0"]] = -1
row += 1

#(A2) L0 = D0
B[row, idx["L0"]] = 1
B[row, idx["D0"]] = -1
row += 1

#(A3) Current conservation
B[row, idx["L1"]] = -1
B[row, idx["L0"]] = 1

B[row, idx["U1"]] = 1
B[row, idx["U0"]] = -1

B[row, idx["D1"]] = 1
B[row, idx["D0"]] = -1

row += 1

#Upper arm (ring → uses λ_ring)
for j in [1, 2, 3]:
    lam = lambda_ring
    
    B[row, idx[f"U{j-1}"]] = +lam/2
    B[row, idx[f"U{j}"]]   = 1 - lam - 1j*V(j)/2
    B[row, idx[f"U{j+1}"]] = +lam/2
    
    row += 1

#Lower arm
for j in [1, 2, 3]:
    lam = lambda_ring
    
    B[row, idx[f"D{j-1}"]] = +lam/2
    B[row, idx[f"D{j}"]]   = 1 - lam - 1j*V(j)/2
    B[row, idx[f"D{j+1}"]] = +lam/2
    
    row += 1

#Junction B
#(B1)
B[row, idx["R0"]] = 1
B[row, idx["U4"]] = -1
row += 1

#(B2)
B[row, idx["R0"]] = 1
B[row, idx["D4"]] = -1
row += 1

#(B3)
B[row, idx["U4"]] = -1
B[row, idx["U3"]] = 1

B[row, idx["D4"]] = -1
B[row, idx["D3"]] = 1

B[row, idx["R1"]] = 1
B[row, idx["R0"]] = -1

row += 1

#Right lead interior
for j in [1, 2, 3]:
    lam = lambda_lead
    
    B[row, idx[f"R{j-1}"]] = +lam/2
    B[row, idx[f"R{j}"]]   = 1 - lam - 1j*V(j)/2
    B[row, idx[f"R{j+1}"]] = +lam/2
    
    row += 1

#Final step: compute RHS
d = B @ psi_old

#Then solve CN step
psi_new = np.linalg.solve(A, d)