import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



#Universal Constants:
h_bar = 0.658212  # meV*ps
m_e = 5.68563e3  # meV*ps^2/nm^2

m = 0.023 * m_e  # InAs Effective Mass (dimensionless)
R = 250 #radious of the ring in nm
L_leads = 100 #length of the leads in nm
L_ring = np.pi*R #length of the branch of the ring in nm

N_l = 5 #number of points per lead:
N_R = 5 #number of points per branch of the ring

delta_x = L_leads/(N_l-1) #space step for the leads
delta_s = L_ring/(N_R-1) #space step for the ring
dt = 5  # ps

Phi = 1 / 2 # Aharonov-Bohm Phase
phi_link = Phi / 2*(N_R-1) #fase de AB agregada en cada paso espacial
phi_U = phi_link  #la fase AB acumulada en upper ring se suma
phi_D = -phi_link #la fase AB acumulada en lower(down) ring se resta
phi_L = phi_R = 0 #la fase AB acumulada en los leads left y right son 0 

Phi_so = 0.1 #Spin-Orbit phase
alpha = 20    # Rashba Parameter in meV.nm
phi_so_link = theta_R = m * alpha * delta_s / h_bar**2

lambda_lead = 1j*h_bar*dt/(2*m*delta_x**2)
lambda_ring = 1j*h_bar/(2*m*delta_s**2)

#constants for the potential function
V0_L = 0
Ux_L = 0.01 
Uy_L = 0.01

V0_U = 0
Ux_U = 0.01
Uy_U = 0.01

V0_R = 0
Ux_R = 0.01
Uy_R = 0.01

s0_L = 0.8 * L_leads
s0_U = 0.5 * L_ring
s0_R = 0.2 * L_leads    



#potential function for the entire system
def V(section, i):
    
    # QPC left lead
    if section == "L":
        s = i*delta_x
        omega_y = np.sqrt(2*Uy_L/m)

        E_trans = h_bar * omega_y * (0.5)

        return V0_L - Ux_L*(s-s0_L)**2 + E_trans

    # QPC upper arm
    elif section == "U":
        s = i*delta_s
        omega_y = np.sqrt(2*Uy_U/m)

        E_trans = h_bar * omega_y * (0.5)

        return V0_U - Ux_U*(s-s0_U)**2 + E_trans
    
    # lower arm
    elif section == "D":
        return 0

    # right lead
    elif section == "R":
        s = i*delta_x
        omega_y = np.sqrt(2*Uy_R/m)

        E_trans = h_bar * omega_y * (0.5)

        return V0_R - Ux_R*(s-s0_R)**2 + E_trans
    return 0

#creates the matrix A and B for the system
def matrix_A_B_generator_single_ring(N_R, k):     #N_R is the numberof points per ring

    #dictionary that contains the columns assinged to each variable 
    idx = {}

    counter = 0

    #fill the dictionary
    #column of the left lead
    for i in range(N_l):
        idx[f"L{i}_up"] = counter
        counter+= 1
        
        idx[f"L{i}_down"] = counter
        counter+= 1
    
    #columns of ring upper arm
    for i in range(N_R):
        idx[f"U{i}_up"] = counter
        counter+= 1
        
        idx[f"U{i}_down"] = counter
        counter+= 1

    #column of the ring lower arm
    for i in range(N_R):
        idx[f"D{i}_up"] = counter
        counter+= 1 
        
        idx[f"D{i}_down"] = counter
        counter+= 1

    #column of the right lead
    for i in range(N_l):
        idx[f"R{i}_up"] = counter
        counter+= 1

        idx[f"R{i}_down"] = counter
        counter+= 1

    size = counter 

    #elements of the diagonal for matrix A 
    def a(lam, phi): return -lam/2* np.exp(-1j*phi)
    def b(lam, section, x): return 1+ lam+ 1j*V(section, x)/2
    def c(lam, phi): return -lam/2* np.exp(+1j*phi)

    #elements of the diagonal for matrix B
    def b_b(lam, section, x): return 1- lam - 1j*V(section, x)/2
    #initialize matrix A and B
    A = np.zeros((size,size), dtype=complex)
    B = np.zeros((size,size), dtype=complex)
    
    #Matriz que determina la rotacion de spin en cada enlace debido al efecto Rashba 
    def U_rashba(theta):
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ], dtype=complex)

    row_up = 0
    row_down = row_up + 1

    #Fill the matrixs
    #Left leads
    for i in range(1,N_l - 1):
        #Spin Up
        phi_total = phi_L
        A[row_up, idx[f"L{i-1}_up"]] += a(lambda_lead, phi_total)*U_rashba(phi_so_link)[0,0]
        A[row_up, idx[f"L{i-1}_down"]] += a(lambda_lead, phi_total)*U_rashba(phi_so_link)[0,1]
        
        A[row_up, idx[f"L{i}_up"]] += b(lambda_lead, "L", i)
        
        A[row_up, idx[f"L{i+1}_up"]] += c(lambda_lead, phi_total)*U_rashba(phi_so_link)[0,0]
        A[row_up, idx[f"L{i+1}_down"]] += c(lambda_lead, phi_total)*U_rashba(phi_so_link)[0,1]


        B[row_up, idx[f"L{i-1}_up"]] = -a(lambda_lead, phi_total)*U_rashba(phi_so_link)[0,0]
        B[row_up, idx[f"L{i-1}_down"]] = -a(lambda_lead, phi_total)*U_rashba(phi_so_link)[0,1]
        
        B[row_up, idx[f"L{i}_up"]] = b_b(lambda_lead, "L", i)
        
        B[row_up, idx[f"L{i+1}_up"]] = -c(lambda_lead, phi_total)*U_rashba(phi_so_link)[0,0]
        B[row_up, idx[f"L{i+1}_down"]] = -c(lambda_lead, phi_total)*U_rashba(phi_so_link)[0,1]
        
        
        #Spin Down
        phi_total = phi_L - phi_so_link #la fase total incluyendo AB y AC
        A[row_down, idx[f"L{i-1}_down"]] = a(lambda_lead, phi_total)*U_rashba(phi_so_link)[1,1]
        A[row_down, idx[f"L{i-1}_up"]] = a(lambda_lead, phi_total)*U_rashba(phi_so_link)[1,0]
        
        A[row_down, idx[f"L{i}_down"]] = b(lambda_lead, "L", i)
        
        A[row_down, idx[f"L{i+1}_down"]] = c(lambda_lead, phi_total)*U_rashba(phi_so_link)[1,1]
        A[row_down, idx[f"L{i+1}_up"]] = c(lambda_lead, phi_total)*U_rashba(phi_so_link)[1,0]

        B[row_down, idx[f"L{i-1}_down"]] = -a(lambda_lead, phi_total)*U_rashba(phi_so_link)[1,1]
        B[row_down, idx[f"L{i-1}_up"]] = -a(lambda_lead, phi_total)*U_rashba(phi_so_link)[1,0]
        
        B[row_down, idx[f"L{i}_down"]] = b_b(lambda_lead, "L", i)
        
        B[row_down, idx[f"L{i+1}_down"]] = -c(lambda_lead, phi_total)*U_rashba(phi_so_link)[1,1]
        B[row_down, idx[f"L{i+1}_up"]] = -c(lambda_lead, phi_total)*U_rashba(phi_so_link)[1,0]

        row_up+=2
        row_down = row_up + 1

    #Junction equations
    # L0=U0
    #Spin Up
    A[row_up, idx["L0_up"]] = B[row_up, idx["L0_up"]] = 1
    A[row_up, idx["U0_up"]] = B[row_up, idx["U0_up"]] = -1
    
    #Spin Down
    A[row_down, idx["L0_down"]] = B[row_down, idx["L0_down"]] = 1
    A[row_down, idx["U0_down"]] = B[row_down, idx["U0_down"]] = -1
    
    row_up += 2
    row_down = row_up + 1

    # L0=D0
    #Spin Up
    A[row_up, idx["L0_up"]] = B[row_up, idx["L0_up"]] = 1
    A[row_up, idx["D0_up"]] = B[row_up, idx["D0_up"]] = -1
    
    #Spin Down
    A[row_down, idx["L0_down"]] = B[row_down, idx["L0_down"]] = 1
    A[row_down, idx["D0_down"]] = B[row_down, idx["D0_down"]] = -1
    
    row_up += 2
    row_down = row_up + 1


    #Current conservation
    # -(L1 - L0) + (U1 - U0) + (D1 - D0) = 0
    #Spin Up
    A[row_up, idx["L1_up"]] = B[row_up, idx["L1_up"]] = -1
    A[row_up, idx["L0_up"]] = B[row_up, idx["L0_up"]] = 1

    A[row_up, idx["U1_up"]] = B[row_up, idx["U1_up"]] = 1
    A[row_up, idx["U0_up"]] = B[row_up, idx["U0_up"]] = -1
    
    A[row_up, idx["D1_up"]] = B[row_up, idx["D1_up"]] = 1
    A[row_up, idx["D0_up"]] = B[row_up, idx["D0_up"]] = -1
    
    #Spin Down
    A[row_down, idx["L1_down"]] = B[row_down, idx["L1_down"]] = -1
    A[row_down, idx["L0_down"]] = B[row_down, idx["L0_down"]] = 1

    A[row_down, idx["U1_down"]] = B[row_down, idx["U1_down"]] = 1
    A[row_down, idx["U0_down"]] = B[row_down, idx["U0_down"]] = -1
    
    A[row_down, idx["D1_down"]] = B[row_down, idx["D1_down"]] = 1
    A[row_down, idx["D0_down"]] = B[row_down, idx["D0_down"]] = -1
    
    row_up += 2
    row_down = row_up + 1

    #Upper arm ring
    for i in range(1,N_R - 1):
        #Spin Up
        phi_total = phi_U #la fase AB
        A[row_up, idx[f"U{i-1}_up"]] = a(lambda_ring, phi_total)*U_rashba(phi_so_link)[0,0]
        A[row_up, idx[f"U{i-1}_down"]] = a(lambda_ring, phi_total)*U_rashba(phi_so_link)[0,1]
        
        A[row_up, idx[f"U{i}_up"]] = b(lambda_ring, "U", i)
        
        A[row_up, idx[f"U{i+1}_up"]] = c(lambda_ring, phi_total)*U_rashba(phi_so_link)[0,0]
        A[row_up, idx[f"U{i+1}_down"]] = c(lambda_ring, phi_total)*U_rashba(phi_so_link)[0,1]

        B[row_up, idx[f"U{i-1}_up"]] = -a(lambda_ring, phi_total)*U_rashba(phi_so_link)[0,0]
        B[row_up, idx[f"U{i-1}_down"]] = -a(lambda_ring, phi_total)*U_rashba(phi_so_link)[0,1]
        
        B[row_up, idx[f"U{i}_up"]] = b_b(lambda_ring, "U", i)
        
        B[row_up, idx[f"U{i+1}_up"]] = -c(lambda_ring, phi_total)*U_rashba(phi_so_link)[0,0]
        B[row_up, idx[f"U{i+1}_down"]] = -c(lambda_ring, phi_total)*U_rashba(phi_so_link)[0,1]
        
        #Spin Down
        phi_total = phi_U - phi_so_link #la fase total incluyendo AB y AC
        A[row_down, idx[f"U{i-1}_down"]] = a(lambda_ring, phi_total)*U_rashba(phi_so_link)[1,1]
        A[row_down, idx[f"U{i-1}_up"]] = a(lambda_ring, phi_total)*U_rashba(phi_so_link)[1,0]
        
        A[row_down, idx[f"U{i}_down"]] = b(lambda_ring, "U", i)
        
        A[row_down, idx[f"U{i+1}_down"]] = c(lambda_ring, phi_total)*U_rashba(phi_so_link)[1,1]
        A[row_down, idx[f"U{i+1}_up"]] = c(lambda_ring, phi_total)*U_rashba(phi_so_link)[1,0]

        B[row_down, idx[f"U{i-1}_down"]] = -a(lambda_ring, phi_total)*U_rashba(phi_so_link)[1,1]
        B[row_down, idx[f"U{i-1}_up"]] = -a(lambda_ring, phi_total)*U_rashba(phi_so_link)[1,0]
        
        B[row_down, idx[f"U{i}_down"]] = b_b(lambda_ring, "U", i)
        
        B[row_down, idx[f"U{i+1}_down"]] = -c(lambda_ring, phi_total)*U_rashba(phi_so_link)[1,1]
        B[row_down, idx[f"U{i+1}_up"]] = -c(lambda_ring, phi_total)*U_rashba(phi_so_link)[1,0]
        row_up += 2
        row_down = row_up + 1
    
    #Lower arm ring
    for i in range(1,N_R - 1):
        #Spin Up
        phi_total = phi_D  #la fase AB
        A[row_up, idx[f"D{i-1}_up"]] = a(lambda_ring, phi_total)*U_rashba(phi_so_link)[0,0]
        A[row_up, idx[f"D{i-1}_down"]] = a(lambda_ring, phi_total)*U_rashba(phi_so_link)[0,1]
        
        A[row_up, idx[f"D{i}_up"]] = b(lambda_ring, "D", i)
        
        A[row_up, idx[f"D{i+1}_up"]] = c(lambda_ring, phi_total)*U_rashba(phi_so_link)[0,0]
        A[row_up, idx[f"D{i+1}_down"]] = c(lambda_ring, phi_total)*U_rashba(phi_so_link)[0,1]

        B[row_up, idx[f"D{i-1}_up"]] = -a(lambda_ring, phi_total)*U_rashba(phi_so_link)[0,0]
        B[row_up, idx[f"D{i-1}_down"]] = -a(lambda_ring, phi_total)*U_rashba(phi_so_link)[0,1]
        
        B[row_up, idx[f"D{i}_up"]] = b_b(lambda_ring, "D", i)
        
        B[row_up, idx[f"D{i+1}_up"]] = -c(lambda_ring, phi_total)*U_rashba(phi_so_link)[0,0]
        B[row_up, idx[f"D{i+1}_down"]] = -c(lambda_ring, phi_total)*U_rashba(phi_so_link)[0,1]
        
        #Spin Down
        phi_total = phi_D #la fase AB
        A[row_down, idx[f"D{i-1}_down"]] = a(lambda_ring, phi_total)*U_rashba(phi_so_link)[1,1]
        A[row_down, idx[f"D{i-1}_up"]] = a(lambda_ring, phi_total)*U_rashba(phi_so_link)[1,0]
        
        A[row_down, idx[f"D{i}_down"]] = b(lambda_ring, "D", i)
        
        A[row_down, idx[f"D{i+1}_down"]] = c(lambda_ring, phi_total)*U_rashba(phi_so_link)[1,1]
        A[row_down, idx[f"D{i+1}_up"]] = c(lambda_ring, phi_total)*U_rashba(phi_so_link)[1,0]

        B[row_down, idx[f"D{i-1}_down"]] = -a(lambda_ring, phi_total)*U_rashba(phi_so_link)[1,1]
        B[row_down, idx[f"D{i-1}_up"]] = -a(lambda_ring, phi_total)*U_rashba(phi_so_link)[1,0]
        
        B[row_down, idx[f"D{i}_down"]] = b_b(lambda_ring, "D", i)
        
        B[row_down, idx[f"D{i+1}_down"]] = -c(lambda_ring, phi_total)*U_rashba(phi_so_link)[1,1]
        B[row_down, idx[f"D{i+1}_up"]] = -c(lambda_ring, phi_total)*U_rashba(phi_so_link)[1,0]
        
        row_up += 2
        row_down = row_up + 1
    
    #Dirichlet boundary condition at the end of the upper arm
    #Spin Up
    A[row_up, idx[f"L{N_l-1}_up"]] = 1
    B[row_up, idx[f"L{N_l-1}_up"]] = 1
    row_up += 2
    
    #Spin Down
    A[row_down, idx[f"L{N_l-1}_down"]] = 1
    B[row_down, idx[f"L{N_l-1}_down"]] = 1
    row_down += 2


    #junction B equations 
    # R0=U4
    #Spin Up
    A[row_up, idx["R0_up"]] = B[row_up, idx["R0_up"]] = 1
    A[row_up, idx[f"U{N_R-1}_up"]] = B[row_up, idx[f"U{N_R-1}_up"]] = -1

    row_up += 2
    
    #Spin Down
    A[row_down, idx["R0_down"]] = B[row_down, idx["R0_down"]] = 1
    A[row_down, idx[f"U{N_R-1}_down"]] = B[row_down, idx[f"U{N_R-1}_down"]] = -1
    row_down += 2

    # R0=D4
    #Spin Up
    A[row_up, idx["R0_up"]] = B[row_up, idx["R0_up"]] = 1
    A[row_up, idx[f"D{N_R-1}_up"]] = B[row_up, idx[f"D{N_R-1}_up"]] = -1

    row_up += 2
    
    #Spin Down
    A[row_down, idx["R0_down"]] = B[row_down, idx["R0_down"]] = 1
    A[row_down, idx[f"D{N_R-1}_down"]] = B[row_down, idx[f"D{N_R-1}_down"]] = -1

    row_down += 2

    #Current conservation
    # −(Ulast​−Uprev​)−(Dlast​−Dprev​)+(R1​−R0​)=0
    #Spin Up
    A[row_up, idx[f"U{N_R-1}_up"]] = B[row_up, idx[f"U{N_R-1}_up"]] = -1
    A[row_up, idx[f"U{N_R-2}_up"]] = B[row_up, idx[f"U{N_R-2}_up"]] = 1

    A[row_up, idx[f"D{N_R-1}_up"]] = B[row_up, idx[f"D{N_R-1}_up"]] = -1
    A[row_up, idx[f"D{N_R-2}_up"]] = B[row_up, idx[f"D{N_R-2}_up"]] = 1
    
    A[row_up, idx["R1_up"]] = B[row_up, idx["R1_up"]] = 1
    A[row_up, idx["R0_up"]] = B[row_up, idx["R0_up"]] = -1

    row_up += 2
    
    #Spin Down
    A[row_down, idx[f"U{N_R-1}_down"]] = B[row_down, idx[f"U{N_R-1}_down"]] = -1
    A[row_down, idx[f"U{N_R-2}_down"]] = B[row_down, idx[f"U{N_R-2}_down"]] = 1
    
    A[row_down, idx[f"D{N_R-1}_down"]] = B[row_down, idx[f"D{N_R-1}_down"]] = -1
    A[row_down, idx[f"D{N_R-2}_down"]] = B[row_down, idx[f"D{N_R-2}_down"]] = 1

    A[row_down, idx["R1_down"]] = B[row_down, idx["R1_down"]] = 1
    A[row_down, idx["R0_down"]] = B[row_down, idx["R0_down"]] = -1

    row_down += 2

    #Right lead
    for i in range(1,N_l - 1):
        #Spin Up
        phi_total = phi_R #la fase AB
        A[row_up, idx[f"R{i-1}_up"]] = a(lambda_lead, phi_total)*U_rashba(phi_so_link)[0,0]
        A[row_up, idx[f"R{i-1}_down"]] = a(lambda_lead, phi_total)*U_rashba(phi_so_link)[0,1]
        
        A[row_up, idx[f"R{i}_up"]] = b(lambda_lead, "R", i)
        
        A[row_up, idx[f"R{i+1}_up"]] = c(lambda_lead, phi_total)*U_rashba(phi_so_link)[0,0]
        A[row_up, idx[f"R{i+1}_down"]] = c(lambda_lead, phi_total)*U_rashba(phi_so_link)[0,1]

        B[row_up, idx[f"R{i-1}_up"]] = -a(lambda_lead, phi_total)*U_rashba(phi_so_link)[0,0]
        B[row_up, idx[f"R{i-1}_down"]] = -a(lambda_lead, phi_total)*U_rashba(phi_so_link)[0,1]
        
        B[row_up, idx[f"R{i}_up"]] = b_b(lambda_lead, "R", i)
        
        B[row_up, idx[f"R{i+1}_up"]] = -c(lambda_lead, phi_total)*U_rashba(phi_so_link)[0,0]
        B[row_up, idx[f"R{i+1}_down"]] = -c(lambda_lead, phi_total)*U_rashba(phi_so_link)[0,1]

        
        #Spin Down
        phi_total = phi_R - phi_so_link
        A[row_down, idx[f"R{i-1}_down"]] = a(lambda_lead, phi_total)*U_rashba(phi_so_link)[1,1]
        A[row_down, idx[f"R{i-1}_up"]] = a(lambda_lead, phi_total)*U_rashba(phi_so_link)[1,0]
        
        A[row_down, idx[f"R{i}_down"]] = b(lambda_lead, "R", i)
        
        A[row_down, idx[f"R{i+1}_down"]] = c(lambda_lead, phi_total)*U_rashba(phi_so_link)[1,1]
        A[row_down, idx[f"R{i+1}_up"]] = c(lambda_lead, phi_total)*U_rashba(phi_so_link)[1,0]

        B[row_down, idx[f"R{i-1}_down"]] = -a(lambda_lead, phi_total)*U_rashba(phi_so_link)[1,1]
        B[row_down, idx[f"R{i-1}_up"]] = -a(lambda_lead, phi_total)*U_rashba(phi_so_link)[1,0]
        
        B[row_down, idx[f"R{i}_down"]] = b_b(lambda_lead, "R", i)
        
        B[row_down, idx[f"R{i+1}_down"]] = -c(lambda_lead, phi_total)*U_rashba(phi_so_link)[1,1]
        B[row_down, idx[f"R{i+1}_up"]] = -c(lambda_lead, phi_total)*U_rashba(phi_so_link)[1,0]
        
        row_up += 2
        row_down = row_up + 1

    phase = np.exp(1j*k*delta_x)

    #Spin Up
    A[row_up, idx[f"R{N_l-1}_up"]] = 1
    A[row_up, idx[f"R{N_l-2}_up"]] = -phase

    row_up += 2

    #Spin Down
    A[row_down, idx[f"R{N_l-1}_down"]] = 1
    A[row_down, idx[f"R{N_l-2}_down"]] = -phase

    row_down += 2

    return A,B, size

def solve_QR(N_R, k, time_steps):
    A, B, size = matrix_A_B_generator_single_ring(N_R, k)
    A = sp.csr_matrix(A)
    B = sp.csr_matrix(B)
        
    psi_array = np.empty(time_steps+1, dtype=object)
    
    #The wavefunction at the current (known) time step
    psi_old = np.zeros(size, dtype=complex)

    #fill initial wave packet with only spin up component 
    for j in range(N_l):
        x = j

        idx_up = 2*j
        idx_down = 2*j + 1

        psi_old[idx_up] = np.exp(-(x-2)**2) * np.exp(1j*k*x)
        psi_old[idx_down] = 0
        
    psi_array[0] = psi_old

    d = B @ psi_old

    #Solve the sistem for all times in time_steps
    for i in range(time_steps):
        psi_new = sp.linalg.spsolve(A, d)
        psi_array[i+1] = psi_new

        psi_old = psi_new
        d = B @ psi_old
        
    #separate the wave functions in up and down arrays
    n_t = time_steps + 1
    n_half = size // 2

    psi_up_array = np.empty((n_t, n_half), dtype=complex)
    psi_down_array = np.empty((n_t, n_half), dtype=complex)

    for j in range(n_t):
        psi = psi_array[j]
        psi_up_array[j, :] = psi[0::2]
        psi_down_array[j, :] = psi[1::2]
            
    return psi_up_array, psi_down_array
        
def plot_animate_psi_total(psi_up_array, psi_down_array, N_total, k):
    fig, ax = plt.subplots()
    x = np.arange(N_total)
    
    line, = ax.plot(x, np.zeros(N_total))

    ax.set_ylim(0, 2)
    
    def update(frame):
        psi = np.abs(psi_up_array[frame])**2 + np.abs(psi_down_array[frame])**2
        
        ax.set_title(f"{k}")
        line.set_ydata(psi)
        return (line,)

    ani = FuncAnimation(fig, update, frames=len(psi_up_array), interval=1, blit=True)

    plt.show()
    
