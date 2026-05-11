import numpy as np

#Universal Constants:
h_bar= 1

m = 0.023  # InAs Effective Mass (dimensionless)
R = 250 #radious of the ring in nm
L_leads = 50 #length of the leads in nm
L_ring = np.pi*R #length of the branch of the ring in nm

N_l = 10 #number of points per lead:
N_R = 10 #number of points per branch of the ring

Phi = 1 / 2 # Aharonov-Bohm Phase
phi_link = Phi / 2*(N_R-1) #fase de AB agregada en cada paso espacial
phi_U = phi_link  #la fase acumulada en upper ring se suma
phi_D = -phi_link #la fase acumulada en lower(down) ring se resta
phi_L = phi_R = 0 #la fase acumulada en los leads left y right son 0 

delta_x = L_leads/(N_l-1) #space step for the leads
delta_s = L_ring/(N_R-1) #space step for the ring
dt = 1 #time step

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
def matrix_A_B_generator_single_ring(N_R):     #N_R is the numberof points per ring

    #dictionary that contains the columns assinged to each variable 
    idx = {}

    counter = 0

    #fill the dictionary
    #column of the left lead
    for i in range(N_l):
        idx[f"L{i}"] = counter
        counter+= 1
    
    #columns of ring upper arm
    for i in range(N_R):
        idx[f"U{i}"] = counter
        counter+= 1

    #column of the ring lower arm
    for i in range(N_R):
        idx[f"D{i}"] = counter
        counter+= 1

    #column of the right lead
    for i in range(N_l):
        idx[f"R{i}"] = counter
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

    row = 0

    #Fill the matrixs
    #Left leads
    for i in range(1,N_l - 1):
        A[row, idx[f"L{i-1}"]] = a(lambda_lead, phi_L)
        A[row, idx[f"L{i}"]] = b(lambda_lead, "L", i)
        A[row, idx[f"L{i+1}"]] = c(lambda_lead, phi_L)

        B[row, idx[f"L{i-1}"]] = -a(lambda_lead, phi_L)
        B[row, idx[f"L{i}"]] = b_b(lambda_lead, "L", i)
        B[row, idx[f"L{i+1}"]] = -c(lambda_lead, phi_L)

        row+=1

    #Junction equations
    # L0=U0
    A[row, idx["L0"]] = B[row, idx["L0"]] = 1
    A[row, idx["U0"]] = B[row, idx["U0"]] = -1

    row += 1
    
    # L0=D0
    A[row, idx["L0"]] = B[row, idx["L0"]] = 1
    A[row, idx["D0"]] = B[row, idx["D0"]] = -1

    row += 1

    #Current conservation
    # -(L1 - L0) + (U1 - U0) + (D1 - D0) = 0
    A[row, idx["L1"]] = B[row, idx["L1"]] = -1
    A[row, idx["L0"]] = B[row, idx["L0"]] = 1

    A[row, idx["U1"]] = B[row, idx["U1"]] = 1
    A[row, idx["U0"]] = B[row, idx["U0"]] = -1

    A[row, idx["D1"]] = B[row, idx["D1"]] = 1
    A[row, idx["D0"]] = B[row, idx["D0"]] = -1

    row += 1

    #Upper arm ring
    for i in range(1,N_R - 1):
        A[row, idx[f"U{i-1}"]] = a(lambda_ring, phi_U)
        A[row, idx[f"U{i}"]] = b(lambda_ring, "U", i)
        A[row, idx[f"U{i+1}"]] = c(lambda_ring, phi_U)

        B[row, idx[f"U{i-1}"]] = -a(lambda_ring, phi_U)
        B[row, idx[f"U{i}"]] = b_b(lambda_ring, "U", i)
        B[row, idx[f"U{i+1}"]] = -c(lambda_ring, phi_U)

        row+=1
    
    #Lower arm ring
    for i in range(1,N_R - 1):
        A[row, idx[f"D{i-1}"]] = a(lambda_ring, phi_D)
        A[row, idx[f"D{i}"]] = b(lambda_ring, "D", i)
        A[row, idx[f"D{i+1}"]] = c(lambda_ring, phi_D)

        B[row, idx[f"D{i-1}"]] = -a(lambda_ring, phi_D)
        B[row, idx[f"D{i}"]] = b_b(lambda_ring, "D", i)
        B[row, idx[f"D{i+1}"]] = -c(lambda_ring, phi_D)

        row+=1
    
    #Dirichlet boundary condition at the end of the upper arm
    A[row, idx[f"L{N_l-1}"]] = 1
    B[row, idx[f"L{N_l-1}"]] = 1
    row += 1

    #junction B equations 
    # R0=U4
    A[row, idx["R0"]] = B[row, idx["R0"]] = 1
    A[row, idx[f"U{N_R-1}"]] = B[row, idx[f"U{N_R-1}"]] = -1

    row += 1

    # R0=D4
    A[row, idx["R0"]] = B[row, idx["R0"]] = 1
    A[row, idx[f"D{N_R-1}"]] = B[row, idx[f"D{N_R-1}"]] = -1

    row += 1

    #Current conservation
    # −(Ulast​−Uprev​)−(Dlast​−Dprev​)+(R1​−R0​)=0
    A[row, idx[f"U{N_R-1}"]] = B[row, idx[f"U{N_R-1}"]] = -1
    A[row, idx[f"U{N_R-2}"]] = B[row, idx[f"U{N_R-2}"]] = 1

    A[row, idx[f"D{N_R-1}"]] = B[row, idx[f"D{N_R-1}"]] = -1
    A[row, idx[f"D{N_R-2}"]] = B[row, idx[f"D{N_R-2}"]] = 1

    A[row, idx["R1"]] = B[row, idx["R1"]] = 1
    A[row, idx["R0"]] = B[row, idx["R0"]] = -1

    row += 1

    #Right lead
    for i in range(1,N_l - 1):
        A[row, idx[f"R{i-1}"]] = a(lambda_lead, phi_R)
        A[row, idx[f"R{i}"]] = b(lambda_lead, "R", i)
        A[row, idx[f"R{i+1}"]] = c(lambda_lead, phi_R)

        B[row, idx[f"R{i-1}"]] = -a(lambda_lead, phi_R)
        B[row, idx[f"R{i}"]] = b_b(lambda_lead, "R", i)
        B[row, idx[f"R{i+1}"]] = -c(lambda_lead, phi_R)

        row+=1

    #dirithclet boundary condition at the end of the right lead
    A[row, idx[f"R{N_l-1}"]] = 1
    B[row, idx[f"R{N_l-1}"]] = 1
    row += 1

    return A,B, size


