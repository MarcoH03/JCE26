import numpy as np

m = 1 # effective mass of the particle 
R = 250 #radious of the ring in nm

lambda_lead = 1j/(2*m)
lambda_ring = 1j/(2*m*R**2)

#number of points per lead:
N_l = 10

#potential function for the entire system
def V(x):
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
    def a(lam): return -lam/2
    def b(lam, x): return 1+ lam+ 1j*V(x)/2
    def c(lam): return -lam/2

    #elements of the diagonal for matrix B
    def b_b(lam,x): return 1- lam - 1j*V(x)/2

    #initialize matrix A and B
    A = np.zeros((size,size), dtype=complex)
    B = np.zeros((size,size), dtype=complex)

    row = 0

    #Fill the matrixs
    #Left leads
    for i in range(1,N_l - 1):
        A[row, idx[f"L{i-1}"]] = a(lambda_lead)
        A[row, idx[f"L{i}"]] = b(lambda_lead, i)
        A[row, idx[f"L{i+1}"]] = c(lambda_lead)

        B[row, idx[f"L{i-1}"]] = -a(lambda_lead)
        B[row, idx[f"L{i}"]] = b_b(lambda_lead, i)
        B[row, idx[f"L{i+1}"]] = -c(lambda_lead)

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
        A[row, idx[f"U{i-1}"]] = a(lambda_ring)
        A[row, idx[f"U{i}"]] = b(lambda_ring, i)
        A[row, idx[f"U{i+1}"]] = c(lambda_ring)

        B[row, idx[f"U{i-1}"]] = -a(lambda_ring)
        B[row, idx[f"U{i}"]] = b_b(lambda_ring, i)
        B[row, idx[f"U{i+1}"]] = -c(lambda_ring)

        row+=1
    
    #Lower arm ring
    for i in range(1,N_R - 1):
        A[row, idx[f"D{i-1}"]] = a(lambda_ring)
        A[row, idx[f"D{i}"]] = b(lambda_ring, i)
        A[row, idx[f"D{i+1}"]] = c(lambda_ring)

        B[row, idx[f"D{i-1}"]] = -a(lambda_ring)
        B[row, idx[f"D{i}"]] = b_b(lambda_ring, i)
        B[row, idx[f"D{i+1}"]] = -c(lambda_ring)

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
        A[row, idx[f"R{i-1}"]] = a(lambda_lead)
        A[row, idx[f"R{i}"]] = b(lambda_lead, i)
        A[row, idx[f"R{i+1}"]] = c(lambda_lead)

        B[row, idx[f"R{i-1}"]] = -a(lambda_lead)
        B[row, idx[f"R{i}"]] = b_b(lambda_lead, i)
        B[row, idx[f"R{i+1}"]] = -c(lambda_lead)

        row+=1

    #dirithclet boundary condition at the end of the right lead
    A[row, idx[f"R{N_l-1}"]] = 1
    B[row, idx[f"R{N_l-1}"]] = 1
    row += 1

    return A,B, size


