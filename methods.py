import numpy as np 
import math
from scipy.sparse import spdiags
import scipy.linalg as sl
from scipy.integrate import solve_ivp
import scipy.sparse

count = 0

def counter(x):
    global count
    count = count + 1
    return x

def matmuls(psi, w, dx, A, Dx, Dy, nu):

    a = Dx._mul_vector(psi)
    b = Dy._mul_vector(w)
    c = (0.25 / math.pow(dx, 2)) * np.multiply(a, b)
    
    a = Dx._mul_vector(w)
    b = Dy._mul_vector(psi)
    d = (0.25 / math.pow(dx, 2)) * np.multiply(b, a)

    e = A._mul_vector(w)

    return (nu/math.pow(dx, 2)) * e - c + d

def inv(tspan, end_time, w_init, delta_x, A, Dx, Dy, diff_coef):

    print("\nDoining the Inverse solver\n")
    
    A_inv = sl.inv(A.toarray())

    def stepper(t, w, dx, A, Dx, Dy, nu): 

        psi = math.pow(dx, 2) * np.matmul(A_inv, w)
        psi = psi - np.min(psi)

        return matmuls(psi, w, dx, A, Dx, Dy, nu)

    sol = solve_ivp(stepper, (0, end_time), y0= w_init, method='RK45', t_eval= tspan, args = (delta_x, A, Dx, Dy, diff_coef), first_step = 0.5)

    return sol

def fft_solver(tspan, end_time, w_init, delta_x, A, Dx, Dy, diff_coef, n, N, L):

    print("\nDoining the FFT solver\n")

    def get_K3():
        indices_a = [i for i in range(int(n/2))]
        indices_b = [n/2 - i for i in range(int(n/2))]
        k = (2 * math.pi / L) * np.asarray(indices_a + indices_b)

        k[0] = 1e-5
        KX, KY = np.meshgrid(k, k, indexing = 'ij')
        K3 = np.power((np.power(KX, 2) + np.power(KY, 2)), -1)

        return K3

    psi_return = []
    omega_return = []
    K3 = get_K3()
    def stepper(t, w, dx, A, Dx, Dy, nu):     

        psi = np.multiply(-np.fft.fft2(np.reshape(w, (n, n))), K3) 
        psi = np.reshape(np.real(np.fft.ifft2(psi)), N)  
        psi = psi - np.min(psi)
        
        psi_return.append(psi)
        omega_return.append(w)

        return matmuls(psi, w, dx, A, Dx, Dy, nu)

    sol = solve_ivp(stepper, (0, end_time), y0= w_init, method='RK45', args = (delta_x, A, Dx, Dy, diff_coef), first_step = 0.5, t_eval=tspan)

    return sol, psi_return, omega_return

def cg_solver(tspan, end_time, w_init, delta_x, A, Dx, Dy, diff_coef):

    print("\nDoining the CG solver\n")
    
    counter_list = []

    def stepper(t, w, dx, A, Dx, Dy, nu):
        global count
        cb = lambda x: counter(x)
        
        count = 0
        cgs, _ = scipy.sparse.linalg.cg(A, w, tol = 1e-5, maxiter=200, callback=cb)
        counter_list.append(count)
        psi = math.pow(dx, 2) * cgs
        psi = psi - np.min(psi)

        return matmuls(psi, w, dx, A, Dx, Dy, nu)

    sol = solve_ivp(stepper, (0, end_time), y0= w_init, method='RK45', t_eval= tspan, args = (delta_x, A, Dx, Dy, diff_coef), first_step = 0.5)

    return sol, counter_list

def gmres_solver(tspan, end_time, w_init, delta_x, A, Dx, Dy, diff_coef):

    print("\nDoining the GMRES solver\n")
    
    counter_list = []

    def stepper(t, w, dx, A, Dx, Dy, nu):
        global count
        cb = lambda x: counter(x)
        
        count = 0
        cgs, _ = scipy.sparse.linalg.gmres(A, w, tol = 1e-5, maxiter=1000, callback=cb)
        counter_list.append(count)
        psi = math.pow(dx, 2) * cgs
        psi = psi - np.min(psi)
        
        return matmuls(psi, w, dx, A, Dx, Dy, nu)

    sol = solve_ivp(stepper, (0, end_time), y0= w_init, method='RK45', t_eval= tspan, args = (delta_x, A, Dx, Dy, diff_coef), first_step = 0.5)

    return sol, counter_list

def bicgstab_solver(tspan, end_time, w_init, delta_x, A, Dx, Dy, diff_coef):

    print("\nDoining the BICGSTAB solver\n")
    
    counter_list = []

    def stepper(t, w, dx, A, Dx, Dy, nu):
        global count
        cb = lambda x: counter(x)
        
        count = 0
        cgs, _ = scipy.sparse.linalg.bicgstab(A, w, tol = 1e-5, maxiter=1000, callback=cb)
        counter_list.append(count)
        psi = math.pow(dx, 2) * cgs
        psi = psi - np.min(psi)

        return matmuls(psi, w, dx, A, Dx, Dy, nu)

    sol = solve_ivp(stepper, (0, end_time), y0= w_init, method='RK45', t_eval= tspan, args = (delta_x, A, Dx, Dy, diff_coef), first_step = 0.5)

    return sol, counter_list