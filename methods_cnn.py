import numpy as np 
import math
from scipy.sparse import spdiags
from scipy.integrate import solve_ivp
import scipy.sparse
from scipy.sparse.linalg import cg, gmres, bicgstab, lgmres

import sys

from give_IG import MGCNN


count = 0

def get_solver(solver):
    return getattr(sys.modules[__name__], solver)
    

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

def iterative_solver(grid_size, tspan, end_time, w_init, delta_x, A, Dx, Dy, diff_coef, solver_name='lgmres', guess = False):

    print("\nUsing {} solver\n".format(solver_name))
    solver = get_solver(solver_name) 
    if guess:
        #IG_generator = MGCNN('MGv2_1', grid_size, 10, 4, 10, 2)
        IG_generator = MGCNN('MGv1', grid_size, 10, 3, 5)
    
    counter_list = []
    A_inv = scipy.sparse.linalg.inv(scipy.sparse.diags(A.diagonal(), 0))

    def stepper(t, w, dx, A, Dx, Dy, nu):
        global count
        cb = lambda x: counter(x)
        
        count = 0
        if guess:
            inital_guess = (IG_generator.get_guess(w)) 
        else:
            inital_guess = None
        psi, _ = solver(A, w, x0 = inital_guess, tol = 1e-5, callback=cb, M=A_inv)
        counter_list.append(count)
        psi = psi - np.min(psi)

        return matmuls(psi, w, dx, A, Dx, Dy, nu)

    sol = solve_ivp(stepper, (0, end_time), y0= w_init, method='RK45', args = (delta_x, A, Dx, Dy, diff_coef), first_step = 0.5, t_eval=tspan)

    return sol, counter_list
