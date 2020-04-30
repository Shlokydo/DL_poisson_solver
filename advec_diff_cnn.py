import numpy as np 
import math
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
import scipy.linalg as sl
from scipy.integrate import solve_ivp
import scipy.sparse
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from methods_cnn import iterative_solver

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--order", '-o', default = 2, type = int, choices = [2, 4], help = "Order of the discretization scheme")
parser.add_argument("--video", '-v', default = 0, type = int, choices = [1, 0], help = "Whether to save video file or not")
parser.add_argument("--method", '-m', default = 'lgmres', type = str, choices = ['cg', 'lgmres', 'bicgstab', 'gmres'], help = "Solver to use")
parser.add_argument("--guess", '-g', default = False, action='store_true', help = "Whether to use inital guess generated from the network")
args = parser.parse_args()

#Set the number of grid points
grid_points = 64 * 2
#Set the domain lenght
domain_length = 20
#Set the end time
end_time = 1
#Set the diffusion coefficient
diff_coef = 0.001

tspan = np.arange(0, end_time, 0.10)
N = int(math.pow(grid_points, 2))

x2 = np.linspace(int(-domain_length/2), int(domain_length/2), grid_points + 1)
x = x2[0:grid_points]
delta_x = x[1] - x[0]
y = x
X, Y = np.meshgrid(x, y, indexing = 'ij')

w_init = np.exp(-2 * np.power(X, 2) - (np.power(Y, 2) / 10)) 
#w_init = np.exp(-2 * np.power(X - 3, 2) - (np.power(Y, 2) / 10)) - np.exp(-1 * np.power(X + 4, 2) / 20 - 2 * (np.power(Y - 2, 2))) + np.exp(-1 * np.power(X + 5, 2) / 20 - 2 * (np.power(Y - 8, 2)))
#w_init = 0.1 * (X + Y) * w_init
#w_init = (np.sin(X) + np.tanh(X * Y))
# w_init = np.exp(-0.25 * np.power(X-5, 2) - 2 * (np.power(Y, 2))) - np.exp(-0.25 * np.power(X+5, 2) - 2 * (np.power(Y, 2)))
#w_init = w_init * ((np.power(0.1 * X, 2) + np.power(0.1 * Y, 2)))
#w_init = w_init * (np.power(0.1 * X, 2) + np.power(0.1 * Y, 5)) + np.cos(X) * np.sin(Y) * np.tan(0.1 * X)
#w_init = w_init * (np.sinc(Y)) * np.cos(Y) 
#w_init = w_init * np.cos(Y) 
w_init = np.reshape(w_init, N)

if args.order == 2:
    print('\nUsing second order space discretization.\n')

    # A using kronecker product
    e1 = np.ones(grid_points) / (math.pow(delta_x, 2))
    _A = spdiags([e1, -2 * e1, e1],[-1, 0, 1], grid_points, grid_points).tocsr()
    _A[0,grid_points-1] = 1 * e1[0] 
    _A[grid_points-1,0] = 1 * e1[0]
    I = scipy.sparse.eye(grid_points).tocsr()
    A = scipy.sparse.kron(I , _A, 'csr') + scipy.sparse.kron(_A, I, 'csr')
    A[0,0] = 2
    #plt.spy(A, marker = '.', markersize = 2)
    #plt.show()

    e1 = np.ones(grid_points) / (2 * math.pow(delta_x, 1)) 
    _Dx = spdiags([-e1, e1],[-1, 1], grid_points, grid_points).tocsr()
    _Dx[0,grid_points-1] = -1 * e1[0] 
    _Dx[grid_points-1,0] = 1 * e1[0]
    I = scipy.sparse.eye(grid_points).tocsr()
    Dx = scipy.sparse.kron(_Dx, I, 'csr')
    #plt.spy(Dx, marker = '.', markersize = 2)
    #plt.show()

    _Dy = spdiags([-e1, e1],[-1, 1], grid_points, grid_points).tocsr()
    _Dy[0,grid_points-1] = -1 * e1[0] 
    _Dy[grid_points-1,0] = 1 * e1[0]
    I = scipy.sparse.eye(grid_points).tocsr()
    Dy = scipy.sparse.kron(I, _Dy, 'csr')
    #plt.spy(Dy, marker = '.', markersize = 2)
    #plt.show()

elif args.order == 4:
    print('\nUsing fourth order space discretization.\n')

    # A using kronecker product 4th order
    e1 = np.ones(grid_points) / (12 * math.pow(delta_x, 2)) 
    _A = spdiags([-e1, 16 * e1, -30 * e1, 16 * e1, -e1], [-2, -1, 0, 1, 2], grid_points, grid_points).tocsr()
    _A[0,grid_points-1] = 16 * e1[0] 
    _A[0,grid_points-2] = -1 * e1[0] 
    _A[1,grid_points-1] = -1 * e1[0] 
    _A[grid_points-1,0] = 16 * e1[0] 
    _A[grid_points-1,1] = -1 * e1[0] 
    _A[grid_points-2,0] = -1 * e1[0] 
    I = scipy.sparse.eye(grid_points).tocsr()
    A = scipy.sparse.kron(I , _A, 'csr') + scipy.sparse.kron(_A, I, 'csr')
    A[0,0] = 2
    #plt.spy(A, marker = '.', markersize = 2)
    #plt.show()

    e1 = np.ones(grid_points) / (12 * math.pow(delta_x, 1)) 
    _Dx = spdiags([e1, -8 * e1, 8 * e1, -e1], [-2, -1, 1, 2], grid_points, grid_points).tocsr()
    _Dx[0, grid_points-1] = -8 * e1[0]
    _Dx[0, grid_points-2] = 1 * e1[0]
    _Dx[1, grid_points-1] = 1 * e1[0]
    _Dx[grid_points-1, 0] = 8 * e1[0]
    _Dx[grid_points-1, 1] = -1 * e1[0]
    _Dx[grid_points-2, 0] = -1 * e1[0]
    Dx = scipy.sparse.kron(I, _Dx, 'csr')
    #plt.spy(Dx, marker = '.', markersize = 2)
    #plt.show()


    _Dy = spdiags([e1, -8 * e1, 8 * e1, -e1], [-2, -1, 1, 2], grid_points, grid_points).tocsr()
    _Dy[0, grid_points-1] = -8 * e1[0]
    _Dy[0, grid_points-2] = 1 * e1[0]
    _Dy[1, grid_points-1] = 1 * e1[0]
    _Dy[grid_points-1, 0] = 8 * e1[0]
    _Dy[grid_points-1, 1] = -1 * e1[0]
    _Dy[grid_points-2, 0] = -1 * e1[0]
    Dy = scipy.sparse.kron(_Dy, I, 'csr')
    #plt.spy(Dy, marker = '.', markersize = 2)
    #plt.show()

sol, count = iterative_solver(grid_points, tspan, end_time, w_init, delta_x, A, Dx, Dy, diff_coef, guess = args.guess, solver_name = args.method)
print(count)

print('Time stepper count: ', len(count))

def make_animation(inputs, name):

    fig = plt.figure()
    lim = domain_length/2
    ax = plt.axes(xlim=(-lim, lim), ylim=(-lim, lim))

    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')
    
    mesh = ax.pcolormesh(X, Y, np.reshape(inputs[0], (grid_points, grid_points)), cmap='jet', shading='gouraud')
    cb = fig.colorbar(mesh, cax=cax)

    def animate(i):
        mesh = ax.pcolormesh(X, Y, np.reshape(inputs[i], (grid_points, grid_points)), cmap='jet', shading='gouraud')
        cax.cla()
        fig.colorbar(mesh, cax=cax)
        return mesh,

    anim = FuncAnimation(fig, animate, frames=len(inputs), interval=50)

    print(f'\nSaving Animation: {name}.mp4\n')
    anim.save(name + '.mp4', fps=20, extra_args=['-vcodec', 'libx264'], dpi = 50)

#y = [(sol.y[:,i] - sol.y[:,i].mean())/sol.y[:,i].std() for i in range(40)]
#if args.video:
#  make_animation(y, 't_advec_diff')
## make_animation(psi_r, 't_random0')
