import numpy as np 
import math
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
import scipy.linalg as sl
from scipy.integrate import solve_ivp
import scipy.sparse
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

import methods

#Set the number of grid points
grid_points = 64
#Set the domain lenght
domain_length = 20
#Set the end time
end_time = 10
#Set the diffusion coefficient
diff_coef = 0.001

tspan = np.arange(0, end_time, 0.5)
N = int(math.pow(grid_points, 2))

x2 = np.linspace(int(-domain_length/2), int(domain_length/2), grid_points + 1)
x = x2[0:grid_points]
delta_x = x[1] - x[0]
y = x
X, Y = np.meshgrid(x, y, indexing = 'ij')
w_init = np.exp(-2 * np.power(X, 2) - (np.power(Y, 2) / 20))
w_init = np.reshape(w_init, N)

# Using kronecker product
e1 = np.ones(grid_points)
_A = spdiags([e1, -2 * e1, e1],[-1, 0, 1], grid_points, grid_points).tocsr()
_A[0,grid_points-1] = 1 
_A[grid_points-1,0] = 1
I = scipy.sparse.eye(grid_points).tocsr()
A = scipy.sparse.kron(I , _A, 'csr') + scipy.sparse.kron(_A, I, 'csr')
A[0,0] = 2

_Dx = spdiags([-e1, e1],[-1, 1], grid_points, grid_points).tocsr()
_Dx[0,grid_points-1] = -1 
_Dx[grid_points-1,0] = 1
I = scipy.sparse.eye(grid_points).tocsr()
Dx = scipy.sparse.kron(_Dx, I, 'csr')

e1 = np.ones(grid_points)
_Dy = spdiags([-e1, e1],[-1, 1], grid_points, grid_points).tocsr()
_Dy[0,grid_points-1] = -1 
_Dy[grid_points-1,0] = 1
I = scipy.sparse.eye(grid_points).tocsr()
Dy = scipy.sparse.kron(I, _Dy, 'csr')

# sol = methods.inv(tspan, end_time, w_init, delta_x, A, Dx, Dy, diff_coef)
# sol, cg_count = methods.cg_solver(tspan, end_time, w_init, delta_x, A, Dx, Dy, diff_coef)
# sol, gmres_count = methods.gmres_solver(tspan, end_time, w_init, delta_x, A, Dx, Dy, diff_coef)
# sol, bicgstab_count = methods.bicgstab_solver(tspan, end_time, w_init, delta_x, A, Dx, Dy, diff_coef)
sol, psi_r = methods.fft_solver(tspan, end_time, w_init, delta_x, A, Dx, Dy, diff_coef, grid_points, N, domain_length)
# # print(cg_count)

# print('Time: ', sol.t)
print('Time stepper count: ', len(psi_r))

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

    anim = FuncAnimation(fig, animate, frames=len(inputs), interval=20)

    print(f'\nSaving Animation: {name}.mp4\n')
    anim.save(name + '.mp4', fps=15, extra_args=['-vcodec', 'libx264'], dpi = 100)

y = [sol.y[:,i] for i in range(sol.y.shape[1])]
make_animation(y, 't_advec_diff')
# make_animation(psi_r, 't_random')