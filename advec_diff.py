import numpy as np 
import math
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
import scipy.linalg as sl
from scipy.integrate import solve_ivp
import scipy.sparse
from matplotlib.animation import FuncAnimation

import methods

grid_points = 64
domain_length = 20
end_time = 100
tspan = np.arange(0, end_time, 0.5)
diff_coef = 0.001
N = int(math.pow(grid_points, 2))

x2 = np.linspace(int(-domain_length/2), int(domain_length/2), grid_points + 1)
x = x2[0:grid_points]
delta_x = x[1] - x[0]
y = x
X, Y = np.meshgrid(x, y, indexing = 'ij')
w_init = np.sin(2 * X) + np.cos(2 * Y)
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
sol = methods.fft_solver(tspan, end_time, w_init, delta_x, A, Dx, Dy, diff_coef, grid_points, N, domain_length)

# # print(cg_count)

fig = plt.figure()
lim = domain_length/2
ax = plt.axes(xlim=(-lim, lim), ylim=(-lim, lim))
mesh = ax.pcolormesh(X, Y, np.reshape(sol.y[:,0], (grid_points, grid_points)), cmap='jet', shading='gouraud')
fig.colorbar(mesh)

def animate(i):
    data = sol.y[:,i]
    mesh.set_array(data)
    return mesh,

anim = FuncAnimation(fig, animate, frames=sol.y.shape[1], interval=500)

print('\nSaving Animation...\n')
anim.save('advec_diff.mp4', fps=15, extra_args=['-vcodec', 'libx264'], dpi = 235)