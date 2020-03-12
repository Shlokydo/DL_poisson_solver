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
end_time = 40
tspan = np.arange(0, end_time, 0.5)
diff_coef = 0.01
N = int(math.pow(grid_points, 2))

x2 = np.linspace(int(-domain_length/2), int(domain_length/2), grid_points + 1)
x = x2[0:grid_points]
delta_x = x[1] - x[0]
y = x
X, Y = np.meshgrid(x, y, indexing = 'ij')
w_init = np.exp(-4 * np.sin(np.power(X, 2)) - (np.power(Y, 2) / 20))
w_init = np.reshape(w_init, N)

e0 = np.zeros(N)
e1 = np.ones(N)
e2 = np.ones(N)
e4 = np.zeros(N)
for i in range(grid_points):
    e2[(grid_points * i) - 1] = 0
    e4[(grid_points * i) - 1] = 1
e3 = np.zeros(N)
e5 = np.zeros(N)
e3[0] = e2[-1]
e3[1:] = e2[:-1]
e5[0] = e4[-1]
e5[1:] = e4[:-1]

A = spdiags([e1, e1, e5, e2, -4 * e1, e3, e4, e1, e1], [-(N - grid_points), -grid_points, -grid_points + 1, -1, 0, 1, grid_points - 1, grid_points, (N - grid_points)], N, N).tocsr()
Dx = spdiags([e1, -e1, e1, -e1], [-(N-grid_points), -grid_points, grid_points, (N-grid_points)], N, N).tocsr()
Dy = spdiags([e5, -e2, e3, -e4], [-grid_points+1, -1, 1, grid_points-1], N, N).tocsr()
A[0,0] = 2

# sol = methods.inv(tspan, end_time, w_init, delta_x, A, Dx, Dy, diff_coef)
# sol, cg_count = methods.cg_solver(tspan, end_time, w_init, delta_x, A, Dx, Dy, diff_coef)
# sol, gmres_count = methods.gmres_solver(tspan, end_time, w_init, delta_x, A, Dx, Dy, diff_coef)
# sol, bicgstab_count = methods.bicgstab_solver(tspan, end_time, w_init, delta_x, A, Dx, Dy, diff_coef)
sol = methods.fft_solver(tspan, end_time, w_init, delta_x, A, Dx, Dy, diff_coef, grid_points, N, domain_length)

# print(cg_count)

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
anim.save('fft_advec_diff.mp4', fps=15, extra_args=['-vcodec', 'libx264'], dpi = 235)