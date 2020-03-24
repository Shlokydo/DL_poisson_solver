import numpy as np 
import math
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
import scipy.linalg as sl
from scipy.integrate import solve_ivp
import scipy.sparse
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

import h5py
import methods

#Opening a HDF5 file for writing datasets
f = h5py.File('Dataset_AD.h5', mode = 'a')

#Set the number of grid points
grid_points = 64
#Set the domain lenght
domain_length = 20
#Set the end time
end_time = 50
#Set the diffusion coefficient
diff_coef = 0.001

#Creating Dataset in the HDF5 file
try:
    p_dataset = f.create_dataset('psi_' + str(grid_points), (1, grid_points * grid_points), maxshape = (None, grid_points * grid_points), dtype = 'float32')
    resize = 0
except:
    p_dataset = f.get('psi_' + str(grid_points))
    resize = p_dataset.shape[0]
    print('Current number of samples: ', resize)

try:
    w_dataset = f.create_dataset('omega_' + str(grid_points), (1, grid_points * grid_points), maxshape = (None, grid_points * grid_points), dtype = 'float32')
except:
    w_dataset = f.get('omega_' + str(grid_points))

tspan = np.arange(0, end_time, 0.5)
N = int(math.pow(grid_points, 2))

x2 = np.linspace(int(-domain_length/2), int(domain_length/2), grid_points + 1)
x = x2[0:grid_points]
delta_x = x[1] - x[0]
y = x
X, Y = np.meshgrid(x, y, indexing = 'ij')

# w_init = np.exp(-2 * np.power(X, 2) - (np.power(Y, 2) / 10)) - np.exp(-1 * np.power(X - 1, 2) / 20 - 2 * (np.power(Y - 1, 2))) + np.exp(-1 * np.power(X + 4, 2) / 20 - 2 * (np.power(Y + 8, 2)))
# w_init = w_init + np.exp(-2 * np.power(X - 3, 2) - (np.power(Y, 2) / 10)) - np.exp(-1 * np.power(X + 4, 2) / 20 - 2 * (np.power(Y - 2, 2))) + np.exp(-1 * np.power(X + 5, 2) / 20 - 2 * (np.power(Y - 8, 2)))
# w_init = w_init + np.sin(X) + np.tanh(X * Y) 
# w_init = np.exp(-0.25 * np.power(X-5, 2) - 2 * (np.power(Y, 2))) - np.exp(-0.25 * np.power(X+5, 2) - 2 * (np.power(Y, 2)))
# w_init = w_init + np.tan(4 * np.random.randint(40, size = (grid_points, grid_points))) * np.sin(np.random.randint(20, size = (grid_points, grid_points)))  
# w_init = w_init + np.tan(2 * np.random.rand(grid_points, grid_points))
#w_init = (np.power(0.1 * X, 2) + np.power(0.1 * Y, 2))
# w_init = (np.power(0.1 * X, 2) + np.power(0.1 * Y, 5)) + np.cos(X) * np.sin(Y) * np.tan(0.1 * X)
# w_init = w_init + (0.1 * np.power(Y, 2)  + np.sinc(Y)) * np.cos(Y)  + np.random.rand(grid_points, grid_points) + np.exp(-2 * np.power(X, 2) - (np.power(Y, 2) / 20))
w_init = Y + np.sin(X)
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
sol, psi_r, omega_r = methods.fft_solver(tspan, end_time, w_init, delta_x, A, Dx, Dy, diff_coef, grid_points, N, domain_length)
# # print(cg_count)

temp = resize
num_samples = 6000
if len(psi_r) < num_samples:
    tmp = 1
    resize += len(psi_r)
    print('Number of samples to be added: ', resize - temp)
else:
    tmp = int(len(psi_r) / num_samples)
    resize += (int(len(psi_r) / tmp) + 1)
    print('Number of samples to be added: ', resize - temp)

#Resize the HDF5 datasets
p_dataset.resize((resize, grid_points * grid_points))
w_dataset.resize((resize, grid_points * grid_points))

#Appending data to HDF5 datasets
p_dataset[temp:] = np.asarray(psi_r)[::tmp]
w_dataset[temp:] = np.asarray(omega_r)[::tmp]
f.attrs['Samples'] = resize
print('Total number of samples: ', resize)

#Flushing and closing the HDF5 file
print('Saving in HDF5 file.')
f.flush()
f.close()
 
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

y = [sol.y[:,i] for i in range(20)]
make_animation(y, 't_advec_diff')
# make_animation(psi_r, 't_random0')