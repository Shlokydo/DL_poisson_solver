#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:10:00 2020

@author: henning
"""


import torch
import numpy as np

from scipy.sparse import spdiags
import scipy.sparse
import scipy.sparse.linalg

import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.cuda.FloatTensor)




def makeA(grid_points):
    '''
    This is taken from your code.

    '''
        
    # Using kronecker product
    e1 = np.ones(grid_points)
    _A = spdiags([e1, -2 * e1, e1],[-1, 0, 1], grid_points, grid_points).tocsr()
    _A[0,grid_points-1] = 1 
    _A[grid_points-1,0] = 1
    I = scipy.sparse.eye(grid_points).tocsr()
    A = scipy.sparse.kron(I , _A, 'csr') + scipy.sparse.kron(_A, I, 'csr')
    A[0,0] = 2
    
    return A



def generate_stencils():
    
    size = 2**np.arange(5,16)
    '''
    This implementation is garbage and will crash at s = 256
    
    '''
    
    for s in size:
        
        print(s)
        
        A = makeA(s)
        A[0,0] = 2
        Ainv = scipy.sparse.linalg.inv(A).toarray()
        
        Amed = np.median(Ainv, 0)
        
        #(s**2 + s)//2 is the position of the center stencil
        stencil = Ainv[(s**2 + s)//2] - Amed
        stencil = np.reshape(stencil, (s,s)) - np.mean(stencil)
        
        np.save('stencil'+str(s), stencil)
        
        #return stencil


class deconvolve(torch.nn.Module):
    
    
    def __init__(self, stencil_size, cropped_size = None):
        
        if type(cropped_size) == type(None):
            cropped_size = stencil_size
        
        assert (cropped_size <= stencil_size), 'Cropped size must be smaller than stencil size'
        assert stencil_size in (32,64,128), 'Only 32x32, 64x64 and 128x128 available'
        super(deconvolve, self).__init__()
        
        
        #Cropping with stencil to the desired size
        w = (stencil_size - cropped_size)//2
        padding_size = cropped_size - 1
        
        self.stencil = np.load('stencil'+str(stencil_size)+'.npy')
        
        if w > 0:
            self.stencil = self.stencil[w:-w,w:-w]
            #self.stencil = self.stencil - np.mean(self.stencil)
            
        self.stencil = np.reshape(self.stencil, (1,1,*self.stencil.shape))
        self.conv = torch.nn.Conv2d(1, 1, cropped_size, stride = 1,
                                    padding = (padding_size, padding_size),
                                    padding_mode = 'circular')
        
        self.conv.weight = torch.nn.Parameter(data = torch.tensor(self.stencil).float())
        
        
        
    def __call__(self, grid):
        
        grid = torch.from_numpy(grid).cuda().float()
        grid = grid.reshape((1,1,*grid.shape))
        
        return self.conv(grid).squeeze().cpu().detach().numpy()
    
    
    
def load_data():
    
    import h5py
    data = h5py.File('/home/henning/Downloads/Dataset_AD.h5')
    om = np.reshape(np.array(data['omega_64']), (-1,64,64))
    psi = np.reshape(np.array(data['psi_64']), (-1,64,64))
    
    return om, psi


def visualize(psi, omega, stencil_size, truncation_size):
    
    plt.subplot(1,2,1)
    plt.suptitle('L('+str(stencil_size)+','+str(truncation_size)+')')
    plt.imshow(psi)
    plt.subplot(1,2,2)
    plt.imshow(deconvolve(stencil_size, truncation_size)(omega))
    plt.tight_layout()
    
    
    
    
om, psi = load_data()
visualize(psi[666], om[666], 128, 55)

