#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 12:55:58 2020

@author: henning
"""

import torch.multiprocessing as mp
#mp.set_start_method('spawn')

import numpy as np
from torch import nn
import torch

import h5py

from multipole_conv2d import multipole_conv2d_v2
import time




def load_data():
    
    bs = np.concatenate(list(np.load('/home/henning/Documents/conv_axb/DL_poisson_solver/bs_subsampled.npy', allow_pickle=True)),0)
    xs = np.concatenate(list(np.load('/home/henning/Documents/conv_axb/DL_poisson_solver/xs_subsampled.npy', allow_pickle=True)),0)
    
    bs = bs.astype(np.float32)
    xs = xs.astype(np.float32)
    
    b_m = np.mean(bs)
    b_std = np.std(bs)
    
    x_m = np.mean(xs, 0)
    x_std = np.std(xs)
    
    bs = (bs-b_m)/b_std
    xs = (xs-x_m)/x_std
    
    return bs, xs, b_m, b_std, x_m, x_std

class green_net(nn.Module):
    

    def __init__(self):
        super(green_net, self).__init__()
        
         
    def optim(self):
        '''
        This is a necessary hack. We cannot create the optimizer in the 
        constructor because the constructor of sub classes will not have
        created its parameters yet.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        
        if not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.SGD(self.parameters(), 3E-3)
            
        return self.optimizer


    def compute_loss(self, xs, bs):
        
        N = len(xs)
        xhat = self(bs)
        loss = torch.mean((xhat.view(N,-1) - xs.view(N,-1))**2)
        
        return loss

        
    def learn_epoch(self, xs, bs):
        
        batch_size = 32
        r_idxs = np.random.permutation(len(xs))
        
        losses = []
        
        for i in range(len(xs)//batch_size):
            idxs = r_idxs[i*batch_size:(i+1)*batch_size]
            
            loss = self.compute_loss(xs[idxs], bs[idxs])
        
            self.optim().zero_grad()
            loss.backward()
            self.optim().step()
            
            losses.append(loss.detach().cpu().numpy())
            
        return np.mean(losses)
            
        
        
class regular_cnn(green_net):
    
    def __init__(self, kernel_size, L):
        
        super(regular_cnn, self).__init__()
        
        self.layers = []
        for i in range(L):

            l = nn.Conv2d(1, 1,kernel_size, padding_mode='circular',
                          padding=kernel_size-1)
            
            super(regular_cnn, self).add_module('layer' + str(i), l)
            self.layers.append(l)
        
        
    def forward(self, x):
        
        for l in self.layers[:-1]:
            x = nn.Tanh()(l(x))
        
        return self.layers[-1](x)
    
    
    
class mp_cnn(green_net):
    
    def __init__(self, kernel_size, L):
        super(mp_cnn, self).__init__()
        
        self.l1 =  nn.Conv2d(1, 5, kernel_size, padding = kernel_size-1, padding_mode='circular')
        self.l2 = multipole_conv2d_v2(5, kernel_size, L, u_func = nn.Tanh())
        #self.l2 = nn.Conv2d(1, 3, kernel_size, padding = kernel_size-1, padding_mode='circular')
        self.l3 = nn.Conv2d(5, 1, kernel_size, padding = kernel_size-1, padding_mode='circular')
        
        
    def forward(self, x):
        #print(x.shape)
        x = nn.Tanh()(self.l1(x))
        #print(x.shape)
        x = nn.Tanh()(self.l2(x))
        
        #print(x.shape)
        return self.l3(x)
    
def train_nets():
    
    L = 4
    kernel_size = [10, 30]
    
    bs, xs, b_m, b_std, x_m, x_std = load_data()
    
    xs2 = torch.unsqueeze(torch.from_numpy(xs).cuda(device='cuda:2').float(),1)
    bs2 = torch.unsqueeze(torch.from_numpy(bs).cuda(device='cuda:2').float(),1)
    
    xs3 = torch.unsqueeze(torch.from_numpy(xs).cuda(device='cuda:3').float(),1)
    bs3 = torch.unsqueeze(torch.from_numpy(bs).cuda(device='cuda:3').float(),1)
    
    losses = []
    
    
    for k in kernel_size:
        
        global regcnn, mpcnn
        
        regcnn = regular_cnn(k, L+2).to('cuda:2')
        mpcnn = mp_cnn(k, L).to('cuda:3')
        
        for _ in range(500):
            
            start_time = time.time()
            
            '''
            p1 = mp.Process(target=regcnn.learn_epoch, args=(xs2,bs2))
            p2 = mp.Process(target=mpcnn.learn_epoch, args=(xs3, bs3))
            
            
            p1.start()
            p2.start()
            
            p1.join()
            p2.join()
            
            
            return p1
            p1.spawn()
            
            return p1'''
            
            
            #l1 = regcnn.learn_epoch(xs2, bs2)
            print('reg', time.time()-start_time)
            l2 = mpcnn.learn_epoch(xs3, bs3)
            
            losses.append((1,l2))
            
            print(losses[-1], time.time()-start_time)
            
        return np.array(losses)

def plot_all(i):
    
    plt.subplot(1,4,1)
    plt.imshow(bs[i])
    plt.title('Source: x')
    plt.xticks([]); plt.yticks([])
    
    plt.subplot(1,4,2)
    plt.imshow(xs[i])
    plt.title('$L^{-1}(x)$')
    plt.xticks([]); plt.yticks([])
    
    plt.subplot(1,4,3)
    plt.imshow(xh_reg[i].squeeze())
    plt.title('CNN(x)')
    plt.xticks([]); plt.yticks([])
    
    plt.subplot(1,4,4)
    plt.imshow(xh_mp[i].squeeze())
    plt.title('MP-CNN(x)')
    plt.xticks([]); plt.yticks([])