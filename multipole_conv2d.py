#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:59:50 2020

@author: henning
"""


import torch
from torch import nn

import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt



img = mpimg.imread('/home/henning/Pictures/seattle.jpg')
img = np.transpose(img, [2,0,1])
    



class multipole_conv2d_v1(nn.Module):
    
    '''
    
    This version works by using dilations but has a substantial memory
    and CPU overhead.
    
    '''
    
    
    def __init__(self, in_channels, out_channels, kernel_size, N,
                 u_func = lambda x: x, padding_mode='circular'):
        
        super(multipole_conv2d_v1, self).__init__()
        
        self.in_channels, self.out_channels, self.kernel_size, self.N = \
            in_channels, out_channels, kernel_size, N
        
        self.padding_mode, self.u_func = padding_mode, u_func
        self.generate_layers()
        
        
        
    def generate_layers(self):
        
        self.layers = []
        for i in range(self.N):
            
            padding = (2**i)*(self.kernel_size-1)
            
            cnn = nn.Conv2d(self.in_channels, self.in_channels, self.kernel_size,
                            padding = padding, padding_mode = self.padding_mode,
                            dilation = 2**i)
            
            super(multipole_conv2d_v1, self).add_module('layer' + str(i), cnn)
            self.layers.append(cnn)
            
        self.combine_layer = nn.Conv2d(self.in_channels*self.N, self.out_channels,
                                       self.kernel_size, padding = self.kernel_size-1,
                                       padding_mode = self.padding_mode)
            
        return self.layers
    
    
    def forward(self, x):
        
        l = self.compute_levels(x, self.N)
        
        out = []
        
        for cnn_i, l_i in zip(self.layers, l):
            o_i = cnn_i(l_i)
            out.append(o_i)
            
        return self.combine_layer(self.u_func(torch.cat(out,1)))
    
    
    
    def compute_levels(self, x, N):
        
        out = [x]
        
        for i in range(N):
            y = out[-1]
            
            N,M = y.shape[2], y.shape[3]
            N = N//2*2
            M = M//2*2
            
            y_in = y[:,:,:N,:M].reshape((-1,self.in_channels,N//2,2,M//2,2))
            y_in = torch.mean(torch.mean(y_in,3),-1)
            
            out.append(y_in)
            
        '''
        This implementation is memory inefficient because we are upsampling
        low resolution data for ease of implementation
        '''
        out_ = []
        for y_in in out:
            
            y_in = nn.Upsample(size=(x.shape[2], x.shape[3]), mode='nearest')(y_in)
            out_.append(y_in)
            
        return out_
        
    
    
    def sketch_receptive_field(self, x):
        '''
        
        Parameters
        ----------
        x : Image of shape [1, 3, width, height]


        '''
        
        
        N = 5
        w = 12
        
        i = 200
        j = 400
        
        l = self.compute_levels(x, N)
        l = [np.transpose(x.cpu().numpy().squeeze()/255,[1,2,0]) for x in l]
        
        img = l[-1].copy()
        
        for n in range(1,N+1):
            
            print('l', N+1-n)
            
            
            k1 = N-n
            k = 2**(N-n)
            print(i-k*w, i+k*w)
            
            img[i-k*w:i+k*w, j-k*w:j+k*w] = l[k1][i-k*w:i+k*w, j-k*w:j+k*w]
            
            
        plt.imshow(img)
        
        return l
            
        


class multipole_conv2d_v2(nn.Module):
    
    '''
    
    This version processes the lower resolution data without dilations and
    should be considerably faster.
    
    '''
    
    
    def __init__(self, in_channels, kernel_size, N,
                 u_func = lambda x: x, padding_mode='circular'):
        
        super(multipole_conv2d_v2, self).__init__()
        
        self.in_channels, self.kernel_size, self.N = \
            in_channels, kernel_size, N
        
        self.padding_mode, self.u_func = padding_mode, u_func
        self.generate_layers()
        
        
        
    def generate_layers(self):
        
        self.layers = []
        for i in range(self.N+1):
            
            padding = self.kernel_size-1
            
            cnn = nn.Conv2d(2*self.in_channels, self.in_channels, self.kernel_size,
                            padding = padding, padding_mode = self.padding_mode)
            
            super(multipole_conv2d_v2, self).add_module('layer' + str(i), cnn)
            self.layers.append(cnn)
            
            
        return self.layers
    
    
    def forward(self, x):
        
        l = self.compute_levels(x, self.N)[::-1]
        
        #print('OLOL', l[0].shape, l[-1].shape)
        
        out = []
        
        x = l[0]
        
        for (cnn_i, l_i) in zip(self.layers, l):
            
            x = nn.Upsample(size=(l_i.shape[2], l_i.shape[3]),
                            mode='nearest')(x)
            
            inpt = torch.cat([x,l_i], 1)
            #print(l_i.shape, x.shape, inpt.shape)
            x = torch.nn.Tanh()(cnn_i(inpt))
            
        return x
    
    
    
    def compute_levels(self, x, N):
        
        out = [x]
        
        for i in range(N):
            y = out[-1]
            
            N,M = y.shape[2], y.shape[3]
            N = N//2*2
            M = M//2*2
            
            y_in = y[:,:,:N,:M].reshape((-1,self.in_channels,N//2,2,M//2,2))
            y_in = torch.mean(torch.mean(y_in,3),-1)
            
            out.append(y_in)
            
            
        return out

    
    
mp_conv = multipole_conv2d_v2(3, 25, 5).to('cuda:0')