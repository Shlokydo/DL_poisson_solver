import numpy as np
import torch
from torch import nn

class multigrid_conv2d_v1(nn.Module):
  '''
    This version of Multigrid Conv2d is based on Henning's version of Mutipole Conv2d. 
    v1 is more or less same to same as Henning's method.
    This network doesn't use weight sharing and Restriction operation is the one used in actual Multigrid.
  '''
  
  def __init__(self, in_channels, kernel_size, N, padding_mode = 'circular'):
    super(multigrid_conv2d_v1, self).__init__()
    
    self.in_channels, self.kernel_size, self.N = in_channels, kernel_size, N
    self.padding_mode = padding_mode

    self.layers = nn.ModuleList(self.generate_layers())
    
    #Stuff for restriction operation
    self.restriction_stencil = np.asarray([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16
    self.restriction_stencil = np.resize(self.restriction_stencil, (self.in_channels, 1, 3, 3))

    self.restrict_conv = nn.Conv2d(self.in_channels, self.in_channels, 3, stride = 2, groups = self.in_channels)
    self.restrict_conv.weight = nn.Parameter(data = torch.tensor(self.restriction_stencil).float(), requires_grad = False)
    self.restrict_conv.bias = nn.Parameter(data = torch.zeros(self.in_channels).float(), requires_grad = False)

    #Stuff for prolongation operation
    self.prolongate_stencil = np.asarray([[1, 1], [1, 1]])/4
    self.prolongate_stencil = np.resize(self.prolongate_stencil, (self.in_channels, 1, 2, 2))

    self.prolongate_conv = nn.Conv2d(self.in_channels, self.in_channels, 2, stride = 1, groups = self.in_channels)
    self.prolongate_conv.weight = nn.Parameter(data = torch.tensor(self.prolongate_stencil).float(), requires_grad = False)
    self.prolongate_conv.bias = nn.Parameter(data = torch.zeros(self.in_channels).float(), requires_grad = False)

    #Resgistering buffer for psi_init
    self.register_buffer('psi_init', None)

  def generate_layers(self):
    #Generating model layers
    layers = []
    padding = self.kernel_size - 1

    for i in range(self.N+1):
      cnn = nn.Conv2d(2 * self.in_channels, self.in_channels, self.kernel_size, padding = padding, padding_mode = self.padding_mode)
      layers.append(cnn)
    
    return layers

  def restriction(self, x, N):
    '''
    Outputs a list of x, restricted to different coarse levels.
    '''
    out = [x]

    for i in range(N):
      y = out[-1]
      y = y.repeat(1, 1, 2, 2)[:, :, (y.shape[2]-1):(2 * y.shape[2]), (y.shape[3]-1):(2 * y.shape[3])]

      y_restricted = self.restrict_conv(y).requires_grad_()
      out.append(y_restricted)

    return out

  def prolongate(self, x):
    '''
    Outputs x prolongated to a level above.
    '''
    inp = x.repeat(1, 1, 2, 3)[:, :, :x.shape[2]+1, :x.shape[3]+1]
    inp_h = nn.functional.interpolate(x, size=None, scale_factor=2, mode='nearest', align_corners=None) 
    
    inp_h[:,:,::2,::2] = inp[:,:,:-1,:-1]
    inp_h[:,:,::2,1::2] = 0.5 * (inp[:,:,:-1,:-1] + inp[:,:,:-1,1:])
    inp_h[:,:,1::2,::2] = 0.5 * (inp[:,:,:-1,:-1] + inp[:,:,1:,:-1])
    inp_h[:,:,1::2,1::2] = self.prolongate_conv(inp)

    return inp_h

  def forward(self, x):
    
    x_restricted = self.restriction(x, self.N)[::-1]

    x = x_restricted[0]
    y = nn.functional.interpolate(torch.zeros_like(x), scale_factor=0.5)

    for (cnn_i, x_i) in zip(self.layers, x_restricted):
      
      y_prolongated = self.prolongate(y).requires_grad_()

      #Input to the network (x, y) channels
      inp_to_net = torch.cat([x_i, y_prolongated], 1)
      y = nn.Tanh()(cnn_i(inp_to_net))
    
    return y

class multigrid_conv2d_v2(multigrid_conv2d_v1):
  '''
  With weight-sharing. 
  '''
  def __init__(self, in_channels, kernel_size, N, padding_mode = 'circular'):
    super(multigrid_conv2d_v2, self).__init__(in_channels, kernel_size, N, padding_mode = 'circular')

    padding = self.kernel_size - 1
    cnn = nn.Conv2d(2 * self.in_channels, self.in_channels, self.kernel_size, padding = padding, padding_mode = self.padding_mode)
    self.layers = nn.ModuleList([cnn])

  def forward(self, x):
    
    x_restricted = self.restriction(x, self.N)[::-1]

    x = x_restricted[0]
    y = nn.functional.interpolate(torch.zeros_like(x), scale_factor=0.5)

    for x_i in x_restricted:
      
      y_prolongated = self.prolongate(y).requires_grad_()

      #Input to the network (x, y) channels
      inp_to_net = torch.cat([x_i, y_prolongated], 1)
      y = nn.Tanh()(self.layers[0](inp_to_net))

    return y

class MG_v1(nn.Module):
  '''
  Class encanpsulating multigrid_conv2d_v1
  '''
  def __init__(self, kernel_size, levels):
    super(MG_v1, self).__init__()

    self.input_conv = nn.Conv2d(1, 5, kernel_size, padding = kernel_size-1, padding_mode = 'circular')
    self.MGlayer = multigrid_conv2d_v1(5, kernel_size, levels)
    self.output_conv = nn.Conv2d(5, 1, kernel_size, padding = kernel_size - 1, padding_mode = 'circular')

  def forward(self, x):

    x = nn.Tanh()(self.input_conv(x))
    x = nn.Tanh()(self.MGlayer(x))
    return self.output_conv(x)

class MG_v2(MG_v1):
  '''
  Class encanpsulating multigrid_conv2d_v2
  '''
  def __init__(self, kernel_size, levels):
    super(MG_v2, self).__init__(kernel_size, levels)
    self.MGlayer = multigrid_conv2d_v2(5, kernel_size, levels)

class regular_cnn(nn.Module):
  '''
    Creates a Regular_CNN model.
    Inputs for initialization:
    Kernel size and number of layers
  '''
  def __init__(self, kernel_size, L):
    super(regular_cnn, self).__init__()

    layers = []
    for i in range(L):
      l = nn.Conv2d(1, 1, kernel_size, padding_mode = 'circular', padding = kernel_size-1)
      layers.append(l)

    self.layers = nn.ModuleList(layers)

  def forward(self, x):

    for l in self.layers[:-1]:
      x = nn.Tanh()(l(x))

    return self.layers[-1](x)
