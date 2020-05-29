import torch
from torch import nn

import spconv

class DoubleConv(nn.Module):
  """(convolution => [BN] => ReLU) * 2"""

  def __init__(self, in_channels, out_channels, mid_channels=None):
    super(DoubleConv, self).__init__()

    if not mid_channels:
      mid_channels = out_channels

    self.double_conv = spconv.SparseSequential()

    self.double_conv.add(spconv.SparseConv2d(in_channels, mid_channels, 3, padding = (1, 1)), 'Conv1')
    self.double_conv.add(nn.PReLU(), 'PReLU1')
    self.double_conv.add(spconv.SparseConv2d(mid_channels, out_channels, 3, padding = (1, 1)), 'Conv2')
    self.double_conv.add(nn.PReLU(), 'PReLU2')

  def forward(self, x):

    return self.double_conv(x)

class Down(nn.Module):
  """Downscaling with maxpool then double conv"""

  def __init__(self, in_channels, out_channels, mid_channels=None, maxpool = True):
    super(Down, self).__init__()

    if maxpool:
      self.pooling_conv = spconv.SparseMaxPool2d(2)
    else:
      self.pooling_conv = spconv.SparseConv2d(in_channels, in_channels, 3, padding = 1, stride = 2)

    self.pooling_dconv = DoubleConv(in_channels, out_channels, mid_channels)

  def forward(self, x):
    
    x = self.pooling_conv(x)
    x = self.pooling_dconv(x)
    return x

class Up(nn.Module): 
  """Upscaling then double conv"""

  def __init__(self, in_channels, out_channels, mid_channels=None):
    super(Up, self).__init__()
    
    self.up = spconv.SparseConvTranspose2d(in_channels , in_channels // 2, 2, padding = (0, 0), stride = (2, 2))
    self.conv = DoubleConv(in_channels, out_channels, mid_channels)

  def addspvec(self, x, y):
    
    assert x.features.shape[1] == y.features.shape[1]
    assert x.spatial_shape == y.spatial_shape

    if len(x.features) >= len(y.features):
      a = x
      b = y
    else:
      a = y
      b = x

    v = a.features
    v[:b.features.shape[0]] = b.features
    return spconv.SparseConvTensor(v, a.indices, a.spatial_shape, a.batch_size)

  def forward(self, x1, x2):
    
    x1 = self.up(x1)
    x = self.addspvec(x1, x2)
    return self.conv(x)

class Up_ws(Up):
  '''Upscaling then double conv for weight-sharing case'''

  def __init__(self, in_channels, out_channels, mid_channels=None):
    super(Up_ws, self).__init__(in_channels, out_channels, mid_channels=None)

    self.up = spconv.SparseConvTranspose2d(in_channels, in_channels, 2, padding = (0, 0), stride = (2, 2))
    self.conv = DoubleConv(in_channels, out_channels, mid_channels)

  def forward(self, x1, x2):
    
    x1 = self.up(x1)
    x = self.addspvec(x1, x2)
    return self.conv(x)
