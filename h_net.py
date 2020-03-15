import numpy as np
import torch
from torch import nn


from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple
import math

from torch.nn import functional as F

import numpy as np 
import math
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
import scipy.linalg as sl
from scipy.integrate import solve_ivp
import scipy.sparse
from matplotlib.animation import FuncAnimation


class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# custom con2d, because pytorch don't have "padding='same'" option.
def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):

    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_needed = max(0, (out_rows - 1) * stride[0] + effective_filter_size_rows -
                  input_rows)
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)

class conv_Axb(nn.Module):

    def __init__(self):
        super(conv_Axb, self).__init__()

        l = 15

        self.l1 = Conv2d(1,l,10)
        self.l2 = Conv2d(l,1,6)

        self.opt = torch.optim.Adam(self.parameters())

    def push(self, x):
        '''
        x shape = [batch x N**2]
        '''

        batch = x.shape[0]
        N = int(np.sqrt(x.shape[-1]))
        x = x.reshape((batch,1,N,N))

        o = nn.Tanh()(self.l1(x))
        o = self.l2(o)

        return o.squeeze()

        return o.reshape((batch,-1))

    def getA(self, grid_points):
        N = grid_points**2

        B = -4*torch.eye(N,N)
        idx = torch.arange(N)

        e1 = torch.ones(N)
        e2 = torch.zeros(N)

        e1[torch.arange(grid_points)*grid_points-1] = 0
        e2[torch.arange(grid_points)*grid_points] = 1

        B[idx,idx-1] = 1-e2
        B[idx-1,idx] = 1-e2

        B[idx-grid_points+1,idx] = 1-e1#e3
        B[idx-grid_points,idx] = 1
        B[idx-(N-grid_points),idx] = 1

        B[idx,idx-grid_points+1] = 1-e1 #4
        B[idx,idx-grid_points] = 1
        B[idx,idx-(N-grid_points)] = 1

        B[10,0] = 2

        # e1 = np.ones(grid_points)
        # _A = spdiags([e1, -2 * e1, e1],[-1, 0, 1], grid_points, grid_points).tocsr()
        # _A[0,grid_points-1] = 1 
        # _A[grid_points-1,0] = 1
        # I = scipy.sparse.eye(grid_points).tocsr()
        # A = scipy.sparse.kron(I , _A, 'csr') + scipy.sparse.kron(_A, I, 'csr')
        # A[0,0] = 2
        # A = A.tocoo()

        # values = A.data
        # indices = np.vstack((A.row, A.col))

        # i = torch.LongTensor(indices)
        # v = torch.FloatTensor(values)
        # shape = A.shape

        # return torch.sparse.FloatTensor(i, v, torch.Size(shape))

        return B


    def learn(self, grid_points, A, N, i):

        x = torch.rand((32,N))*2-1
        b = torch.matmul(x,A.T)

        x = x.reshape((32,grid_points,grid_points))
        x_hat = self.push(b)

        loss = torch.mean((x-x_hat)**2)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if i % 100 == 0:
            print(loss.detach().numpy())
        
        return loss.detach().numpy()

    def get_error(self, grid_points):

        A = self.getA(grid_points)
        N = grid_points**2

        x = torch.rand((32,N))*2-1
        b = torch.matmul(x,A.T)

        x_hat = self.push(b)

        x = x.reshape((32,grid_points,grid_points))
        loss = (x-x_hat)**2

        loss = torch.mean(loss,0).detach().cpu().numpy()
        print(loss.shape)

        return torch.mean((x-x_hat)**2).data, loss

cAxb = conv_Axb()

def training():
    min_a = 100
    grid_points = 20
    A = cAxb.getA(grid_points)
    N = grid_points ** 2
    for i in range(10000):
        a = cAxb.learn(grid_points, A, N, i)
        if (min_a > a):
            min_a = a
            print('saving model: {}'.format(a))
            torch.save(cAxb.state_dict(), './model.ckp')

# training()

cAxb = conv_Axb()
print('Loading saved model')
cAxb.load_state_dict(torch.load('./model.ckp'))
import matplotlib.pyplot as plt

l, img = cAxb.get_error(16)

a = plt.imshow(img)
plt.colorbar(a)
plt.show()
print(l)

l, img = cAxb.get_error(256)

print(l)
a = plt.imshow(img)
plt.colorbar(a)
plt.show()