import numpy as np
import torch 
from torch import nn

import spconv
from utils_loader import make_symmetric, Spconvec_to_torch, to_Spconvec

import time
from unet_parts import *

def cg_batch(A_bmm, B, M_bmm=None, X0=None, rtol=1e-3, atol=0., maxiter=None, verbose=False):
  """Solves a batch of PD matrix linear systems using the preconditioned CG algorithm.

  This function solves a batch of matrix linear systems of the form

      A_i X_i = B_i,  i=1,...,K,

  where A_i is a n x n positive definite matrix and B_i is a n x m matrix,
  and X_i is the n x m matrix representing the solution for the ith system.

  Args:
      A_bmm: A callable that performs a batch matrix multiply of A and a K x n x m matrix.
      B: A K x n x m matrix representing the right hand sides.
      M_bmm: (optional) A callable that performs a batch matrix multiply of the preconditioning
          matrices M and a K x n x m matrix. (default=identity matrix)
      X0: (optional) Initial guess for X, defaults to M_bmm(B). (default=None)
      rtol: (optional) Relative tolerance for norm of residual. (default=1e-3)
      atol: (optional) Absolute tolerance for norm of residual. (default=0)
      maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
      verbose: (optional) Whether or not to print status messages. (default=False)
  """
  K, n, m = B.shape

  if M_bmm is None:
    M_bmm = lambda x: x
  if X0 is None:
    X0 = torch.zeros_like(B)
  if maxiter is None:
    maxiter = 5 * n

  assert B.shape == (K, n, m)
  assert X0.shape == (K, n, m)
  assert rtol > 0 or atol > 0
  assert isinstance(maxiter, int)

  X_k = X0
  R_k = B - A_bmm(X_k)
  Z_k = M_bmm(R_k)

  P_k = torch.zeros_like(Z_k)

  P_k1 = P_k
  R_k1 = R_k
  R_k2 = R_k
  X_k1 = X0
  Z_k1 = Z_k
  Z_k2 = Z_k

  B_norm = torch.norm(B, dim=1)
  stopping_matrix = torch.max(rtol*B_norm, atol*torch.ones_like(B_norm))

  if verbose:
    print("%03s | %010s %06s" % ("it", "dist", "it/s"))

  optimal = False
  start = time.perf_counter()
  for k in range(1, maxiter + 1):
    start_iter = time.perf_counter()
    Z_k = M_bmm(R_k)

    if k == 1:
      P_k = Z_k
      R_k1 = R_k
      X_k1 = X_k
      Z_k1 = Z_k
    else:
      R_k2 = R_k1
      Z_k2 = Z_k1
      P_k1 = P_k
      R_k1 = R_k
      Z_k1 = Z_k
      X_k1 = X_k
      denominator = (R_k2 * Z_k2).sum(1)
      denominator[denominator == 0] = 1e-8
      beta = (R_k1 * Z_k1).sum(1) / denominator
      P_k = Z_k1 + beta.unsqueeze(1) * P_k1

    denominator = (P_k * A_bmm(P_k)).sum(1)
    denominator[denominator == 0] = 1e-8
    alpha = (R_k1 * Z_k1).sum(1) / denominator
    X_k = X_k1 + alpha.unsqueeze(1) * P_k
    R_k = R_k1 - alpha.unsqueeze(1) * A_bmm(P_k)
    end_iter = time.perf_counter()

    residual_norm = torch.norm(A_bmm(X_k) - B, dim=1)

    if verbose:
      print("%03d | %8.4e %4.2f" %
          (k, torch.max(residual_norm-stopping_matrix),
            1. / (end_iter - start_iter)))

    if (residual_norm <= stopping_matrix).all():
      optimal = True
      break

  end = time.perf_counter()

  if verbose:
    if optimal:
      print("Terminated in %d steps (reached maxiter). Took %.3f ms." %
            (k, (end - start) * 1000))
    else:
      print("Terminated in %d steps (optimal). Took %.3f ms." %
            (k, (end - start) * 1000))
  
  return X_k

class conjugate_gradient_loss(nn.Module):
  
  '''
  Module for calculating 
  '''
  def __init__(self, num_iterations = 5):
    super(conjugate_gradient_loss, self).__init__()

    self.iterations = num_iterations

  def forward(self, A_inv, A, b):
	  
    A = A.requires_grad_(False)

    def A_bmm(X):
      Y = [(torch.sparse.mm(A[i], X[i])).unsqueeze(0) for i in range(A.shape[0])]
      return torch.cat(Y, dim=0)

    def M_bmm(X):
      Y = [(torch.sparse.mm(A_inv[i], X[i])).unsqueeze(0) for i in range(A.shape[0])]
      return torch.cat(Y, dim=0)
    
    b = b.unsqueeze(-1).requires_grad_(False)
    x_approx = cg_batch(A_bmm, B=b, M_bmm=M_bmm, rtol=1e-5, atol=1e-5, verbose=False, maxiter = self.iterations).squeeze().requires_grad_(True)	

    x_approx = x_approx - 1
    return torch.pow(x_approx, 2).mean()

class precond_net(nn.Module):
  
  '''
  Network architecture for Sparse Convolution Network. 
  '''
  def __init__(self):
    super(precond_net, self).__init__()
    
    self.layers = spconv.SparseSequential()

    self.layers.add(spconv.SubMConv2d(1, 64, 1, padding = (1, 1), stride = (1, 1), use_hash=False, indice_key="subm1"), 'SubM_1') #[B, 64, W, H]
    self.layers.add(nn.PReLU())
    self.layers.add(spconv.SparseConv2d(64, 256, 3, padding = (1, 1), stride = (1, 1), use_hash = False), 'SConv_1') #[B, 256, W, H]
    self.layers.add(nn.PReLU())
    self.layers.add(spconv.SparseConv2d(256, 512, 3, padding = (1, 1), stride = (1, 1), use_hash = False), 'SConv_2') #[B, 512, W, H]
    self.layers.add(nn.PReLU())
    self.layers.add(spconv.SparseConv2d(512, 256, 3, padding = (1, 1), stride = (1, 1), use_hash = False), 'SConv_3') #[B, 256, W, H]
    self.layers.add(nn.PReLU())
    self.layers.add(spconv.SparseConv2d(256, 64, 3, padding = (1, 1), stride = (1, 1), use_hash = False), 'SConv_4') #[B, 128, W, H]
    self.layers.add(nn.PReLU())
    self.layers.add(spconv.SubMConv2d(64, 1, 3, padding = (1, 1), stride = (1, 1), use_hash=False, indice_key="subm2"), 'SubM_2') #[B, 1, W, H]
    self.layers.add(nn.PReLU())

  def forward(self, A):
    
    A = to_Spconvec(A)
    A_inv = self.layers(A)
    A_inv = Spconvec_to_torch(A_inv)
    A_inv = make_symmetric(A_inv)
    return A_inv

class precondUNet_ws(nn.Module):

  '''
  UNet like architecture but with weight shared layers
  '''

  def __init__(self, maxpool = True):
    super(precondUNet_ws, self).__init__()

    self.inConv = DoubleConv(1, 64, 32)
    self.downs = Down(64, 64, 128, maxpool)
    self.ups = Up_ws(64, 64, 128)
    self.outConv = DoubleConv(64, 1, 32)
  
  def forward(self, A):
    
    A = to_Spconvec(A)              #[B, 1, W, H]
    A_inp = self.inConv(A)          #[B, 64, W, H]
    A1 = self.downs(A_inp)          #[B, 64, W/2, H/2] -> [64, 128, 64]
    A2 = self.downs(A1)             #[B, 64, W/4, H/2] -> [64, 128, 64]
    A3 = self.downs(A2)             #[B, 64, W/8, H/2] -> [64, 128, 64]
    A4 = self.downs(A3)             #[B, 64, W/16, H/2] -> [64, 128, 64]
    A3 = self.ups(A4, A3)           #[B, 64, W/8, H/8] -> [64, 128, 64]
    A2 = self.ups(A3, A2)           #[B, 64, W/4, H/4] -> [64, 128, 64]
    A1 = self.ups(A2, A1)           #[B, 64, W/2, H/2] -> [64, 128, 64]
    A_out = self.ups(A1, A_inp)     #[B, 64, W, H] -> [64, 128, 64]
    A_inv = self.outConv(A_out)     #[B, 1, W, H] -> [64, 32, 1]

    A_inv = Spconvec_to_torch(A_inv)
    A_inv = make_symmetric(A_inv)
    return A_inv