import numpy as np
np.random.seed(5)

import torch
torch.manual_seed(0)

from torch.utils.data import TensorDataset, DataLoader, Dataset, sampler, Subset
from torch import optim
from scipy import sparse
from scipy.io import loadmat

from glob import glob

import spconv

class sparseDataset(Dataset):
  
  '''
  Custom Dataset for getting SPD matrices (tril) stored inside a folder.
  Return: Sparse tril(A)
  '''

  def __init__(self, location = './SpMat'):
    
    #Get all the files
    files = sorted(glob(location + '/a*.mat'))
    self.mats = []

    for i in files:
      self.mats.append(loadmat(i)['A'].tocoo())

  def __len__(self):
    
    return len(self.mats)

  def __getitem__(self, idx):
    
    if torch.is_tensor(idx):
      idx = idx.tolist()

    mat = self.mats[idx] 
    #Dividing the A matrix with the maximum element in A matrix.
    mat = mat / mat.max()

    b = mat @ np.ones(mat.shape[0])

    l_mat = scipy.sparse.tril(mat)
    features = torch.from_numpy(l_mat.data).float()
    coors = torch.from_numpy(np.stack((l_mat.row, l_mat.col), axis = 1)).long()
    torch_lmat = torch.sparse.FloatTensor(coors, features, torch.Size(l_mat.shape)) 

    torch_fullmat = torch.sparse.FloatTensor(torch.LongTensor(np.vstack((mat.row, mat.col))), torch.FloatTensor(mat.data), torch.Size(mat.shape))

    return [torch_lmat, torch_fullmat, b]

def dataloader(batch_size, split_percent = 0.8):
  
  '''
  Returns train and val dataloader
  '''

  matdata = sparseDataset()
  rand_index = np.random.permutation(len(matdata))
  
  split = int(len(matdata) * split_percent)
  test_data = Subset(matdata, rand_index[:split]) 
  val_data = Subset(matdata, rand_index[split:]) 

  train_sampler = sampler.SequentialSampler([i for i in range(len(train_data))])
  val_sampler = sampler.SequentialSampler([i for i in range(len(val_data))])

  train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = batch_size, drop_last = True, pin_memory= True)
  val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size = test_batch_size, drop_last = True, pin_memory= True)

  return train_dataloader, val_dataloader

def optimizer_scheduler(params, lr):
  
  '''
  Inputs: model.parameters, learning_rate
  Returns an optimizer and its scheduler
  '''
  optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=15, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=lr * 0.005, eps=1e-08)
  return optimizer, scheduler

def to_Spconvec(batched_sparse_mat):
  
  batch_size = batched_sparse_mat.shape[0]
  shape = batched_sparse_mat.shape[1:]
  indices = batched_sparse_mat._indices().T.int()
  features = batched_sparse_mat._values().unsqueeze(-1)
  x = spconv.SparseConvTensor(features, indices, shape, batch_size)

  return x

def Spconvec_to_torch(x):
  
  non_zeros = x.features.squeeze().nonzero().squeeze()
  features = x.features[non_zeros].squeeze()
  indices = x.indices[non_zeros].T.long()
  shape = [x.batch_size] + x.spatial_shape

  return torch.sparse.FloatTensor(indices, features, shape).coalesce()

def make_symmetric(x):
  
  return torch.add(x, x.transpose(2, 1)).coalesce()

def train(model, data_loader, opt, scheduler, cuda, cg_loss):
  model.train()
  
  avg_loss = 0
  iters = len(data_loader)

  for batch_idx, (A_lower, A, b) in enumerate(data_loader):

    if cuda:
      A_lower, A, b = A_lower.to('cuda'), A.to('cuda'), b.to('cuda')
    
    opt.zero_grad()
    A_lower = to_Spconvec(A_lower)
    A_inv = model(A_lower)
    loss = cg_loss(A_inv, A, b)
    avg_loss += loss
    loss.backward()
    opt.step()

  avg_loss /= len(data_loader)
  return avg_loss

def val(model, data_loader, scheduler, cuda, cg_loss):
  model.eval()

  loss = 0
  for batch_idx, (A_lower, A, b) in enumerate(data_loader):

    if cuda:
      A_lower, A, b = A_lower.to('cuda'), A.to('cuda'), b.to('cuda')
    A_lower = to_Spconvec(A_lower)
    A_inv = model(A_lower)
    loss += cg_loss(A_inv, A, b)

  loss /= len(data_loader)
  scheduler.step(loss)
  return loss
