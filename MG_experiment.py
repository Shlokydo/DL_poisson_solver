import numpy as np
np.random.seed(5)

import torch
torch.manual_seed(0)

import h5py
import argparse
import time
import matplotlib.pyplot as plt
import os

from torch.utils.data import TensorDataset, DataLoader, Dataset, sampler
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils
from network import MGv1, MGv2, MGv2_1, regular_cnn

# Training settings
parser = argparse.ArgumentParser(description='MultiGrid ConvNet Training and Testing')
parser.add_argument('--action', type=int, default=1, metavar='Train/test', choices=[1, 0],
                    help='Change mode between training(1) and testing(0)')
parser.add_argument('--batch-size', type=int, default=512, metavar='batch_size',
                    help='input batch size for training (default: 512)')
parser.add_argument('--test-batch-size', type=int, default=2048, metavar='batch_size',
                    help='input batch size for training (default: 2048)')
parser.add_argument('--grid-size', type=int, default=128, metavar='grid_size',
                    help='input grid size for training (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--lr', type=float, default=0.001, metavar='Learning_rate',
                    help='learning rate (default: 0.001)')
parser.add_argument('--epochs', type=int, default=5, metavar='epochs',
                    help='number of epochs to train (default: 5)')

class bxDataset(Dataset):
  '''Custom Dataset for get b and x values in Ax = b (Building it for practice purpose and also to later on to integrate with Horovod for distributed training.)'''

  def __init__(self, location = './Dataset_AD.h5', grid_size = 128):
    #HDF5 file object
    data = h5py.File('./Dataset_AD.h5', 'r')
    
    self.grid_size = grid_size
    self.om = np.asarray(data['omega_' + str(grid_size)])
    self.psi = np.asarray(data['psi_' + str(grid_size)])

  def __len__(self):
    return len(self.om)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    x = self.psi[idx]
    x = (x - x.mean()) / x.std()
    x = torch.from_numpy(np.reshape(x, (self.grid_size, self.grid_size))).float()
    b = self.om[idx]
    b = (b - b.mean()) / b.std()
    b = torch.from_numpy(np.reshape(b, (1, self.grid_size, self.grid_size))).float()
    sample = (b, x)

    return sample

def dataloader(grid_size):
  '''
  Returns train and val dataloader
  '''
  #Main dataset
  bxdata = bxDataset(grid_size=grid_size)
  #Creating a subset for actual training
  #num_samples = len(bxdata) - (len(bxdata) % args.batch_size)
  num_samples = args.batch_size * 8 + args.test_batch_size
  rand_index =  np.random.permutation(len(bxdata))
  sub_bxdata = torch.utils.data.Subset(bxdata, rand_index[:num_samples])
  test_data = torch.utils.data.Subset(bxdata, rand_index[num_samples:])

  #Create a train val dataset 
  split = args.test_batch_size
  train_data, val_data = torch.utils.data.random_split(sub_bxdata, [len(sub_bxdata) - split, split]) 

  #Create samplers (Need for Horovod practice)
  train_sampler = sampler.SequentialSampler([i for i in range(len(train_data))])
  val_sampler = sampler.SequentialSampler([i for i in range(len(val_data))])
  test_sampler = sampler.SequentialSampler(test_data)

  #Create DataLoader
  train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = args.batch_size, drop_last = True, pin_memory= True)
  val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size = args.test_batch_size, drop_last = True, pin_memory= True)
  test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size = 5, drop_last = True, pin_memory= True)

  return train_dataloader, val_dataloader, test_dataloader

def optimizer_scheduler(params, lr):
  '''
  Inputs: model.parameters, learning_rate
  Returns an optimizer and its scheduler
  '''
  optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=15, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=lr * 0.005, eps=1e-08)
  #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 10, T_mult=2, eta_min=lr*0.25, last_epoch=-1)
  return optimizer, scheduler

def compute_loss(data, target):
  target = torch.squeeze(target)
  criterion = nn.MSELoss()
  return criterion(data, target)

def train(model, data_loader, opt, scheduler, epoch):
  model.train()

  avg_loss = 0
  iters = len(data_loader)
  for batch_idx, (data, target) in enumerate(data_loader):
    if args.cuda:
      data, target = data.to(device), target.to(device)
    opt.zero_grad()
    output = model(data)
    loss = compute_loss(target, output)
    avg_loss += loss.item()
    loss.backward()
    opt.step()
    #scheduler.step(epoch + batch_idx / iters)

  avg_loss /= len(data_loader)
  return avg_loss

def val(model, data_loader, scheduler):
  model.eval()

  loss = 0
  for batch_idx, (data, target) in enumerate(data_loader):
    if args.cuda:
      data, target = data.to(device), target.to(device)
    output = model(data)
    loss += compute_loss(target, output).item()

  loss /= len(data_loader)
  scheduler.step(loss)
  return loss

def test(model, data_loader):
  model.eval()

  x = torch.squeeze(model(b)).detach().cpu().numpy()
  return x


if __name__ == '__main__':
  args = parser.parse_args()
  args.cuda = not args.no_cuda and torch.cuda.is_available()

  # Initializing Tensorboard SummaryWriter
  dir_name = './logs/MGv2_1_10_4_10_2'
  writer = SummaryWriter(dir_name + '/summary') 

  # Limit # of CPU threads to be used per worker.
  torch.set_num_threads(1)

  train_dataloader, val_dataloader, test_dataloader = dataloader(args.grid_size)

  #Loading Model
  kernel_size = 10
  levels = 4
  filters = 10
  depth = 2
  model = MGv2_1(kernel_size, levels, filters, depth)
  print(model)
  if args.cuda:
    device = 'cuda'
    if torch.cuda.device_count() > 1:
      model = nn.DataParallel(model)
      print("Using ", torch.cuda.device_count(), " GPUs!")
  else:
    device = 'cpu'

  # Move model to device.
  model.to(device)
  #print(model)
  num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print('The number of parameters of model is', num_params)
  num_train_samples = len(train_dataloader) * args.batch_size
  print('Number of training samples: ', num_train_samples)

  #Get an optimizer
  lr = args.lr * args.batch_size / 512
  opt, sch = optimizer_scheduler(model.parameters(), lr)

  epochs = 0
  loss_min = 100
  train_loss = 0
  if os.path.exists(dir_name + '/model.pth'):
    checkpoint = torch.load(dir_name + '/model.pth')
    print('Loading model state from checkpoint.')
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs = checkpoint['epoch']
    loss_min = checkpoint['loss']

  #writer.add_graph(model,torch.ones(2, 1, 16, 16).to(device))
  #writer.close()
  if args.action:

    start_time = time.time()

    for epoch in range(epochs + 1, epochs + args.epochs + 1):
      epoch_time = time.time()
      print('\nStart of Epoch: ', epoch)
      loss = train(model, train_dataloader, opt, sch, epoch)
      print('Avg. training loss at end of epoch ', epoch, '  : ', loss)
      test_loss = val(model, val_dataloader, sch)
      print('Avg. validation loss at end of epoch ', epoch, ': ', test_loss)
      if (loss_min > test_loss):
        loss_min = test_loss
        train_loss = loss
        print('Saving model\nVal_loss:  {}'.format(loss_min))
        torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': opt.state_dict(),
              'loss': loss_min,
              }, dir_name + '/model.pth')
      writer.add_scalars('MSELoss', {'Train': loss, 'Test': test_loss}, global_step=epoch, walltime=None)
      writer.close()
      print('Epoch time: ', time.time() - epoch_time)
    end_time = (time.time()-start_time)/60
    print('Total training time (in minutes): ', end_time)
    writer.add_hparams(hparam_dict = {'LR':lr, 'Model': dir_name, 'Num_params': num_params, 'Kernel_size': kernel_size, 'Levels': levels, 'Num_kernels': filters, 'Train_samples': num_train_samples, 'CNN_depth': depth}, metric_dict = {'Train_loss': train_loss, 'Val_loss': loss_min, 'Time': end_time})
    writer.flush()
    writer.close()

  else:

    image_dir = dir_name + '/images/'
    if not os.path.exists(image_dir):
      print('\nCreating Image directory\n')
      os.mkdir(image_dir)

    #Going to test and create plots 
    iterable = iter(test_dataloader)
    for i in range(5):
      (b, x) = next(iterable)
      x_model = test(model, b.to(device))
      b = torch.squeeze(b).numpy()
      x = torch.squeeze(x).numpy()

      fig, axs = plt.subplots(5, 4, figsize=(20, 16))
      fig.suptitle(dir_name, fontsize = 20)
      for row in range(5):
        rb = b[row]
        rx = x[row]
        rx_model = x_model[row]
        error = rx - rx_model
        ls = [rb, rx, rx_model, error]
        ls_name = ['Model Input', 'Expected Output', 'Output', 'Error']
        for col in range(4):
          ax = axs[row, col]
          ax.set_title(ls_name[col], fontsize = 10)
          dummy = ax.imshow(ls[col], cmap = 'jet')
          ax.set_xticks([])
          ax.set_yticks([])

        cbar = fig.colorbar(dummy, ax=axs[row, :], shrink=1.0, ticks=[s for s in np.linspace(error.min(), error.max(), 8)])

      print('Saving file')
      writer.add_figure(str(i), fig)
