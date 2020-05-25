import numpy as np
np.random.seed(5)

import torch
torch.manual_seed(0)

import argparse
import time
import os

from utils_loader import dataloader, optimizer_scheduler, train, val
from network_modules import conjugate_gradient_loss, precond_net

# Training settings
parser = argparse.ArgumentParser(description='Preconditioner SparseConvent Training.')
parser.add_argument('--batch-size', type=int, default=512, metavar='batch_size',
                    help='input batch size for training (default: 512)')
parser.add_argument('--split-percent', type=float, default=0.8, metavar='split-percentage',
                    help='Percentage of dataset to use for training (rest goes to validation)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--lr', type=float, default=0.001, metavar='Learning_rate',
                    help='learning rate (default: 0.001)')
parser.add_argument('--epochs', type=int, default=5, metavar='epochs',
                    help='number of epochs to train (default: 5)')

if __name__ == '__main__':
  args = parser.parse_args()
  args.cuda = not args.no_cuda and torch.cuda.is_available()

  # Initializing Tensorboard SummaryWriter
  dir_name = './precond_logs/Exp1'
  writer = SummaryWriter(dir_name + '/summary') 

  # Limit # of CPU threads to be used per worker.
  torch.set_num_threads(1)

  train_dataloader, val_dataloader = dataloader(args.batch_size, args.split_percent)

  #Get Model here
  model = precond_net() 
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
  print(model)
  num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print('The number of parameters of model is', num_params)
  num_train_samples = len(train_dataloader) * args.batch_size
  print('Number of training samples: ', num_train_samples)

  #Get an optimizer
  lr = args.lr * args.batch_size / 512
  opt, sch = optimizer_scheduler(model.parameters(), lr)
  cg_loss = conjugate_gradient_loss(5)

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

  #Training loop goes here.
  start_time = time.time()

  for epoch in range(epochs + 1, epochs + args.epochs + 1):
    epoch_time = time.time()
    print('\nStart of Epoch: ', epoch)

    loss = train(model, train_dataloader, opt, sch, cuda, cg_loss)
    print('Avg. training loss at end of epoch ', epoch, '  : ', loss)

    test_loss = val(model, val_dataloader, sch, cuda, cg_loss)
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

  #writer.add_hparams(hparam_dict = {'LR':lr, 'Model': dir_name, 'Num_params': num_params, 'Kernel_size': kernel_size, 'Levels': levels, 'Num_kernels': filters, 'Train_samples': num_train_samples, 'CNN_depth': depth}, metric_dict = {'Train_loss': train_loss, 'Val_loss': loss_min, 'Time': end_time})
  writer.flush()
  writer.close()
