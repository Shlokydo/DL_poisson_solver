import torch
torch.manual_seed(0)

import os
import sys

from network import MGv1, MGv2, MGv2_1

is_cuda = torch.cuda.is_available()

class MGCNN():
  
  def __init__(self, net_type, grid_size, kernel_size, levels, filters, depth = 0):
    super(MGCNN, self).__init__()
    
    model_class = self.get_model_class(net_type)
    if not depth:
      args = (kernel_size, levels, filters)
    else:
      args = (kernel_size, levels, filters, depth)
    
    self.grid_size = grid_size
    dir_name = './logs/' + net_type + '_' + '_'.join(str(s) for s in args)
    self.model = model_class(*args)
    if is_cuda:
      self.device = 'cuda'
    else:
      self.device = 'cpu'

    #Move model to appropriate device
    self.model.to(self.device)

    if os.path.exists(dir_name + '/model.pth'):
      checkpoint = torch.load(dir_name + '/model.pth')
      print('Loading Model {} state from checkpoint.'.format(net_type + '_' + '_'.join(str(s) for s in args)))
      state_dict = checkpoint['model_state_dict']

      # create new OrderedDict that does not contain `module.`
      from collections import OrderedDict
      new_state_dict = OrderedDict()
      for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
      self.model.load_state_dict(new_state_dict)
  
  def get_model_class(self, net_type):
    return getattr(sys.modules[__name__], net_type)

  def get_guess(self, inp):
    self.model.eval()
    
    inp = (inp - inp.mean()) / inp.std()
    inp = torch.from_numpy(np.reshape(inp, (1, 1, self.grid_size, self.grid_size))).float().to(self.device)

    guess = torch.squeeze(self.model(inp)).detach().cpu().numpy()
    guess = np.resize(guess, -1)

    return guess
