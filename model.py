
import math
import glob
import io
import base64
import time
from IPython.display import HTML
from IPython import display as ipythondisplay
from collections import namedtuple
from itertools import count

# Colab comes with PyTorch
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

import copy

"""
define your network here
"""
class RecruiterNetwork(nn.Module):

  def __init__(self, n_state_vector, n_action_space):
    super(RecruiterNetwork, self).__init__()
    self.layer = nn.Sequential(
        # nn.BatchNorm1d(n_state_vector + n_action_space),
        nn.Linear(n_state_vector + n_action_space, 30),
        nn.ReLU(inplace=True),
        #TODO: change this back
        # nn.BatchNorm1d(30),
        nn.Linear(30,30),
        nn.ReLU(inplace=True),

        # nn.BatchNorm1d(30),
        nn.Linear(30,30),
        nn.ReLU(inplace=True),

        # nn.BatchNorm1d(30),
        nn.Linear(30,1),
        # nn.ReLU(inplace=True)
    )

  def forward(self,x):
    out = self.layer(x)
    return out