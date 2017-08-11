# coding: utf-8
# =============================================================================
# Make a simple RNN learn binray addition 
# Binary string pairs and the sum is generated for a given #numBits

# ==============================================================================
# import various modules
from __future__ import print_function

%matplotlib notebook
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import random
import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
random.seed( 10 ) # set the random seed (for reproducibility)