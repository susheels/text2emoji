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
from model import EmojiRNN

# set here the size of the RNN state:
stateSize = 10
# set here the size of the binary strings to be used for training:
stringLen = 3

# create the model:
model = EmojiRNN(stateSize)
print ('Model initialized')

# create the loss-function:
lossFunction = nn.MSELoss() # or nn.CrossEntropyLoss() -- see question #2 below

# uncomment below to change the optimizers:
#optimizer = optim.SGD(model.parameters(), lr=3e-2, momentum=0.8)
optimizer = optim.Adam(model.parameters(),lr=0.01)
iterations = 500
min_epochs = 20
num_epochs,totalLoss = 0,float("inf")
while num_epochs < min_epochs:
    print("[epoch %d/%d] Avg. Loss for last 500 samples = %lf"%(num_epochs+1,min_epochs,totalLoss))
    num_epochs += 1
    totalLoss = 0
    for i in range(0,iterations):
        # get a new random training sample:
        x,y = getSample(stringLen)
        # zero the gradients from the previous time-step:
        model.zero_grad()
        #convert to torch tensor and variable:
        ## unsqueeze() is used to add the extra BATCH dimension:
        x_var = autograd.Variable(torch.from_numpy(x).unsqueeze(1).float()) 
        seqLen = x_var.size(0)
        x_var = x_var.contiguous()
        y_var = autograd.Variable(torch.from_numpy(y).float())
        # push the inputs through the RNN (this is the forward pass):
        pred = model(x_var)
        # compute the loss:
        loss = lossFunction(pred,y_var)
        totalLoss += loss.data[0]
        optimizer.zero_grad()
        # perform the backward pass:
        loss.backward()
        # update the weights:
        optimizer.step()
    totalLoss=totalLoss/iterations
print('Training finished!')