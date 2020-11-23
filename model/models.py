import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
%matplotlib inline
sns.set()
# Setting seed for reproducability
np.random.seed(1234)  
PYTHONHASHSEED = 0

class DAG_CNN_LSTM_Regressor(nn.Module):
  def __init__(self,L1_input_size, L1_hidden_size, conv_size, 
              conv_stride, maxpool_size, maxpool_stride, L2_input_size, L2_hidden_size, output_size):
    super().__init__()
    #L1
    self.L1_input_size = L1_input_size
    self.L1_hidden_size = L1_hidden_size
    self.LSTM1 = nn.LSTM(self.L1_input_size, self.L1_hidden_size)

    # CNN
    self.conv_size = conv_size
    self.conv_stride = conv_stride
    self.conv_out_size = conv_out_size
    self.maxpool_size = maxpool_size
    self.maxpool_stride = maxpool_stride
    self.CNN = nn.Sequential(nn.Conv2d(1, self.conv_out_size, kernel_size=self.conv_size,
                                       stride=self.conv_stride, padding=[1, 1]), 
                              nn.MaxPool2d(self.maxpool_size, self.maxpool_stride), 
                              nn.Flatten())

    #L2
    self.L2_input_size = L2_input_size
    self.L2_hidden_size = L2_hidden_size
    self.LSTM2 = nn.LSTM(self.L2_input_size, self.L2_hidden_size)

    # fully connected layer
    self.output_size = output_size
    self.Linear = nn.Linear(self.L2_hidden_size, self.output_size)
  
  def init_hidden(self, batch_size):
        hidden_state1 = torch.zeros(1, batch_size, self.L1_hidden_size)
        cell_state1 = torch.zeros(1, batch_size, self.L1_hidden_size)
        self.hidden1 = (hidden_state1, cell_state1)
        hidden_state2 = torch.zeros(1, batch_size, self.L2_hidden_size)
        cell_state2 = torch.zeros(1, batch_size, self.L2_hidden_size)
        self.hidden2 = (hidden_state2, cell_state2)


  def forward(self, x):
    batch_size, seq_size, width, n_features = x.size()

    #LSTM1
    #input size = (seq_size, batch_size, width*n_features)
    #output_size = (seq_size, batch_size, hidden_size)
    out1, self.hidden1 = self.LSTM1(torch.transpose(x, 0, 1).reshape(seq_size, batch_size, -1), self.hidden1)
    
    #CNN
    #input size = (seq_size, 1, n_features, width)
    #output_size = (seq_size, batch_size, hidden_size)
    out2 = torch.zeros((batch_size,seq_size, self.L1_hidden_size ))
    for i in range(batch_size):
      out2[i] = self.CNN(x[i].view(seq_size, 1, n_features, width))
    # print(out2.size())
    
    # combine outputs
    out = out1+ torch.transpose(out2, 0, 1)
    #LSTM2
    _, self.hidden2 = self.LSTM2(out, self.hidden2)
    
    #fully connecter
    out = self.Linear(self.hidden2[0][0].view(batch_size, -1))
    return out
