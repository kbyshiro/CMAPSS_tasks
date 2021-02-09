import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASL(nn.Module):
  '''
  m : シフト量
  gm : 負値のダウンウェート
  gp : 正値のダウンウェート

  '''
  def __init__(self, m=0, gm=0 , gp=0):
    super(ASL, self).__init__()
    self.m = m
    self.gm = gm
    self.gp = gp
    self.softmax = nn.Softmax(dim=1)
  
  def forward(self, input, target):
    '''
    input(Tensor, Float) : (batch, class)
    target(Tensor, Long) : (batch, ) 
    '''
    input = self.softmax(input)
    target = target.unsqueeze(1)
    target = torch.cat([target, 1-target], axis=1)

    
    lossp = (-target[:, 0]*(input[:, 1]**self.gp)*torch.log(input[:, 0])).sum()

    input_m = torch.maximum(input[:, 0]-self.m, torch.zeros((input.size()[0]))).unsqueeze(1)

    input_m = torch.cat([input_m, 1-input_m], axis=1)
    lossm = (-target[:, 1]*(input_m[:, 0]**self.gm)*torch.log(input_m[:, 1])).sum()
    loss = (lossp+lossm)/input.size()[0]
    return loss
