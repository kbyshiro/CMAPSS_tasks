import numpy as np
import torch.nn as nn

class CostOptimizationWithTBM():
  def __init__(self, c0, k1, k2, k3, T0):
    self.c0 = c0
    self.k1 = k1
    self.k2 = k2
    self.k3 = k3
    self.T0 = T0
    self.softmax = nn.Softmax(dim=1)
    return

  def setprob(self, input, target, unit_size = 100):
    input = self.softmax(input)
    self.lda = unit_size/len(target)
    N_TN, N_FP = 0, 0
    for x, y in zip(input[:, 0], target):
      if x < 0.2 and y == 1:
        N_TN+= 1
      elif x > 0.85 and y == 0:
        N_FP += 1
    self.mu = (N_TN+N_FP)/len(target)
    self.delta = N_TN/(N_TN+N_FP)
    return 

  def OptimizeTBM(self):
    '''
    t_opt1 :コストによる最適なTBM実施区間
    t_opt2 : 安全によるTBM実施区間の上限
    '''
    self.t_opt1 = np.sqrt(2*self.c0/(self.lda*self.mu*self.delta*(self.k3-self.k1-(1-self.delta)*self.mu/self.lda*self.k2)))
    self.t_opt2 = (self.T0**2/(self.delta*self.mu)*(3-self.lda*self.T0))**(1/3)
    self.t_opt = min(self.t_opt1, self.t_opt2)
    return self.t_opt
  
  def OptimizeCost(self, t):
    '''
    2次までテーラー展開した時のコストの算出
    '''
    self.C0 = self.c0/t    
    self.C1 = self.k1*self.lda*(1-self.delta*self.mu*t/2)
    self.C2 = self.k2*(1-self.delta)*self.mu*(1-self.delta*self.mu*t/2)
    self.C3 = 1/2*self.k3*self.lda*self.mu*self.delta*t
    self.c_opt = self.C0 + self.C1 + self.C2 + self.C3
    return self.c_opt

  def GetCostAndTBM(self, input, target, unit_size):
    self.setprob(input, target, unit_size)
    t_opt = self.OptimizeTBM()
    self.OptimizeCost(t_opt)
    return self.t_opt, self.c_opt
