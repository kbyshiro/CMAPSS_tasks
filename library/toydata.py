import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def create_toydata(lifetime, target, residual, n_size):
  lifetime_t = np.random.normal(loc=lifetime, scale=50, size=n_size)
  lifetime_t = list(map(int, lifetime_t))
  lifetime_s2 = []
  lifetime_s3 = []
  for x in lifetime_t:
    lifetime_s2.append(int(np.random.normal(loc=x-target-residual, scale= 10)))
    lifetime_s3.append(int(np.random.normal(loc=x-target+residual, scale=10)))
  t_true = []
  y_s2 = []
  y_s3 = []
  for x, y, z in zip(lifetime_t, lifetime_s2, lifetime_s3):
    tmp = np.zeros(x)
    tmp[-target:]  = 1
    t_true.append(tmp)

    tmp = np.full((x, 2), fill_value=[np.log(0.1), np.log(0.9)])
    tmp[y:] = np.log(.9), np.log(.1)
    y_s2.append(tmp)

    tmp = np.full((x, 2), fill_value=[np.log(0.1), np.log(0.9)])
    tmp[z:] = np.log(.9), np.log(.1)
    y_s3.append(tmp)
  t_true = torch.tensor([item for sub in t_true for item in sub], dtype=torch.long)
  y_s2 = torch.tensor([item for sub in y_s2 for item in sub])
  y_s3 = torch.tensor([item for sub in y_s3 for item in sub])
  return t_true, y_s2, y_s3
