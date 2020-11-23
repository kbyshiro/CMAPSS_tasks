import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DataSet:
    def __init__(self, X=[], t=[]):
        self.X = X # 入力
        self.t = t # 出力

    def __len__(self):
        return len(self.X) # データ数(10)を返す

    def __getitem__(self, index):
        # index番目の入出力ペアを返す
        return self.X[index], self.t[index]
    

class FD_Dataset():
  def __init__(self):
    self._dataset_path ="/content/drive/My Drive/CMAPSSData"
    
    # the files did not contain headers. Here we create labels based on documentation
    index_columns_names =  ["UnitNumber","Cycle"]
    op_settings_columns = ["Op_Setting_"+str(i) for i in range(1,4)]
    sensor_columns =["Sensor_"+str(i) for i in range(1,22)]
    self.column_names = index_columns_names + op_settings_columns + sensor_columns
    self.target_name = 'Target_Remaining_Useful_Life'
    self.train = DataSet()
    self.test = DataSet()

  def load(self, data_id, drop_const = True, rul=True):
    self._train_name = 'train_FD00{}.txt'.format(data_id)
    self._test_name = 'test_FD00{}.txt'.format(data_id)
    self._test_rul_name = 'RUL_FD00{}.txt'.format(data_id)
    self.raw_train = pd.read_csv(os.path.join(self._dataset_path, self._train_name), sep=" ", header=None)
    self.raw_test = pd.read_csv(os.path.join(self._dataset_path, self._test_name), sep=" ", header=None)
    self.raw_rul_test = pd.read_csv(os.path.join(self._dataset_path, self._test_rul_name), header=None)
    
    # drop pesky NULL columns
    self.raw_train.drop(self.raw_train.columns[[26, 27]], axis=1, inplace=True)
    self.raw_test.drop(self.raw_test.columns[[26, 27]], axis=1, inplace=True)

    # set name columns
    self.raw_train.columns = self.column_names
    self.raw_test.columns = self.column_names
    
    if rul: 
      self.GetTrainRUL(self.raw_train)
      self.GetTestRUL(self.raw_test, self.raw_rul_test)

    if drop_const:
      self.DropConst()

  def GetTrainRUL(self, raw_train):
      max_cycle = raw_train.groupby('UnitNumber')['Cycle'].max().reset_index()
      max_cycle.columns = ['UnitNumber', 'MaxOfCycle']
      
      # merge the max cycle back into the original frame
      raw_train_merged = raw_train.merge(max_cycle, left_on='UnitNumber', right_on='UnitNumber', how='inner')
      
      # calculate RUL for each row
      Target_Remaining_Useful_Life = raw_train_merged["MaxOfCycle"] - raw_train_merged["Cycle"]
      
      # set DataSet instance
      self.train.X = raw_train_merged.drop("MaxOfCycle", axis=1)
      self.train.t = pd.Series(Target_Remaining_Useful_Life, name=self.target_name)


  def GetTestRUL(self, raw_test, raw_rul_test):
      raw_rul_test['UnitNumber'] = raw_rul_test.index+1
      raw_rul_test.rename(columns={0:"RUL"}, inplace=True)
      max_cycle = raw_test.groupby('UnitNumber')['Cycle'].max().reset_index()
      max_cycle.columns = ['UnitNumber', 'MaxOfCycle']

      raw_rul_test["RUL_failed"] = raw_rul_test['RUL']+max_cycle["MaxOfCycle"]
      raw_rul_test.drop("RUL", axis=1, inplace=True)
      raw_test = raw_test.merge(raw_rul_test,on='UnitNumber',how='left')
      Target_Remaining_Useful_Life = raw_test['RUL_failed']-raw_test['Cycle']
      
      # set DataSet instance
      self.test.X = raw_test.drop("RUL_failed", axis=1)
      self.test.t = pd.Series(Target_Remaining_Useful_Life, name = self.target_name)
  
  def DropConst(self):
      leakage_to_drop = ['Cycle', 'Op_Setting_1', 'Op_Setting_2', 'Op_Setting_3']
      leakage_to_drop += ['Sensor_'+str(i) for i in [1, 5, 6, 10, 16, 18, 19]]
      self.train.X.drop(leakage_to_drop, axis = 1, inplace=True)
      self.test.X.drop(leakage_to_drop, axis = 1, inplace=True)

  def MinMaxNorm(self):
    tmp = pd.concat([self.train.X.iloc[:, 1:], self.test.X.iloc[:, 1:]]).reset_index(drop=True)
    train_size = len(self.train)
    X_min, X_max = tmp.min(), tmp.max()
    tmp = (tmp-X_min)/(X_max-X_min)
    self.train.X.iloc[:, 1:] = tmp.iloc[:train_size, :]
    self.test.X.iloc[:, 1:] = tmp.iloc[train_size:, :].reset_index(drop=True)

  def LimitRUL(self, RUL_limit):
    self.train.t = pd.Series(np.minimum(self.train.t, RUL_limit), name=self.target_name)
    self.test.t = pd.Series(np.minimum(self.test.t, RUL_limit), name=self.target_name)

  def SlideWindow(self, dataset, ws, cs=1, mode='LSTMCNN'):
    #X.size= (n-ws, cs, ws/cs, n_feautures)
    combined_dataset = pd.concat([dataset.X, dataset.t], axis=1)
    unit_max = combined_dataset['UnitNumber'].max()
    width = int(ws/cs)
    X, y = [], []
    if mode == 'LSTMCNN':
      for k in range(unit_max):
        tmp_data = combined_dataset[combined_dataset['UnitNumber']==k+1].drop('UnitNumber', axis=1).values
        n = len(tmp_data)
        for i in range(n-ws):
            X.append([tmp_data[i+(j)*width:i+(j+1)*width, :-1] for j in range(cs)])
            y.append(tmp_data[i+ws, -1])
    elif mode == 'LSTM':
      for k in range(unit_max):
        tmp_data = combined_dataset[combined_dataset['UnitNumber']==k+1].drop('UnitNumber', axis=1).values
        n = len(tmp_data)
        for i in range(n-ws):
            X.append(tmp_data[i:i+width, :-1])
            y.append(tmp_data[i+ws, -1])
    dataset.X = torch.tensor(X).float()
    dataset.t = torch.tensor(y).float()
    return dataset

if __name__ == '__main__':
  fd = FD_Dataset()
  fd.load(data_id=1)
  fd.MinMaxNorm()
  fd.LimitRUL(125)
  train = fd.SlideWindow(fd.train, 30 ,10)
  train.X.size()
