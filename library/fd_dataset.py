import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

class DataSet:
    def __init__(self, X=[], t=[]):
        self.X = X # 入力
        self.t = t # 出力

    def __len__(self):
        return len(self.X) # データ数(10)を返す

    def __getitem__(self, index):
        # index番目の入出力ペアを返す
        return self.X[index], self.t[index]
    

class DataSet:
    def __init__(self, X=[], t=[]):
        self.X = X # 入力
        self.t = t # 出力

    def __len__(self):
        return len(self.X) 

    def __getitem__(self, index):
        # index番目の入出力ペアを返す
        return self.X[index], self.t[index]
    

class FD_Dataset():
  def __init__(self, dataset_path="/content/drive/My Drive/CMAPSSData"):
    self._dataset_path = dataset_path
    
    # the files did not contain headers. Here we create labels based on documentation
    index_columns_names =  ["UnitNumber","Cycle"]
    op_settings_columns = ["Op_Setting_"+str(i) for i in range(1,4)]
    sensor_columns =["Sensor_"+str(i) for i in range(1,22)]
    self.column_names = index_columns_names + op_settings_columns + sensor_columns
    self.target_name = 'Target_RUL'
    self.label_name = 'Label'
    self.train = DataSet()
    self.test = DataSet()


  def load(self, data_id=1):
    '''
    やってること：
    train, test, rul.txt の読み込み
    Noneの値を落とす
    カラムの名前付け
    '''

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
    
  def CreateRUL(self):
      '''
      Train：CycleのデータからRULをせいせい
      Test:Cycleのデータとrul.txtの値からRULを生成
      rul.txtはTestUnitのサイクルが終了した時点でのRULの値を示している。
      '''
      # Train
      max_cycle = self.raw_train.groupby('UnitNumber')['Cycle'].max().reset_index()
      max_cycle.columns = ['UnitNumber', 'MaxOfCycle']
      
      # merge the max cycle back into the original frame
      raw_train_merged = self.raw_train.merge(max_cycle, on='UnitNumber', how='inner')
      
      # calculate RUL for each row
      raw_train_merged[self.target_name] = raw_train_merged["MaxOfCycle"] - raw_train_merged["Cycle"]
      
      # set DataSet instance
      self.train_dataframe = raw_train_merged.drop('MaxOfCycle', axis=1)

      #Test
      self.raw_rul_test['UnitNumber'] = self.raw_rul_test.index+1
      self.raw_rul_test.rename(columns={0:"ActualRUL"}, inplace=True)
      max_cycle = self.raw_test.groupby('UnitNumber')['Cycle'].max().reset_index()
      max_cycle.columns = ['UnitNumber', 'MaxOfCycle']

      self.raw_rul_test["MaxRUL"] = self.raw_rul_test['ActualRUL']+max_cycle["MaxOfCycle"]
      self.raw_rul_test.drop("ActualRUL", axis=1, inplace=True)
      raw_test_merged = self.raw_test.merge(self.raw_rul_test,on='UnitNumber',how='left')
      raw_test_merged[self.target_name] = raw_test_merged['MaxRUL']-raw_test_merged['Cycle']
      
      # set DataSet instance
      self.test_dataframe = raw_test_merged.drop('MaxRUL', axis=1)
  
  def InsertFailure(self, x, target_alarm):
      last = x.index[-1]
      x.iloc[-target_alarm:] = 1
      return x
  
  def CreateLabel(self, target_alarm):
      '''
      Cycleの値からLabelの生成
      '''
      all_data = self.raw_train
      all_data['Label'] = 0
      all_data['Label'] = all_data.groupby('UnitNumber')['Label'].transform(self.InsertFailure, target_alarm=target_alarm)

      #split test data
      random.seed(1234)
      idx = random.sample(range(1, 101), 20) #idx for test data
      rest_idx = [i for i in range(1, 101) if i not in idx]
      self.test_dataframe = all_data[all_data['UnitNumber'].isin(idx)].reset_index(drop=True)
      self.train_dataframe = all_data[all_data['UnitNumber'].isin(rest_idx)].reset_index(drop=True)

      return 

  def DropConst(self):
      '''
      変化のないカラムとオプションを落とす
      '''
      leakage_to_drop = ['Cycle', 'Op_Setting_1', 'Op_Setting_2', 'Op_Setting_3']
      leakage_to_drop += ['Sensor_'+str(i) for i in [1, 5, 6, 10, 16, 18, 19]]
      self.train_dataframe.drop(leakage_to_drop, axis = 1, inplace=True)
      self.test_dataframe.drop(leakage_to_drop, axis = 1, inplace=True)

  def MinMaxNorm(self):
      X = pd.concat([self.train_dataframe.iloc[:, 1:-1], self.test_dataframe.iloc[:, 1:-1]]).reset_index(drop=True)
      train_size = self.train_dataframe.shape[0]
      
      X_min, X_max = X.min(), X.max()
      X = (X-X_min)/(X_max-X_min)
      self.train_dataframe.iloc[:, 1:-1] = X.iloc[:train_size, :]
      self.test_dataframe.iloc[:, 1:-1] = X.iloc[train_size:, :].reset_index(drop=True)

  def LimitRUL(self, RUL_limit):
    self.train_dataframe[self.target_name] = self.train_dataframe[self.target_name].transform(lambda x: np.minimum(x, RUL_limit))
    self.test_dataframe[self.target_name] = self.test_dataframe[self.target_name].transform(lambda x: np.minimum(x, RUL_limit))

  def SlideWindow(self, dataframe, ws, cs=1, model_name='DAG'):
    '''
    input: dataframe
    output: dataset obj
    X.size= (n-ws, cs, ws//cs, n_feautures)
    '''
    dataset = DataSet()
    X, t = [], []
    width = ws//cs
    groups = dataframe.groupby('UnitNumber')
    for _, x in groups:
      n = len(x)
      x = torch.tensor(x.drop('UnitNumber', axis=1).values).float()
      for i in range(n-ws):
          X.append(x[i:i+ws, :-1])
          t.append(x[i+ws, -1])
    X = torch.stack(X, dim=0).float()
    t = torch.stack(t, dim=0).float()

    if model_name == 'DAG':
      X = X.reshape(X.shape[0], cs, width, X.shape[-1])
    dataset.X = X
    dataset.t = t
    return dataset

  def GetRULDataset(self, ws, cs=1, batch_size, data_id=1, limit_flag = False, limit_value = None, model_name='DAG'):
    self.load(data_id)
    self.CreateRUL()
    self.DropConst()
    self.MinMaxNorm()

    if limit_flag: 
      self.LimitRUL(limit_value)

    self.train = self.SlideWindow(self.train_dataframe, ws, cs, model_name)
    self.test = self.SlideWindow(self.test_dataframe, ws, cs, model_name)

    train_dataloader = torch.utils.data.DataLoader(self.train, batch_size= batch_size)
    test_dataloader = torch.utils.data.DataLoader(self.test, batch_size= 1)
    return train_dataloader, test_dataloader

  def GetLabelDataset(self, ws, cs=1, batch_size, data_id=1, target_alarm = 20, model_name='DAG'):
    self.load(data_id)
    self.CreateLabel(target_alarm)
    self.DropConst()
    self.MinMaxNorm()

    self.train = self.SlideWindow(self.train_dataframe, ws, cs, model_name)
    self.test = self.SlideWindow(self.test_dataframe, ws, cs, model_name)

    train_dataloader = torch.utils.data.DataLoader(self.train, batch_size= batch_size)
    test_dataloader = torch.utils.data.DataLoader(self.test, batch_size= 1)
    return train_dataloader, test_dataloader

if __name__ == '__main__':
  fd = FD_Dataset()
  fd.load(data_id=1)
  fd.MinMaxNorm()
  fd.LimitRUL(125)
  train = fd.SlideWindow(fd.train, 30 ,10)
  train.X.size()
