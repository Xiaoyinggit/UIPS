# coding:utf8
import numpy as np

class DataInput:
  def __init__(self, data, batch_size,model_name):

    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0
    self.model_name = model_name
    self.start = 0  # no off policy 的情况下需要记录开始的位置

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration
    if self.model_name == 'no_off_policy' or self.model_name =='no_off_policy_sigmoid' or self.model_name=='sigmoid':
      ts = []
      count = 0
      if self.start>=len(self.data):
        raise StopIteration
      for i in range(self.start, len(self.data)):
        if self.data[i][3]==1:  # 见过的
          ts.append(self.data[i])
          count+=1
          if count == self.batch_size:
            break
      self.start += self.batch_size
    else:
      ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                   len(self.data))]

    self.i += 1
    u, i, y,display = [], [], [], []
    for t in ts:
      u.append(t[0])
      i.append(t[1])
      y.append(t[2])
      display.append(t[3])

    return self.i, (u, i, y, display)