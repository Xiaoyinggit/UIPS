# coding:utf8
import numpy as np

class DataInput:
  def __init__(self, data, batch_size):

    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.data))]
    self.i += 1

    u, i = [], []
    for t in ts:
      u.append(t[0])
      i.append(t[1])
    return self.i, (u, i)

class DataInputOracle:
  def __init__(self, data, batch_size):

    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.data))]
    self.i += 1

    u, i, gt_prob = [], [], []
    for t in ts:
      u.append(t[0])
      i.append(t[1])
      gt_prob.append(t[2])
    return self.i, (u, i, gt_prob)

class DataInputSyn:
  def __init__(self, data, batch_size):

    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.data))]
    self.i += 1

    u, i,y,display = [], [],[],[]
    for t in ts:
      u.append(t[0])
      i.append(t[1])
      y.append(t[2])
      display.append(t[3])

    return self.i, (u, i,y,display)

class DataInputSynFliter:
  def __init__(self, data, batch_size):
      self.batch_size = batch_size
      self.data = data
      self.epoch_size = len(self.data) // self.batch_size
      if self.epoch_size * self.batch_size < len(self.data):
        self.epoch_size += 1
      self.i = 0
      self.start = 0

  def __iter__(self):
      return self

  def __next__(self):
      if self.i == self.epoch_size:
        raise StopIteration

      ts = []
      count = 0
      if self.start>=len(self.data):
        raise StopIteration
      for xx in range(self.start, len(self.data)):
        if self.data[xx][3]==1:
          ts.append(self.data[xx])
          count+=1
          if count==self.batch_size:
            break
        self.start+=1
      if len(ts)==0:
        raise StopIteration

      self.i += 1

      u, i, y, display = [], [], [], []
      for t in ts:
        u.append(t[0])
        i.append(t[1])
        y.append(t[2])
        display.append(t[3])
      return self.i, (u, i, y, display)

class DataInputWithProb:
  def __init__(self, data, batch_size):
      self.batch_size = batch_size
      self.data = data
      self.epoch_size = len(self.data) // self.batch_size
      if self.epoch_size * self.batch_size < len(self.data):
        self.epoch_size += 1
      self.i = 0
      self.start = 0

  def __iter__(self):
      return self

  def __next__(self):
      if self.i == self.epoch_size:
        raise StopIteration

      ts = []
      count = 0
      if self.start>=len(self.data):
        raise StopIteration
      for xx in range(self.start, len(self.data)):
        if self.data[xx][3]==1:
          ts.append(self.data[xx])
          count+=1
          if count==self.batch_size:
            break
        self.start+=1
      if len(ts)==0:
        raise StopIteration

      self.i += 1

      u, i, y, display = [], [], [], []
      beta_prob = []
      for t in ts:
        u.append(t[0])
        i.append(t[1])
        y.append(t[2])
        display.append(t[3])
        beta_prob.append(t[4])
        # prob = [0]*30938
        # prob[t[1]] = t[4]
        # beta_prob.append(prob)

      return self.i, (u, i, y, display,beta_prob)


class DataInputTrainImpute:
  def __init__(self, data, batch_size):

    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.data))]
    self.i += 1

    u, i,y = [], [],[]
    for t in ts:
      u.append(t[0])
      i.append(t[1])
      y.append(t[2])

    return self.i, (u,i,y)