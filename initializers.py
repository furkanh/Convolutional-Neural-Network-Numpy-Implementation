# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 19:35:17 2019

@author: User
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class Initializer:
  def __init__(self):
    self.in_shape = None
    self.out_shape = None
  
  def get_weights(self, shape):
    if len(shape) == 2:
      self.in_shape = shape[0]
      self.out_shape = shape[1]
    else:
      self.in_shape = np.prod(shape[1:])
      self.out_shape = shape[0]

class GlorotUniform(Initializer):
  def __init__(self):
    super().__init__()
  
  def get_weights(self, shape):
    super().get_weights(shape)
    scale = np.sqrt(6.0 / (self.in_shape + self.out_shape))
    W = np.random.uniform(low=-scale, high=scale, size=shape)
    return W
  
class GlorotNormal(Initializer):
  def __init__(self):
    super().__init__()
    
  def get_weights(self, shape):
    super().get_weights(shape)
    scale = np.sqrt(2.0/(self.in_shape+self.out_shape))
    return np.random.normal(loc=0.0, scale=scale, size=shape)
    
  
class HeNormal(Initializer):
  def __init__(self):
    super().__init__()
  
  def get_weights(self, shape):
    super().get_weights(shape)
    scale = np.sqrt(2.0/self.in_shape)
    return np.random.normal(loc=0.0, scale=scale, size=shape)