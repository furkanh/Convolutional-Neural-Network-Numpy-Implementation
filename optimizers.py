# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 19:34:14 2019

@author: User
"""

import networkx as nx
import numpy as np
from layers import *
from initializers import *
from losses import *

class Optimizer:
    def __init__(self, network, reset = False):
        self.params = []
        self.network = network
        self.optim_dict = {}
        self.optim_dict['loss_list'] = []
        self.loss_value = 0
        for node in nx.topological_sort(network):
            if isinstance(node,Weight):
                self.params.append(node)
        if reset == False:
          self.load_history()
        
    def step(self):
        '''
        An optimizer can override this def to
        implement different update rules.
        It can store grad matrices for RMSProp
        and can delete by this def.
        '''
        pass
        
    def epoch(self):
      self.optim_dict['loss_list'].append(self.loss_value)
      self.loss_value = 0
      self.save_history()
      
    def batch(self):
      self.loss_value = self.loss_value + network.last.out[0]
    
    def zero_grad(self):
        for node in nx.topological_sort(self.network):
            node.grad = None
            node.d = {}
            
    def save_history(self):
      path = os.path.join(Network.path, self.network.name + '_optim.npy')
      np.save(path, self.optim_dict, allow_pickle=True)
      self.network.save_model()
      
    def load_history(self):
      path = os.path.join(Network.path, self.network.name + '_optim.npy')
      if os.path.exists(path):
        self.optim_dict = np.load(path, allow_pickle=True).item()
        print('Optimizer loaded from memory.')
        return True
      return False

class SGD(Optimizer):
    def __init__(self, network, learning_rate = 0.1, momentum=0.9, weight_decay=0.01, reset=False):
        super().__init__(network, reset=reset)
        self.learning_rate = learning_rate 
        self.momentum = momentum
        self.N = 0
        self.weight_decay = weight_decay
        for param in self.params:
          param.m = 0
        
    def step(self):
        for param in self.params:
            param.m = self.momentum*param.m + (1-self.momentum)*param.grad
            param.out = (1-(2*self.learning_rate*self.weight_decay)/self.N)*param.out - self.learning_rate*param.m
            
    def batch(self):
      super().batch()
      self.N = self.N + 1
    
    def epoch(self):
      super().epoch()
      self.N = 0
      
class Adadelta(Optimizer):
  def __init__(self, network, decay_rate=0.9, epsilon = 1e-6, reset=False):
    super().__init__(network, reset=reset)
    self.decay_rate = decay_rate
    self.epsilon = epsilon
    for param in self.params:
      param.cum_grad = 0
      param.cum_update = 0
    
  def step(self):
    for param in self.params:
      param.cum_grad = self.decay_rate*param.cum_grad + ((1-self.decay_rate)*param.grad)*param.grad
      rms_grad = np.sqrt(param.cum_grad+self.epsilon)
      rms_update = np.sqrt(param.cum_update+self.epsilon)
      update = -(rms_update/rms_grad)*param.grad
      param.out = param.out + update
      param.cum_update = self.decay_rate*param.cum_update + ((1-self.decay_rate)*update)*update
            
class Adam(Optimizer):
  def __init__(self, network, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon = 1.0,reset=False):
    super().__init__(network, reset=reset)
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.optim_dict['is_first_epoch'] = True
    for param in self.params:
      param.m = 0
      param.v = 0
    
  def step(self):
    for param in self.params:
      if param.out.shape != param.grad.shape:
        print("Error with grad")
      """if self.optim_dict['is_first_epoch']:
        param.m = param.grad
        param.v = param.grad*param.grad
        self.optim_dict['is_first_epoch'] = False
      else:"""
      param.m = self.beta1*param.m + (1-self.beta1)*param.grad
      param.v = self.beta2*param.v + ((1-self.beta2)*param.grad)*param.grad
      param.out = param.out - self.learning_rate*param.m/(np.sqrt(param.v)+self.epsilon)
  
  def save_history(self):
    i = 0
    for param in self.params:
      self.optim_dict[str(i)+'_adam'] = (param.m, param.v)
      i = i + 1
    super().save_history()
    
  def load_history(self):
    if super().load_history():
      i = 0
      for param in self.params:
        param.m, param.v = self.optim_dict[str(i)+'_adam']
        i = i + 1