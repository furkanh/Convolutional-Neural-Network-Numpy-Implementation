# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 19:33:29 2019

@author: User
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from layers import *

class CrossEntropyLoss(Ensemble):
    def __init__(self, prediction, label):
        layer_list = []
        out = Log(prediction)
        out = Multiply(out, label)
        out = ReduceSum(out)
        out = Multiply(out, Constant(-1))
        layer_list.append(out)
        super().__init__(layer_list)
        
class L2Norm(Ensemble):
  def __init__(self, params):
    layer_list = []
    param_list = []
    for param in params:
      param_list.append(ReduceSum(Multiply(param, Identity(param))))
    layer_list.append(Add(*param_list))
    super().__init__(layer_list)