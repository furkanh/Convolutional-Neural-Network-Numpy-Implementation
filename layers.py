# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 19:31:39 2019

@author: Furkan Hüseyin
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from initializers import *

class Layer:
    EPSILON = 1e-8
    def __init__(self):
        self.d = {}
        self.out = None
        self.graph = None
        self.grad = None
    
    def add_backward(self, network):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        last = self.graph.last
        if len(list(self.graph.neighbors(self))) == 0:
            d = np.ones_like(self.out, dtype=np.float64)
        else:
            d = np.zeros_like(self.out, dtype=np.float64)
            for node in self.graph.neighbors(self):
                d = d + node.d[self]*last.d[node]
        last.d[self] = d
        if self.grad is None:
            self.grad = d
        else:
            self.grad = self.grad + d
    
    def size(self):
        return self.out.shape
    
class Variable(Layer):
    #Any dimension
    def __init__(self, size=(1,)):
        super().__init__()
        self.out = np.zeros(size, dtype=np.float64)
        network.node_labels[self] = 'Variable'
    
    def feed(self, x):
        if x.shape != self.out.shape:
            raise Exception('''Input to the variable
                            does not have
                            the right size''')
        self.out = x

class Constant(Layer):
    #Any dimension
    def __init__(self, const, size=(1,)):
        super().__init__()
        self.out = np.zeros(size, dtype=np.float64)+const
        network.node_labels[self] = 'Constant'
        
class Weight(Layer):
    #Any dimension
    def __init__(self, initial_value = np.zeros((1,), dtype=np.float64)):
        super().__init__()
        self.out = initial_value
        network.node_labels[self] = 'Weight'

class Operator(Layer):
    def __init__(self, *layers):
        super().__init__()
        self.layer_list = []
        for layer in layers:
            self.layer_list.append(layer)
        self.out = np.zeros_like(self.layer_list[0].out)
        
    def add_backward(self, network):
        for layer in self.layer_list:
            if not network.has_node(layer):
              network.add_node(layer)
              network.node_labels[self] = 'Operator'
            if not network.has_edge(layer, self):
              network.add_edge(layer, self)

class UnaryOperator(Layer):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.out = np.zeros(layer.size())
        
    def add_backward(self, network):
        if not network.has_node(self.layer):
          network.add_node(self.layer)
          network.node_labels[self] = 'UnaryOperator'
        if not network.has_edge(self.layer, self):
          network.add_edge(self.layer, self)
            
class Add(Operator):
    def __init__(self, *layers):
      super().__init__(*layers)
      self.ones_grad = np.ones_like(self.layer_list[0].out, dtype=np.float64)
      
    def forward(self):
        self.out = None
        for layer in self.layer_list:
            if self.out is None:
                self.out = layer.out
            else:
                self.out = self.out + layer.out
        
    def backward(self):
        super().backward()
        for i in range(len(self.layer_list)):
            self.d[self.layer_list[i]] = self.ones_grad[:]

class Multiply(Operator):    
    def forward(self):
        self.out = None
        for layer in self.layer_list:
            if self.out is None:
                self.out = layer.out
            else:
                self.out = self.out * layer.out
        
    def backward(self):
        super().backward()
        for layer in self.layer_list:
            self.d[layer] = self.out / (layer.out + Layer.EPSILON)

class ReduceSum(Layer):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.out = np.zeros((1,))
        self.ones_grad = np.ones_like(self.layer.out, dtype=np.float64)
    
    def add_backward(self, network):
        if not network.has_node(self.layer):
          network.add_node(self.layer)
          network.node_labels[self] = 'ReduceSum'
        if not network.has_edge(self.layer, self):
          network.add_edge(self.layer, self)
    
    def forward(self):
        self.out = np.sum(self.layer.out)
    
    def backward(self):
        super().backward()
        self.d[self.layer] = self.ones_grad[:]

class Reciprocal(UnaryOperator):
    def forward(self):
        self.out = np.power(self.layer.out, -1)
    
    def backward(self):
        super().backward()
        self.d[self.layer] = -np.power(self.layer.out*self.layer.out+Layer.EPSILON, -1)
        
class Exp(UnaryOperator):
    def forward(self):
        self.out = np.exp(np.clip( self.layer.out, -1000, 700 ))
    
    def backward(self):
        super().backward()
        self.d[self.layer] = np.exp(np.clip( self.layer.out, -1000, 700 ))
        
class Sigmoid(UnaryOperator):   
    def forward(self):
        self.out = 1.0/(1+np.exp(np.clip( -self.layer.out, -1000, 700 )))
        
    def backward(self):
        super().backward()
        self.d[self.layer] = self.out*(1-self.out)
        
class Identity(UnaryOperator):
    def __init__(self, layer):
      super().__init__(layer)
      self.ones_grad = np.ones_like(self.layer.out, dtype=np.float64)   
      
    def forward(self):
        self.out = self.layer.out
    
    def backward(self):
        super().backward()
        self.d[self.layer] = self.ones_grad[:]

class Log(UnaryOperator):
    def forward(self):
        self.out = np.log(self.layer.out+Layer.EPSILON)
    
    def backward(self):
        super().backward()
        self.d[self.layer] = 1.0/(self.layer.out+Layer.EPSILON)

class ReLU(UnaryOperator):
    def forward(self):
        self.out = np.maximum(self.layer.out, 0)
    
    def backward(self):
        super().backward()
        self.d[self.layer] = (self.out>0)*1.0

class Ensemble(Layer):
    def __init__(self, layer_list):
        super().__init__()
        self.layer_list = layer_list
        size = self.layer_list[0].size()
        size = (len(self.layer_list),) + size
        self.out = np.zeros(size, dtype=np.float64)
        self.graph = self.layer_list[0].graph
    
    def forward(self):
      for i in range(len(self.layer_list)):
        self.out[i] = self.layer_list[i].out
        
    def backward(self):
        super().backward()
        last = self.graph.last
        for i in range(len(self.layer_list)):
          self.d[self.layer_list[i]] = last.d[self][i]
        last.d[self] = 1
        
    def add_backward(self, network):
        for layer in self.layer_list:
            if not network.has_node(layer):
              network.add_node(layer)
              network.node_labels[self] = 'Ensemble'
            if not network.has_edge(layer, self):
              network.add_edge(layer, self)
            
class Max(Layer):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.out = np.zeros((layer.out.shape[-1],), dtype=np.float64)
        self.zeros_grad = np.zeros_like(self.layer.out)
    
    def add_backward(self, network):
        if not network.has_node(self.layer):
          network.add_node(self.layer)
          network.node_labels[self] = 'Max'
        if not network.has_edge(self.layer, self):
          network.add_edge(self.layer, self)
    
    def forward(self):
        self.out = np.max(self.layer.out, axis=(0,1))
    
    def backward(self):
        super().backward()
        grad = np.zeros_like(self.layer.out, dtype=np.float64)
        for i in range(self.layer.out.shape[-1]):
            grad_ch = grad[:,:,i]
            grad_ravel = grad_ch.ravel()
            grad_ravel[grad_ch.argmax()] = 1
            grad_ch = grad_ravel.reshape(grad_ch.shape)
            grad[:,:,i] = grad_ch[:]
        self.d[self.layer] = grad
        
class Flatten(UnaryOperator):
    def __init__(self, layer):
        super().__init__(layer)
        self.out = self.layer.out.ravel()
    
    def forward(self):
        self.out = self.layer.out.ravel()
        
    def backward(self):
        super().backward()
        last = self.graph.last
        self.d[self.layer] = last.d[self].reshape(self.layer.out.shape)
        last.d[self] = 1
        
class Reshape(UnaryOperator):
    def __init__(self, layer, shape):
        super().__init__(layer)
        self.out = self.layer.out.reshape(shape)
        self.shape = shape
    
    def forward(self):
        self.out = self.layer.out.reshape(self.shape)
        
    def backward(self):
        super().backward()
        last = self.graph.last
        self.d[self.layer] = last.d[self].reshape(self.layer.out.shape)
        last.d[self] = 1

class Cut2D(UnaryOperator):
    def __init__(self, layer, location=(0,0), size=(3,3)):
        super().__init__(layer)
        self.location = location
        self.filter_size = size + (layer.size()[-1],)
        self.out = np.zeros(self.filter_size)
        self.zeros_grad = np.zeros(self.layer.size(), dtype=np.float64)
    
    def forward(self):
        self.out = self.layer.out[self.location[0]:self.location[0]+self.filter_size[0],
                                  self.location[1]:self.location[1]+self.filter_size[1]]
        
    def backward(self):
        super().backward()
        last = self.graph.last
        grad = self.zeros_grad[:]
        grad[self.location[0]:self.location[0]+self.filter_size[0],
             self.location[1]:self.location[1]+self.filter_size[1],:] = last.d[self]
        self.d[self.layer] = grad
        last.d[self] = 1
        
class Conv2DVector(Ensemble):
    def __init__(self, layer, filter_layer, bias, filter_size=(3,3), stride=1, location=0):
        layer_list = []
        for i in range(int( (layer.size()[1]-filter_size[1])/stride + 1)):
            cut_layer = Cut2D(layer, location=(location,i*stride), size=filter_size)
            reduce_sum = Add(bias, ReduceSum(Multiply(cut_layer,filter_layer)))
            layer_list.append(reduce_sum)
        super().__init__(layer_list)
        
class MaxPool2DVector(Ensemble):
    def __init__(self, layer, filter_size=(2,2),stride=2, location=0):
        layer_list = []
        for i in range(int( (layer.size()[1]-filter_size[1])/stride + 1)):
            layer_list.append(Max(Cut2D(layer, location=(location,i*stride), size=filter_size)))
        super().__init__(layer_list)

class Filter(Ensemble):
    def __init__(self, layer, filter_size=(3,3), stride=1, initializer=GlorotNormal()):
        layer_list = []
        W = initializer.get_weights((filter_size[0], filter_size[1], layer.size()[-1]))
        filter_layer = Weight(initial_value=W)
        bias = Weight(initial_value=np.zeros((1,)))
        for i in range(int( (layer.size()[0]-filter_size[0])/stride + 1)):
          conv2dvec = Conv2DVector(layer,  filter_layer, bias, filter_size=filter_size, stride=stride, location=i*stride)
          layer_list.append(conv2dvec)
        super().__init__(layer_list)
        
class MaxPool2D(Ensemble):
    def __init__(self, layer, filter_size=(2,2), stride=2):
        layer_list = []
        for i in range(int( (layer.size()[0]-filter_size[0])/stride + 1)):
            pool_vector = MaxPool2DVector(layer, filter_size=filter_size, stride=stride, location=i*stride )
            layer_list.append(pool_vector)
        super().__init__(layer_list)

class Conv2D(Layer):
    def __init__(self, layer, out_channel, filter_size=(3,3), stride=1, initializer=GlorotUniform()):
        super().__init__()
        layer_list = []
        for i in range(out_channel):
            filt = Filter(layer, filter_size=filter_size, stride=stride, initializer=initializer)
            filt = Reshape(filt, filt.size()[:-1])
            layer_list.append(filt)
        self.layer_list = layer_list
        size = self.layer_list[0].size() + (out_channel,)
        self.out = np.zeros(size, dtype=np.float64)
    
    def forward(self):
        for i in range(len(self.layer_list)):
            self.out[:,:,i] = self.layer_list[i].out[:,:]
        
    def backward(self):
        super().backward()
        last = self.graph.last
        for i in range(len(self.layer_list)):
            self.d[self.layer_list[i]] = last.d[self][:,:,i]
        last.d[self] = 1
        
    def add_backward(self, network):
        for layer in self.layer_list:
            if not network.has_node(layer):
              network.add_node(layer)
              network.node_labels[self] = 'Conv2D'
            if not network.has_edge(layer, self):
              network.add_edge(layer, self)

class MatMul(Ensemble):
    def __init__(self, W, layer):
        layer_list = []
        for i in range(W.shape[0]):
            layer_list.append(
                    ReduceSum(Multiply(Weight(initial_value=W[i,:]),
                                       layer)))
        super().__init__(layer_list)
        
class Dense(Ensemble):
    def __init__(self, layer, n, initializer=GlorotNormal()):
        W = initializer.get_weights((n, layer.size()[0]))
        bias = Weight(initial_value=np.zeros((n,1)))
        layer_list = []
        matmul = MatMul(W, layer)
        layer_list.append(Add(matmul,bias))
        super().__init__(layer_list)
        
class RepeatingVector(Ensemble):
    def __init__(self, layer, n):
        layer_list = []
        for i in range(n):
            layer_list.append(layer)
        super().__init__(layer_list)
        
class SubtractMax(Layer):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.out = np.zeros(layer.size(), dtype=np.float64)
    
    def add_backward(self, network):
        if not network.has_node(self.layer):
          network.add_node(self.layer)
        if not network.has_edge(self.layer, self):
          network.add_edge(self.layer, self)
    
    def forward(self):
        self.out = self.layer.out-np.max(self.layer.out)
    
    def backward(self):
        super().backward()
        self.d[self.layer] = np.ones_like(self.layer.out)

class Softmax(Ensemble):
    def __init__(self, layer):
        layer_list = []
        layer_sub = SubtractMax(layer)
        exp = Exp(layer_sub)
        inv = Reciprocal(ReduceSum(exp))
        inv = RepeatingVector(inv, exp.size()[0])
        layer_list.append(Multiply(exp, inv))
        super().__init__(layer_list)
        
class Network(nx.DiGraph):
    path = '/content/drive/My Drive/SL_Project/model'
    def __init__(self, name='network'):
        super().__init__()
        self.last = None
        self.node_labels = {}
        self.name = name
        self.model_dict = {}
        self.model_dict['smallest_valid_error'] = -1
        self.model_dict['valid_error_list'] = []
        
    def get_params(self):
      params = []
      for node in nx.topological_sort(network):
            if isinstance(node,Weight):
                params.append(node)
      return params
    
    def forward(self):
        for node in nx.topological_sort(self):
            node.forward()
        
    def backward(self):
        topological_sorted = list(nx.topological_sort(self))
        self.last = topological_sorted[-1]
        for node in reversed(topological_sorted):
            node.backward()
    
    def add_node(self, node):
        super().add_node(node)
        node.graph = self
        node.add_backward(self)
        
    def save_model(self):
        i = 0
        for node in nx.topological_sort(self):
            if isinstance(node, Weight):
                self.model_dict[str(i)] = node.out
                i = i + 1
        np.save(os.path.join(Network.path, self.name+'.npy'), self.model_dict, allow_pickle=True)
        
    def save_validation(self):
        i = 0
        for node in nx.topological_sort(self):
            if isinstance(node, Weight):
                self.model_dict[str(i)+'_valid'] = node.out
                i = i + 1
        
    def load_model(self):
        self.model_dict = np.load(os.path.join(Network.path, self.name+'.npy'), allow_pickle=True).item()
        i = 0
        for node in nx.topological_sort(self):
            if isinstance(node, Weight):
                node.out = self.model_dict[str(i)]
                i = i + 1
        print('Model loaded from memory.')
        
    def load_validation(self):
        self.load_model()
        i = 0
        for node in nx.topological_sort(self):
            if isinstance(node, Weight):
                node.out = self.model_dict[str(i)+'_valid']
                i = i + 1
        print('Smallest validation model loaded from memory.')
        
    def predict(self, x, y, x_test):
      y_pred = np.empty((x_test.shape[0],)+y.size())
      for i in range(x_test.shape[0]):
        x_feed = x_test[i,:].reshape(x.size())
        x.feed(x_feed)
        self.forward()
        y_pred[i,:] = y.out
      return y_pred
        
    def train(self, x, y, x_train, y_train, optimizer,
              x_validation = None, y_validation = None,
              num_of_epoch=5, batch_size=None, save=True):
      for epoch in range(1,num_of_epoch+1):
          i = 0
          ETA = 0
          total_time = 0
          if batch_size is None:
            batch_size = x_train.shape[0]
          while i < x_train.shape[0]:
              x_batch = x_train[i:i+batch_size, :]
              y_batch = y_train[i:i+batch_size, :]
              i = i + batch_size
              optimizer.zero_grad()
              sys.stdout.flush()
              sys.stdout.write('\r')
              percent = int(100*(i-batch_size)/x_train.shape[0])
              sys.stdout.write("EPOCH %d/%d [%-20s] %d%% loss: %f ETA: %d s" % (epoch, num_of_epoch,int(percent/5)*'=', percent, optimizer.loss_value, ETA))
              start = time.time()
              for j in range(x_batch.shape[0]):
                  x_feed = x_batch[j,:].reshape(x.size())
                  y_feed = y_batch[j,:].reshape(y.size())
                  x.feed(x_feed)
                  y.feed(y_feed)
                  self.forward()
                  self.backward()
                  optimizer.batch()
              end = time.time()
              total_time = total_time + int(end-start)
              sys.stdout.flush()
              percent = int(100*(i-batch_size+x_batch.shape[0])/x_train.shape[0])
              if percent == 0:
                ETA = 0
              else:
                ETA = int(100*total_time/percent)-total_time
              sys.stdout.write('\r')
              sys.stdout.write("EPOCH %d/%d [%-20s] %d%% loss: %f ETA: %d s" % (epoch, num_of_epoch,int(percent/5)*'=', percent, optimizer.loss_value, ETA))
              optimizer.step()
          optimizer.epoch()
          if x_validation is not None and y_validation is not None:
            valid_loss = 0
            for j in range(x_validation.shape[0]):
              x_feed = x_validation[j,:].reshape(x.size())
              y_feed = y_validation[j,:].reshape(y.size())
              x.feed(x_feed)
              y.feed(y_feed)
              self.forward()
              valid_loss = valid_loss + self.last.out[0]
            sys.stdout.write(' validation loss: %f' % (valid_loss))
            self.model_dict['valid_error_list'].append(valid_loss)
            if self.model_dict['smallest_valid_error'] < 0 or valid_loss<self.model_dict['smallest_valid_error']:
              self.model_dict['smallest_valid_error'] = valid_loss
              self.save_validation()
          sys.stdout.write('\n')
      if save:
        optimizer.save_history()