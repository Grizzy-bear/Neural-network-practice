# encoding: utf-8
"""
@version:3.6
@author:lamplight
@file:02.py
@time:2017/12/1916:46
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# V_data = [1.,2.,3.]
# V = torch.Tensor(V_data)
# print(V)
# x = torch.randn((3, 4, 5))
# print(x)

lin = nn.Linear(5,3)
data = autograd.Variable(torch.randn(2, 5))
print(data)