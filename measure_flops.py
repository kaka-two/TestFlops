# """
# This file is used to measure the flop of DNN models.
# before using, "pip3 install thop" should be used.
# """
import torch
from torchvision import models
from thop import profile
import math

batch = 7
size = 224
# update the modelP
net = models.densenet121(pretrained=False, progress=True)

# warm up
inputs = torch.randn(1, 3, size, size)
flops, params = profile(net, inputs=(inputs, ))

# test
inputs = torch.randn(batch, 3, size, size)
flops, params = profile(net, inputs=(inputs, ))
flops = flops / batch

# output the flops and the network parameter
print("-"*100)
print(batch)
print("flops: ", round(flops / math.pow(10, 9),3), "G")
print("params: ", params)
