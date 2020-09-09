import torch
import torch.nn as nn
from utils import count_conv_flop
import torch.autograd.profiler as profiler
OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: AvgPool2d(3, C, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: MaxPool2d(3, C, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
}

class AvgPool2d(nn.Module):

  def __init__(self, kernel_size, C, stride, padding, count_include_pad):
    super(AvgPool2d, self).__init__()
    self.op = nn.AvgPool2d(kernel_size, stride=stride, padding=padding, count_include_pad=count_include_pad)
    self.bn = nn.BatchNorm2d(C, affine=False)
    self.block_str = 'AvgPool2d-c_in:{}-c_out:{}-kernel:{}-stride:{}'.format(C, C, kernel_size, stride)
  def forward(self, x):
    input_shape = '-input_shape:' + (','.join([str(x) for x in list(x.size())]))
    with profiler.record_function(self.block_str+input_shape):
      out = self.bn(self.op(x))
    return out

  def get_flops(self, input_size):
    return 0


class MaxPool2d(nn.Module):

  def __init__(self, kernel_size, C, stride, padding):
    super(MaxPool2d, self).__init__()
    self.op = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
    self.bn = nn.BatchNorm2d(C, affine=False)
    self.block_str = 'MaxPool2d-c_in:{}-c_out:{}-kernel:{}-stride:{}'.format(C,C,kernel_size,stride)
  def forward(self, x):
    input_shape = '-input_shape:' + (','.join([str(x) for x in list(x.size())]))
    with profiler.record_function(self.block_str+input_shape):
      out = self.bn(self.op(x))
    return out

  def get_flops(self, input_size):
    return 0


class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.in_channels = C_in
    self.out_channels = C_out
    self.kernel_size = kernel_size
    self.stride = stride

    self.r_1 = nn.ReLU(inplace=False)
    self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)
    self.block_str = 'DilConv-c_in:{}-c_out:{}-kernel:{}-stride{}'.format(C_in, C_out, kernel_size, stride)

  def forward(self, x):
    input_shape = '-input_shape:' + (','.join([str(x) for x in list(x.size())]))
    with profiler.record_function(self.block_str+input_shape):
      x = self.r_1(x)
      fea = self.conv(x)
      out = self.bn(fea)
    return out

  def get_flops(self, input_size):
    flop = count_conv_flop(self.conv, input_size)
    print('ReLBN:   Flops:  {:.2e}, input_size: {} ,in_c : {} , out_c: {} , kernel: {} , stride: {} '.format(flop, input_size, self.in_channels, self.out_channels, self.kernel_size, self.stride))
    return flop


class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.in_channels = C_in
    self.out_channels = C_out
    self.kernel_size = kernel_size
    self.stride = stride
    self.r_1 = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)
    self.block_str = 'DilConv-c_in:{}-c_out:{}-kernel:{}-stride{}-dilation:{}'.format(C_in, C_out, kernel_size, stride, dilation)

  def forward(self, x):
    input_shape = '-input_shape:' + (','.join([str(x) for x in list(x.size())]))
    with profiler.record_function(self.block_str+input_shape):
      x = self.r_1(x)
      fea_1 = self.conv_1(x)
      fea_2 = self.conv_2(fea_1)
      out = self.bn(fea_2)
    return out

  def get_flops(self, input_size):
    flop_1 = count_conv_flop(self.conv_1, input_size)
    if self.stride>1:
      input_size = (input_size[0]//self.stride, input_size[1]//self.stride)
    flop_2 = count_conv_flop(self.conv_2, input_size)
    print(
      'DilConv:   Flops:  {:.2e}, input_size: {} ,in_c : {} , out_c: {} , kernel: {} , stride: {} '.format(flop_1+flop_2, input_size,
                                                                                                     self.in_channels,
                                                                                                     self.out_channels,
                                                                                                     self.kernel_size,
                                                                                                     self.stride))

    return flop_1+flop_2



class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()

    self.in_channels = C_in
    self.out_channels = C_out
    self.kernel_size = kernel_size
    self.stride = stride

    self.r_1 = nn.ReLU(inplace=False)
    self.conv_1_1 = nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False)
    self.conv_1_2 = nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False)
    self.bn_1 = nn.BatchNorm2d(C_in, affine=affine)
    self.r_2 = nn.ReLU(inplace=False)
    self.conv_2_1 = nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False)
    self.conv_2_2 = nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)
    self.bn_2 = nn.BatchNorm2d(C_out, affine=affine)
    self.block_str = 'SepConv-c_in:{}-c_out:{}-kernel:{}-stride{}'.format(C_in, C_out, kernel_size,stride)

  def forward(self, x):
    input_shape = '-input_shape:' + (','.join([str(x) for x in list(x.size())]))
    with profiler.record_function(self.block_str+input_shape):
      x = self.r_1(x)
      fea_1 = self.conv_1_1(x)
      fea_1 = self.conv_1_2(fea_1)
      fea_1 = self.bn_1(fea_1)

      x = self.r_2(fea_1)
      fea_2 = self.conv_2_1(x)
      fea_2 = self.conv_2_2(fea_2)
      out = self.bn_2(fea_2)
    return out

  def get_flops(self, input_size):
    f_depth_1 = count_conv_flop(self.conv_1_1, input_size)
    if self.stride>1:
      input_size = (input_size[0]//self.stride, input_size[1]//self.stride)
    f_point_1 = count_conv_flop(self.conv_1_2, input_size)
    f_depth_2 = count_conv_flop(self.conv_2_1, input_size)
    f_point_2 = count_conv_flop(self.conv_2_2, input_size)
    f_sum = f_depth_1+f_depth_2+f_point_1+f_point_2
    print(
      'SepConv:   Flops:  {:.2e}, input_size: {} ,in_c : {} , out_c: {} , kernel: {} , stride: {} '.format(f_sum,
                                                                                                       input_size,
                                                                                                       self.in_channels,
                                                                                                       self.out_channels,
                                                                                                       self.kernel_size,
                                                                                                       self.stride))
    return f_sum



class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x
  def get_flops(self, input_size):
    return 0


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)

  def get_flops(self, input_size):
    return 0


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)
    self.block_str = 'FactorizedReduce-c_in:{}-c_out:{}-stride:{}'.format(C_in, C_out,2)

  def forward(self, x):
    input_shape = '-input_shape:' + (','.join([str(x) for x in list(x.size())]))
    with profiler.record_function(self.block_str+input_shape):
      x = self.relu(x)
      out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
      out = self.bn(out)
    return out

  def get_flops(self, input_size):
    f_1 = count_conv_flop(self.conv_1, input_size)
    f_2 = count_conv_flop(self.conv_2,input_size)
    return f_1+f_2

