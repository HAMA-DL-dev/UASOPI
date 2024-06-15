"""

Reference : https://github.com/ezjong/lightprobnets/blob/master/contrib/adf.py

ADD belows for ADF 
- BatchNorm 1d
- spconv.SparseConv3d
- spconv.SubMConv3d 

"""
import operator
from collections import OrderedDict
from itertools import islice
from numbers import Number
import numpy as np
import torch
import torch.nn as nn

from torch.nn import init
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.conv import _ConvTransposeMixin
from torch.nn.modules.utils import _pair

import math
import time
import sys
from typing import List, Optional, Tuple, Union

from spconv import pytorch as spconv
from spconv.core import ConvAlgo
from spconv.debug_utils import spconv_save_debug_data
from spconv.pytorch import functional as Fsp
from spconv.pytorch import ops
from spconv.cppconstants import CPU_ONLY_BUILD
from spconv.pytorch.core import IndiceData, SparseConvTensor, ImplicitGemmIndiceData, expand_nd
from spconv.pytorch.modules import SparseModule
from spconv.constants import FILTER_HWIO
from spconv.utils import nullcontext
from torch.nn.init import calculate_gain

from spconv.pytorch.conv import SparseConvolution

def keep_variance_fn(x):
    return x + 2e-7

def resize2D(inputs, size_targets, mode="bilinear"):
    size_inputs = [inputs.size(2), inputs.size(3)]

    if all([size_inputs == size_targets]):
        return inputs  # nothing to do
    elif any([size_targets < size_inputs]):
        resized = F.adaptive_avg_pool2d(inputs, size_targets)  # downscaling
    else:
        resized = F.upsample(inputs, size=size_targets, mode=mode)  # upsampling

    # correct scaling
    return resized


def resize2D_as(inputs, output_as, mode="bilinear"):
    size_targets = [output_as.size(2), output_as.size(3)]
    return resize2D(inputs, size_targets, mode=mode)


def normcdf(value, mu=0.0, stddev=1.0):
    sinv = (1.0 / stddev) if isinstance(stddev, Number) else stddev.reciprocal()
    return 0.5 * (1.0 + torch.erf((value - mu) * sinv / np.sqrt(2.0)))


def _normal_log_pdf(value, mu, stddev):
    var = (stddev ** 2)
    log_scale = np.log(stddev) if isinstance(stddev, Number) else torch.log(stddev)
    return -((value - mu) ** 2) / (2.0*var) - log_scale - np.log(np.sqrt(2.0*np.pi))


def normpdf(value, mu=0.0, stddev=1.0):
    return torch.exp(_normal_log_pdf(value, mu, stddev))


class AvgPool2d(nn.Module):
    def __init__(self, keep_variance_fn=None,kernel_size=2):
        super(AvgPool2d, self).__init__()
        self._keep_variance_fn = keep_variance_fn
        self.kernel_size = kernel_size

    def forward(self, inputs_mean, inputs_variance):
        outputs_mean = F.avg_pool2d(inputs_mean, self.kernel_size,stride=2,padding=1)
        outputs_variance = F.avg_pool2d(inputs_variance, self.kernel_size,stride=2,padding=1)
        outputs_variance = outputs_variance / (inputs_mean.size(2) * inputs_mean.size(3))

        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)

        # TODO: avg pooling means that every neuron is multiplied by the same
        #       weight, that is 1/number of neurons in the channel
        #      outputs_variance*1/(H*W) should be enough already

        return outputs_mean, outputs_variance/(inputs_mean.shape[2]*inputs_mean.shape[3])


class Softmax(nn.Module):
    def __init__(self, dim=1, keep_variance_fn=None):
        super(Softmax, self).__init__()
        self.dim = dim
        self._keep_variance_fn = keep_variance_fn

    def forward(self, features_mean, features_variance, eps=1e-5):
        """Softmax function applied to a multivariate Gaussian distribution.
        It works under the assumption that features_mean and features_variance
        are the parameters of a the indepent gaussians that contribute to the
        multivariate gaussian.
        Mean and variance of the log-normal distribution are computed following
        https://en.wikipedia.org/wiki/Log-normal_distribution."""

        log_gaussian_mean = features_mean + 0.5 * features_variance
        log_gaussian_variance = 2 * log_gaussian_mean

        log_gaussian_mean = torch.exp(log_gaussian_mean)
        log_gaussian_variance = torch.exp(log_gaussian_variance)
        log_gaussian_variance = log_gaussian_variance * (torch.exp(features_variance) - 1)

        constant = torch.sum(log_gaussian_mean, dim=self.dim) + eps
        constant = constant.unsqueeze(self.dim)
        outputs_mean = log_gaussian_mean / constant
        outputs_variance = log_gaussian_variance / (constant ** 2)

        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance


class ReLU(nn.Module):
    def __init__(self, keep_variance_fn=None, type_conv = None):
        super(ReLU, self).__init__()
        self._keep_variance_fn = keep_variance_fn
        self.type_conv = type_conv
        
    def forward(self, features_mean, features_variance):
        
        features_variance = torch.abs(features_variance)
        # features_variance = torch.clamp(features_variance, min=2e-7)
        features_stddev = torch.sqrt(features_variance) # torch.sqrt(features_variance)  # torch.log(features_variance)
        
        div = features_mean / features_stddev
        div = torch.where(features_mean == 0, torch.tensor(1e-10, device=features_mean.device), div)
        
        pdf = normpdf(div)
        cdf = normcdf(div)
        outputs_mean = features_mean * cdf + features_stddev * pdf
        outputs_variance = (features_mean ** 2 + features_variance) * cdf \
                           + features_mean * features_stddev * pdf - outputs_mean ** 2
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        # if self.type_conv is not None:
            # breakpoint()
        return outputs_mean, outputs_variance


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01, keep_variance_fn=None):
        super(LeakyReLU, self).__init__()
        self._keep_variance_fn = keep_variance_fn
        self._negative_slope = negative_slope

    def forward(self, features_mean, features_variance):
        features_stddev = torch.sqrt(features_variance)
        div = features_mean / features_stddev
        pdf = normpdf(div)
        cdf = normcdf(div)
        negative_cdf = 1.0 - cdf
        mu_cdf = features_mean * cdf
        stddev_pdf = features_stddev * pdf
        squared_mean_variance = features_mean ** 2 + features_variance
        mean_stddev_pdf = features_mean * stddev_pdf
        mean_r = mu_cdf + stddev_pdf
        variance_r = squared_mean_variance * cdf + mean_stddev_pdf - mean_r ** 2
        mean_n = - features_mean * negative_cdf + stddev_pdf
        variance_n = squared_mean_variance * negative_cdf - mean_stddev_pdf - mean_n ** 2
        covxy = - mean_r * mean_n
        outputs_mean = mean_r - self._negative_slope * mean_n
        outputs_variance = variance_r \
                           + self._negative_slope * self._negative_slope * variance_n \
                           - 2.0 * self._negative_slope * covxy
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
            
        return outputs_mean, outputs_variance


class Dropout(nn.Module):
    """ADF implementation of nn.Dropout2d"""

    def __init__(self, p: float = 0.5, keep_variance_fn=None, inplace=False):
        super(Dropout, self).__init__()
        self._keep_variance_fn = keep_variance_fn
        self.inplace = inplace
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p

    def forward(self, inputs_mean, inputs_variance):
        if self.training:
            binary_mask = torch.ones_like(inputs_mean)
            binary_mask = F.dropout2d(binary_mask, self.p, self.training, self.inplace)

            outputs_mean = inputs_mean * binary_mask
            outputs_variance = inputs_variance * binary_mask ** 2

            if self._keep_variance_fn is not None:
                outputs_variance = self._keep_variance_fn(outputs_variance)
            return outputs_mean, outputs_variance

        outputs_variance = inputs_variance
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return inputs_mean, outputs_variance


class MaxPool2d(nn.Module):
    def __init__(self, keep_variance_fn=None):
        super(MaxPool2d, self).__init__()
        self._keep_variance_fn = keep_variance_fn

    def _max_pool_internal(self, mu_a, mu_b, var_a, var_b):
        stddev = torch.sqrt(var_a + var_b)
        ab = mu_a - mu_b
        alpha = ab / stddev
        pdf = normpdf(alpha)
        cdf = normcdf(alpha)
        z_mu = stddev * pdf + ab * cdf + mu_b
        z_var = ((mu_a + mu_b) * stddev * pdf +
                 (mu_a ** 2 + var_a) * cdf +
                 (mu_b ** 2 + var_b) * (1.0 - cdf) - z_mu ** 2)
        if self._keep_variance_fn is not None:
            z_var = self._keep_variance_fn(z_var)
        return z_mu, z_var

    def _max_pool_1x2(self, inputs_mean, inputs_variance):
        mu_a = inputs_mean[:, :, :, 0::2]
        mu_b = inputs_mean[:, :, :, 1::2]
        var_a = inputs_variance[:, :, :, 0::2]
        var_b = inputs_variance[:, :, :, 1::2]
        outputs_mean, outputs_variance = self._max_pool_internal(
            mu_a, mu_b, var_a, var_b)
        return outputs_mean, outputs_variance

    def _max_pool_2x1(self, inputs_mean, inputs_variance):
        mu_a = inputs_mean[:, :, 0::2, :]
        mu_b = inputs_mean[:, :, 1::2, :]
        var_a = inputs_variance[:, :, 0::2, :]
        var_b = inputs_variance[:, :, 1::2, :]
        outputs_mean, outputs_variance = self._max_pool_internal(
            mu_a, mu_b, var_a, var_b)
        return outputs_mean, outputs_variance

    def forward(self, inputs_mean, inputs_variance):
        z_mean, z_variance = self._max_pool_1x2(inputs_mean, inputs_variance)
        outputs_mean, outputs_variance = self._max_pool_2x1(z_mean, z_variance)
        return outputs_mean, outputs_variance


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, keep_variance_fn=None):
        super(Linear, self).__init__()
        self._keep_variance_fn = keep_variance_fn
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs_mean, inputs_variance):
        outputs_mean = F.linear(inputs_mean, self.weight, self.bias)
        outputs_variance = F.linear(inputs_variance, self.weight ** 2, None)
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
            
        return outputs_mean, outputs_variance

class BatchNorm1d(nn.Module):
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked']

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, keep_variance_fn=None):
        super(BatchNorm1d, self).__init__()
        self._keep_variance_fn = keep_variance_fn
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.uniform_(self.weight)
            nn.init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, inputs_mean, inputs_variance):
        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        # if torch.isnan(inputs_mean).any():
        #     breakpoint()
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
                    
        outputs_mean = F.batch_norm(
            inputs_mean, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        outputs_variance = inputs_variance
        weight = self.weight.unsqueeze(0)  # (1, num_features)
        try:
            # print(outputs_variance.shape, weight.shape)
            # print(inputs_mean, inputs_variance)
            outputs_variance = outputs_variance * (weight ** 2)
        except:
            breakpoint()

        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        # print("\n")
        return outputs_mean, outputs_variance
        
class BatchNorm2d(nn.Module):
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked']

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, keep_variance_fn=None):
        super(BatchNorm2d, self).__init__()
        self._keep_variance_fn = keep_variance_fn
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.uniform_(self.weight)
            nn.init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, inputs_mean, inputs_variance):

        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        outputs_mean = F.batch_norm(
            inputs_mean, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        outputs_variance = inputs_variance
        weight = ((self.weight.unsqueeze(0)).unsqueeze(2)).unsqueeze(3)
        outputs_variance = outputs_variance * weight ** 2
        """
        for i in range(outputs_variance.size(1)):
            outputs_variance[:,i,:,:]=outputs_variance[:,i,:,:].clone()*self.weight[i]**2
        """
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
            
        return outputs_mean, outputs_variance

class ZeroPad2d(nn.Module):
    def __init__(self, padding):
        super(ZeroPad2d, self).__init__()
        self.padding = padding
        self.zeropad = nn.ZeroPad2d(self.padding)
        
    def forward(self, inputs_mean, inputs_variance):
        return self.zeropad(inputs_mean), self.zeropad(inputs_variance) 

class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 keep_variance_fn=None, padding_mode='zeros'):
        self._keep_variance_fn = keep_variance_fn
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def forward(self, inputs_mean, inputs_variance):
        outputs_mean = F.conv2d(
            inputs_mean, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        outputs_variance = F.conv2d(
            inputs_variance, self.weight ** 2, None, self.stride, self.padding, self.dilation, self.groups)
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance


class ConvTranspose2d(_ConvTransposeMixin, _ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1,
                 keep_variance_fn=None, padding_mode='zeros'):
        self._keep_variance_fn = keep_variance_fn
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super(ConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode)

    def forward(self, inputs_mean, inputs_variance, output_size=None):
        # FIXME : missing 1 required positional argument: 'num_spatial_dims'
        # output_padding = self._output_padding(inputs_mean, output_size, self.stride, self.padding, self.kernel_size)
        num_spatial_dims = 2
        output_padding = self._output_padding(inputs_mean, output_size, self.stride, self.padding, self.kernel_size, num_spatial_dims) 
        # [reference] : https://github.com/Deci-AI/super-gradients/issues/1045#issuecomment-2145065766
        # [reference] : https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py#L943
        
        outputs_mean = F.conv_transpose2d(
            inputs_mean, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
        outputs_variance = F.conv_transpose2d(
            inputs_variance, self.weight ** 2, None, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
            
        return outputs_mean, outputs_variance


def concatenate_as(tensor_list, tensor_as, dim, mode="bilinear"):
    means = [resize2D_as(x[0], tensor_as[0], mode=mode) for x in tensor_list]
    variances = [resize2D_as(x[1], tensor_as[0], mode=mode) for x in tensor_list]
    means = torch.cat(means, dim=dim)
    variances = torch.cat(variances, dim=dim)
    return means, variances


class Sequential(nn.Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Sequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, inputs, inputs_variance):
        for module in self._modules.values():
            inputs, inputs_variance = module(inputs, inputs_variance)

        return inputs, inputs_variance

# spconv
from spconv.pytorch.modules import is_spconv_module, SparseModule 
from ...OpenPCDet.pcdet.utils.spconv_utils import replace_feature_adf

class SparseSequential(SparseModule):
    def __init__(self, *args, **kwargs):
        super(SparseSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)
        # self._sparity_dict = {}

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    # @property
    # def sparity_dict(self):
    #     return self._sparity_dict

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input, variance):
        for k, module in self._modules.items():
            if is_spconv_module(module):  # use SpConvTensor as input
                if isinstance(input, list):
                    input, variance = module(input, variance)
                else:
                    assert isinstance(input, spconv.SparseConvTensor)
                    # self._sparity_dict[k] = input.sparity
                    input, variance = module(input, variance)
            else:
                if isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        # input, _ = input.replace_feature_adf(module(input.features, variance.features)
                        # _, variance = variance.replace_feature_adf(module(input.features, variance.features))
                        input, variance = replace_feature_adf(input, variance, *module(input.features, variance.features))
                else:
                    input, variance = module(input, variance)
        return input, variance

class SubMConv3d(SparseConvolution):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None,
                 algo: Optional[ConvAlgo] = None,
                 fp32_accum: Optional[bool] = None,
                 name=None):
        super(SubMConv3d, self).__init__(3,
                                         in_channels,
                                         out_channels,
                                         kernel_size,
                                         stride,
                                         padding,
                                         dilation,
                                         groups,
                                         bias,
                                         True,
                                         indice_key=indice_key,
                                         algo=algo,
                                         fp32_accum=fp32_accum,
                                         name=name)
    
    def forward(self, input: SparseConvTensor, var: SparseConvTensor):
        assert isinstance(input, SparseConvTensor) and isinstance(var, SparseConvTensor)
        assert input.features.shape[1] == self.in_channels, "channel size mismatch"
        features = input.features
        var_features = var.features
        device = features.device
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        if not self.subm:
            if self.transposed:
                out_spatial_shape = ops.get_deconv_output_size(
                    spatial_shape, self.kernel_size, self.stride, self.padding,
                    self.dilation, self.output_padding)
            else:
                out_spatial_shape = ops.get_conv_output_size(
                    spatial_shape, self.kernel_size, self.stride, self.padding,
                    self.dilation)
        else:
            out_spatial_shape = spatial_shape
        
        out_tensor = input.shadow_copy()
        var_tensor = var.shadow_copy()
        
        if self.conv1x1:
            if FILTER_HWIO:
                features = torch.mm(
                    input.features,
                    self.weight.view(self.out_channels, self.in_channels).T)
                var_features = torch.mm(
                    var.features,
                    self.weight.view(self.out_channels, self.in_channels).T)
            else:
                features = torch.mm(
                    input.features,
                    self.weight.view(self.in_channels, self.out_channels))
                var_features = torch.mm(
                    var.features,
                    self.weight.view(self.in_channels, self.out_channels))

            if self.bias is not None:
                features += self.bias
                var_features += self.bias
            out_tensor = out_tensor.replace_feature(features)
            var_tensor = var_tensor.replace_feature(var_features)
            out_tensor.spatial_shape = out_spatial_shape
            var_tensor.spatial_shape = out_spatial_shape
            return out_tensor, var_tensor

        indice_dict = input.indice_dict.copy()
        algo = self.algo
        if self.indice_key is not None:
            datas = input.find_indice_pair(self.indice_key)
            if datas is not None:
                msg = "due to limitation of pytorch, you must provide same algo to layers share same indice key."
                assert algo == datas.algo, msg
        
        profile_ctx = nullcontext()
        if input._timer is not None and self._sparse_unique_name:
            profile_ctx = input._timer.namespace(self._sparse_unique_name)
        
        with profile_ctx:
            if algo == ConvAlgo.Native:
                datas = input.find_indice_pair(self.indice_key)
                if datas is not None:
                    assert isinstance(datas, IndiceData)
                if self.inverse:
                    assert datas is not None and self.indice_key is not None
                    assert datas.is_subm is False, "inverse conv can only be used with standard conv and pool ops."

                    outids = datas.indices
                    indice_pairs = datas.indice_pairs
                    indice_pair_num = datas.indice_pair_num
                    out_spatial_shape = datas.spatial_shape
                    assert datas.ksize == self.kernel_size, "inverse conv must have same kernel size as its couple conv"
                else:
                    if self.indice_key is not None and datas is not None:
                        outids = datas.out_indices
                        indice_pairs = datas.indice_pairs
                        indice_pair_num = datas.indice_pair_num
                        assert self.subm, "only support reuse subm indices"
                        self._check_subm_reuse_valid(input, spatial_shape, datas)
                    else:
                        if input.benchmark:
                            torch.cuda.synchronize()
                            t = time.time()
                        try:
                            outids, indice_pairs, indice_pair_num = ops.get_indice_pairs(
                                indices, batch_size, spatial_shape, algo,
                                self.kernel_size, self.stride, self.padding,
                                self.dilation, self.output_padding, self.subm,
                                self.transposed)
                        except Exception as e:
                            msg = "[Exception|native_pair]"
                            msg += f"indices={indices.shape},bs={batch_size},ss={spatial_shape},"
                            msg += f"algo={algo},ksize={self.kernel_size},stride={self.stride},"
                            msg += f"padding={self.padding},dilation={self.dilation},subm={self.subm},"
                            msg += f"transpose={self.transposed}"
                            print(msg, file=sys.stderr)
                            spconv_save_debug_data(indices)
                            raise e
                        if input.benchmark:
                            torch.cuda.synchronize()
                            interval = time.time() - t
                            out_tensor.benchmark_record[self.name]["indice_gen_time"].append(interval)

                        indice_data = IndiceData(outids, indices, indice_pairs, indice_pair_num,
                                                 spatial_shape, out_spatial_shape,
                                                 is_subm=self.subm, algo=algo,
                                                 ksize=self.kernel_size, stride=self.stride,
                                                 padding=self.padding, dilation=self.dilation)
                        if self.indice_key is not None:
                            msg = f"your indice key {self.indice_key} already exists in this sparse tensor."
                            assert self.indice_key not in indice_dict, msg
                            indice_dict[self.indice_key] = indice_data
                if input.benchmark:
                    torch.cuda.synchronize()
                    t = time.time()
                indice_pairs_calc = indice_pairs
                if indice_pairs.device != features.device:
                    indice_pairs_calc = indice_pairs.to(features.device)
                if self.subm:
                    out_features = Fsp.indice_subm_conv(features, self.weight, indice_pairs_calc,
                                                        indice_pair_num, outids.shape[0], algo, input._timer)
                    var_features = Fsp.indice_subm_conv(var_features, self.weight, indice_pairs_calc,
                                                        indice_pair_num, outids.shape[0], algo, input._timer)
                else:
                    if self.inverse:
                        out_features = Fsp.indice_inverse_conv(features, self.weight, indice_pairs_calc,
                                                               indice_pair_num, outids.shape[0], algo)
                        var_features = Fsp.indice_inverse_conv(var_features, self.weight, indice_pairs_calc,
                                                               indice_pair_num, outids.shape[0], algo)
                    else:
                        out_features = Fsp.indice_conv(features, self.weight, indice_pairs_calc,
                                                       indice_pair_num, outids.shape[0], algo, input._timer)
                        var_features = Fsp.indice_conv(var_features, self.weight, indice_pairs_calc,
                                                       indice_pair_num, outids.shape[0], algo, input._timer)
            else:
                datas = input.find_indice_pair(self.indice_key)
                if datas is not None:
                    assert isinstance(datas, ImplicitGemmIndiceData)
                if self.inverse:
                    assert datas is not None and self.indice_key is not None
                    assert datas.is_subm is False, "inverse conv can only be used with standard conv and pool ops."
                    outids = datas.indices
                    pair_fwd = datas.pair_bwd
                    pair_bwd = datas.pair_fwd
                    pair_mask_fwd_splits = datas.pair_mask_bwd_splits
                    pair_mask_bwd_splits = datas.pair_mask_fwd_splits
                    mask_argsort_fwd_splits = datas.mask_argsort_bwd_splits
                    mask_argsort_bwd_splits = datas.mask_argsort_fwd_splits
                    masks = datas.masks
                    out_spatial_shape = datas.spatial_shape
                    assert datas.ksize == self.kernel_size, "inverse conv must have same kernel size as its couple conv"
                else:
                    if self.indice_key is not None and datas is not None:
                        outids = datas.out_indices
                        pair_fwd = datas.pair_fwd
                        pair_bwd = datas.pair_bwd
                        pair_mask_fwd_splits = datas.pair_mask_fwd_splits
                        pair_mask_bwd_splits = datas.pair_mask_bwd_splits
                        mask_argsort_fwd_splits = datas.mask_argsort_fwd_splits
                        mask_argsort_bwd_splits = datas.mask_argsort_bwd_splits
                        masks = datas.masks
                        assert self.subm, "only support reuse subm indices"
                        self._check_subm_reuse_valid(input, spatial_shape, datas)
                    else:
                        with input._timer.namespace("gen_pairs"):
                            try:
                                res = ops.get_indice_pairs_implicit_gemm(
                                    indices, batch_size, spatial_shape, algo,
                                    ksize=self.kernel_size, stride=self.stride,
                                    padding=self.padding, dilation=self.dilation,
                                    out_padding=self.output_padding, subm=self.subm,
                                    transpose=self.transposed, is_train=(not self.subm) or self.training,
                                    alloc=input.thrust_allocator, timer=input._timer)
                            except Exception as e:
                                msg = "[Exception|implicit_gemm_pair]"
                                msg += f"indices={indices.shape},bs={batch_size},ss={spatial_shape},"
                                msg += f"algo={algo},ksize={self.kernel_size},stride={self.stride},"
                                msg += f"padding={self.padding},dilation={self.dilation},subm={self.subm},"
                                msg += f"transpose={self.transposed}"
                                print(msg, file=sys.stderr)
                                spconv_save_debug_data(indices)
                                raise e

                        outids = res[0]
                        num_inds_per_loc = res[1]
                        pair_fwd = res[2]
                        pair_bwd = res[3]
                        pair_mask_fwd_splits = res[4]
                        pair_mask_bwd_splits = res[5]
                        mask_argsort_fwd_splits = res[6]
                        mask_argsort_bwd_splits = res[7]
                        masks = res[8]
                        if self.indice_key is not None:
                            indice_data = ImplicitGemmIndiceData(
                                outids, indices, pair_fwd, pair_bwd,
                                pair_mask_fwd_splits=pair_mask_fwd_splits,
                                pair_mask_bwd_splits=pair_mask_bwd_splits,
                                mask_argsort_fwd_splits=mask_argsort_fwd_splits,
                                mask_argsort_bwd_splits=mask_argsort_bwd_splits,
                                masks=masks, is_subm=self.subm,
                                spatial_shape=spatial_shape,
                                out_spatial_shape=out_spatial_shape, algo=algo,
                                ksize=self.kernel_size, stride=self.stride,
                                padding=self.padding, dilation=self.dilation)
                            msg = f"your indice key {self.indice_key} already exists in this sparse tensor."
                            assert self.indice_key not in indice_dict, msg
                            indice_dict[self.indice_key] = indice_data
                if input.benchmark:
                    torch.cuda.synchronize()
                    t = time.time()
                num_activate_out = outids.shape[0]
                out_features = Fsp.implicit_gemm(features, self.weight, pair_fwd, pair_bwd,
                                                 pair_mask_fwd_splits, pair_mask_bwd_splits,
                                                 mask_argsort_fwd_splits, mask_argsort_bwd_splits,
                                                 num_activate_out, masks, self.training, self.subm,
                                                 input._timer, self.fp32_accum)
                var_features = Fsp.implicit_gemm(var_features, self.weight**2, pair_fwd, pair_bwd,
                                                 pair_mask_fwd_splits, pair_mask_bwd_splits,
                                                 mask_argsort_fwd_splits, mask_argsort_bwd_splits,
                                                 num_activate_out, masks, self.training, self.subm,
                                                 input._timer, self.fp32_accum)
        
        if self.bias is not None:
            out_features += self.bias
            var_features += self.bias
        
        if input.benchmark:
            torch.cuda.synchronize()
            interval = time.time() - t
            out_tensor.benchmark_record[self.name]["time"].append(interval)
            out_tensor.benchmark_record[self.name]["num_points"].append(features.shape[0])
            out_tensor.benchmark_record[self.name]["num_out_points"].append(out_features.shape[0])
        
        out_tensor = out_tensor.replace_feature(out_features)
        var_tensor = var_tensor.replace_feature(var_features)
        
        out_tensor.indices = outids
        out_tensor.indice_dict = indice_dict
        out_tensor.spatial_shape = out_spatial_shape
        
        var_tensor.indices = outids
        var_tensor.indice_dict = indice_dict
        var_tensor.spatial_shape = out_spatial_shape
        
        return out_tensor, var_tensor
        
class SparseConv3d(SparseConvolution):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None,
                 algo: Optional[ConvAlgo] = None,
                 fp32_accum: Optional[bool] = None,
                 name=None):
        super(SparseConv3d, self).__init__(3,
                                           in_channels,
                                           out_channels,
                                           kernel_size,
                                           stride,
                                           padding,
                                           dilation,
                                           groups,
                                           bias,
                                           indice_key=indice_key,
                                           algo=algo,
                                           fp32_accum=fp32_accum,
                                           name=name)
        
    def forward(self, input: SparseConvTensor, var: SparseConvTensor):
        assert isinstance(input, SparseConvTensor)
        assert input.features.shape[
            1] == self.in_channels, "channel size mismatch"
        features = input.features
        device = features.device
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        if not self.subm:
            if self.transposed:
                out_spatial_shape = ops.get_deconv_output_size(
                    spatial_shape, self.kernel_size, self.stride, self.padding,
                    self.dilation, self.output_padding)
            else:
                out_spatial_shape = ops.get_conv_output_size(
                    spatial_shape, self.kernel_size, self.stride, self.padding,
                    self.dilation)
        else:
            out_spatial_shape = spatial_shape
        # print(self._sparse_unique_name, spatial_shape, out_spatial_shape)
        # input.update_grid(out_spatial_shape)
        # t = time.time()
        out_tensor = input.shadow_copy()
        var_tensor = var.shadow_copy()
        if input.benchmark:
            if self.name is None:
                raise ValueError(
                    "you need to assign name to spmodules before benchmark (spconv.utils.bench.assign_name_to_spmod)"
                )
            if self.name not in input.benchmark_record:
                input.benchmark_record[self.name] = {
                    "type": "SparseConvolution",
                    "indice_gen_time": [],
                    "time": [],
                    "num_points": [],
                    "num_out_points": [],
                    "params": {
                        "kernel_size": self.kernel_size,
                        "stride": self.stride,
                        "padding": self.padding,
                        "dilation": self.dilation,
                        "output_padding": self.output_padding,
                        "subm": self.subm,
                        "transposed": self.transposed,
                        "input_channels": self.in_channels,
                        "out_channels": self.out_channels,
                    }
                }
        if self.conv1x1:
            if FILTER_HWIO:
                features = torch.mm(
                    input.features,
                    self.weight.view(self.out_channels, self.in_channels).T)
            else:
                features = torch.mm(
                    input.features,
                    self.weight.view(self.in_channels, self.out_channels))

            if self.bias is not None:
                features += self.bias
            out_tensor = out_tensor.replace_feature(features)
            var_tensor = var_tensor.replace_feature(features)
            # padding may change spatial shape of conv 1x1.
            out_tensor.spatial_shape = out_spatial_shape
            var_tensor.spatial_shape = out_spatial_shape
            return out_tensor, var_tensor
        indice_dict = input.indice_dict.copy()

        algo = self.algo
        if self.indice_key is not None:
            datas = input.find_indice_pair(self.indice_key)
            if datas is not None:
                msg = "due to limitation of pytorch, you must provide same algo to layers share same indice key."
                assert algo == datas.algo, msg
                # algo = datas.algo
        profile_ctx = nullcontext()
        if input._timer is not None and self._sparse_unique_name:
            profile_ctx = input._timer.namespace(self._sparse_unique_name)
        with profile_ctx:
            if algo == ConvAlgo.Native:
                datas = input.find_indice_pair(self.indice_key)
                if datas is not None:
                    assert isinstance(datas, IndiceData)
                if self.inverse:
                    assert datas is not None and self.indice_key is not None
                    assert datas.is_subm is False, "inverse conv can only be used with standard conv and pool ops."

                    outids = datas.indices
                    indice_pairs = datas.indice_pairs
                    indice_pair_num = datas.indice_pair_num
                    out_spatial_shape = datas.spatial_shape
                    assert datas.ksize == self.kernel_size, "inverse conv must have same kernel size as its couple conv"
                else:
                    if self.indice_key is not None and datas is not None:
                        outids = datas.out_indices
                        indice_pairs = datas.indice_pairs
                        indice_pair_num = datas.indice_pair_num
                        assert self.subm, "only support reuse subm indices"
                        self._check_subm_reuse_valid(input, spatial_shape,
                                                     datas)
                    else:
                        if input.benchmark:
                            torch.cuda.synchronize()
                            t = time.time()
                        try:
                            outids, indice_pairs, indice_pair_num = ops.get_indice_pairs(
                                indices, batch_size, spatial_shape, algo,
                                self.kernel_size, self.stride, self.padding,
                                self.dilation, self.output_padding, self.subm,
                                self.transposed)
                        except Exception as e:
                            msg = "[Exception|native_pair]"
                            msg += f"indices={indices.shape},bs={batch_size},ss={spatial_shape},"
                            msg += f"algo={algo},ksize={self.kernel_size},stride={self.stride},"
                            msg += f"padding={self.padding},dilation={self.dilation},subm={self.subm},"
                            msg += f"transpose={self.transposed}"
                            print(msg, file=sys.stderr)
                            spconv_save_debug_data(indices)
                            raise e
                        if input.benchmark:
                            torch.cuda.synchronize()
                            interval = time.time() - t
                            out_tensor.benchmark_record[
                                self.name]["indice_gen_time"].append(interval)

                        indice_data = IndiceData(outids,
                                                 indices,
                                                 indice_pairs,
                                                 indice_pair_num,
                                                 spatial_shape,
                                                 out_spatial_shape,
                                                 is_subm=self.subm,
                                                 algo=algo,
                                                 ksize=self.kernel_size,
                                                 stride=self.stride,
                                                 padding=self.padding,
                                                 dilation=self.dilation)
                        if self.indice_key is not None:
                            msg = f"your indice key {self.indice_key} already exists in this sparse tensor."
                            assert self.indice_key not in indice_dict, msg
                            indice_dict[self.indice_key] = indice_data
                if input.benchmark:
                    torch.cuda.synchronize()
                    t = time.time()
                indice_pairs_calc = indice_pairs
                if indice_pairs.device != features.device:
                    indice_pairs_calc = indice_pairs.to(features.device)
                if self.subm:
                    out_features = Fsp.indice_subm_conv(
                        features, self.weight, indice_pairs_calc,
                        indice_pair_num, outids.shape[0], algo, input._timer)
                else:
                    if self.inverse:
                        out_features = Fsp.indice_inverse_conv(
                            features, self.weight, indice_pairs_calc,
                            indice_pair_num, outids.shape[0], algo)
                    else:
                        out_features = Fsp.indice_conv(features, self.weight,
                                                       indice_pairs_calc,
                                                       indice_pair_num,
                                                       outids.shape[0], algo,
                                                       input._timer)

            else:
                datas = input.find_indice_pair(self.indice_key)
                if datas is not None:
                    assert isinstance(datas, ImplicitGemmIndiceData)
                if self.inverse:
                    assert datas is not None and self.indice_key is not None
                    assert datas.is_subm is False, "inverse conv can only be used with standard conv and pool ops."
                    outids = datas.indices
                    pair_fwd = datas.pair_bwd
                    pair_bwd = datas.pair_fwd
                    pair_mask_fwd_splits = datas.pair_mask_bwd_splits
                    pair_mask_bwd_splits = datas.pair_mask_fwd_splits
                    mask_argsort_fwd_splits = datas.mask_argsort_bwd_splits
                    mask_argsort_bwd_splits = datas.mask_argsort_fwd_splits
                    masks = datas.masks
                    out_spatial_shape = datas.spatial_shape
                    assert datas.ksize == self.kernel_size, "inverse conv must have same kernel size as its couple conv"
                else:
                    if self.indice_key is not None and datas is not None:
                        outids = datas.out_indices
                        pair_fwd = datas.pair_fwd
                        pair_bwd = datas.pair_bwd
                        pair_mask_fwd_splits = datas.pair_mask_fwd_splits
                        pair_mask_bwd_splits = datas.pair_mask_bwd_splits
                        mask_argsort_fwd_splits = datas.mask_argsort_fwd_splits
                        mask_argsort_bwd_splits = datas.mask_argsort_bwd_splits
                        masks = datas.masks
                        assert self.subm, "only support reuse subm indices"
                        self._check_subm_reuse_valid(input, spatial_shape,
                                                     datas)
                    else:

                        with input._timer.namespace("gen_pairs") if input._timer else nullcontext():
                            try:
                                res = ops.get_indice_pairs_implicit_gemm(
                                    indices,
                                    batch_size,
                                    spatial_shape,
                                    algo,
                                    ksize=self.kernel_size,
                                    stride=self.stride,
                                    padding=self.padding,
                                    dilation=self.dilation,
                                    out_padding=self.output_padding,
                                    subm=self.subm,
                                    transpose=self.transposed,
                                    is_train=(not self.subm) or self.training,
                                    alloc=input.thrust_allocator,
                                    timer=input._timer)
                            except Exception as e:
                                msg = "[Exception|implicit_gemm_pair]"
                                msg += f"indices={indices.shape},bs={batch_size},ss={spatial_shape},"
                                msg += f"algo={algo},ksize={self.kernel_size},stride={self.stride},"
                                msg += f"padding={self.padding},dilation={self.dilation},subm={self.subm},"
                                msg += f"transpose={self.transposed}"
                                print(msg, file=sys.stderr)
                                spconv_save_debug_data(indices)
                                raise e

                        outids = res[0]
                        num_inds_per_loc = res[1]
                        pair_fwd = res[2]
                        pair_bwd = res[3]
                        pair_mask_fwd_splits = res[4]
                        pair_mask_bwd_splits = res[5]
                        mask_argsort_fwd_splits = res[6]
                        mask_argsort_bwd_splits = res[7]
                        masks = res[8]
                        if self.indice_key is not None:
                            indice_data = ImplicitGemmIndiceData(
                                outids,
                                indices,
                                pair_fwd,
                                pair_bwd,
                                pair_mask_fwd_splits=pair_mask_fwd_splits,
                                pair_mask_bwd_splits=pair_mask_bwd_splits,
                                mask_argsort_fwd_splits=mask_argsort_fwd_splits,
                                mask_argsort_bwd_splits=mask_argsort_bwd_splits,
                                masks=masks,
                                is_subm=self.subm,
                                spatial_shape=spatial_shape,
                                out_spatial_shape=out_spatial_shape,
                                algo=algo,
                                ksize=self.kernel_size,
                                stride=self.stride,
                                padding=self.padding,
                                dilation=self.dilation)
                            msg = f"your indice key {self.indice_key} already exists in this sparse tensor."
                            assert self.indice_key not in indice_dict, msg
                            indice_dict[self.indice_key] = indice_data
                if input.benchmark:
                    torch.cuda.synchronize()
                    t = time.time()
                num_activate_out = outids.shape[0]
                out_features = Fsp.implicit_gemm(
                    features, self.weight, pair_fwd, pair_bwd,
                    pair_mask_fwd_splits, pair_mask_bwd_splits,
                    mask_argsort_fwd_splits, mask_argsort_bwd_splits,
                    num_activate_out, masks, self.training, self.subm,
                    input._timer, self.fp32_accum)
                var_features = Fsp.implicit_gemm(
                    features, self.weight**2, pair_fwd, pair_bwd,
                    pair_mask_fwd_splits, pair_mask_bwd_splits,
                    mask_argsort_fwd_splits, mask_argsort_bwd_splits,
                    num_activate_out, masks, self.training, self.subm,
                    input._timer, self.fp32_accum)
        if self.bias is not None:
            out_features += self.bias
        if input.benchmark:
            torch.cuda.synchronize()
            interval = time.time() - t
            out_tensor.benchmark_record[self.name]["time"].append(interval)
            out_tensor.benchmark_record[self.name]["num_points"].append(
                features.shape[0])
            out_tensor.benchmark_record[self.name]["num_out_points"].append(
                out_features.shape[0])
        out_tensor = out_tensor.replace_feature(out_features)
        out_tensor.indices = outids
        out_tensor.indice_dict = indice_dict
        out_tensor.spatial_shape = out_spatial_shape
        
        # var_tensor = var_tensor.replace_feature(var.features)
        var_tensor = var_tensor.replace_feature(var_features)
        var_tensor.indices = outids
        var_tensor.indice_dict = indice_dict
        var_tensor.spatial_shape = out_spatial_shape
        
        return out_tensor, var_tensor
        
class SparseInverseConv3d(SparseConvolution):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 indice_key,
                 bias=True,
                 algo: Optional[ConvAlgo] = None,
                 fp32_accum: Optional[bool] = None,
                 name=None):
        super(SparseInverseConv3d, self).__init__(3,
                                                  in_channels,
                                                  out_channels,
                                                  kernel_size,
                                                  bias=bias,
                                                  inverse=True,
                                                  indice_key=indice_key,
                                                  algo=algo,
                                                  fp32_accum=fp32_accum,
                                                  name=name)
    def forward(self, input: SparseConvTensor, var: SparseConvTensor):
        assert isinstance(input, SparseConvTensor)
        assert input.features.shape[
            1] == self.in_channels, "channel size mismatch"
        features = input.features
        device = features.device
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        if not self.subm:
            if self.transposed:
                out_spatial_shape = ops.get_deconv_output_size(
                    spatial_shape, self.kernel_size, self.stride, self.padding,
                    self.dilation, self.output_padding)
            else:
                out_spatial_shape = ops.get_conv_output_size(
                    spatial_shape, self.kernel_size, self.stride, self.padding,
                    self.dilation)
        else:
            out_spatial_shape = spatial_shape
        # print(self._sparse_unique_name, spatial_shape, out_spatial_shape)
        # input.update_grid(out_spatial_shape)
        # t = time.time()
        out_tensor = input.shadow_copy()
        if input.benchmark:
            if self.name is None:
                raise ValueError(
                    "you need to assign name to spmodules before benchmark (spconv.utils.bench.assign_name_to_spmod)"
                )
            if self.name not in input.benchmark_record:
                input.benchmark_record[self.name] = {
                    "type": "SparseConvolution",
                    "indice_gen_time": [],
                    "time": [],
                    "num_points": [],
                    "num_out_points": [],
                    "params": {
                        "kernel_size": self.kernel_size,
                        "stride": self.stride,
                        "padding": self.padding,
                        "dilation": self.dilation,
                        "output_padding": self.output_padding,
                        "subm": self.subm,
                        "transposed": self.transposed,
                        "input_channels": self.in_channels,
                        "out_channels": self.out_channels,
                    }
                }
        if self.conv1x1:
            if FILTER_HWIO:
                features = torch.mm(
                    input.features,
                    self.weight.view(self.out_channels, self.in_channels).T)
            else:
                features = torch.mm(
                    input.features,
                    self.weight.view(self.in_channels, self.out_channels))

            if self.bias is not None:
                features += self.bias
            out_tensor = out_tensor.replace_feature(features)
            # padding may change spatial shape of conv 1x1.
            out_tensor.spatial_shape = out_spatial_shape
            return out_tensor
        indice_dict = input.indice_dict.copy()

        algo = self.algo
        if self.indice_key is not None:
            datas = input.find_indice_pair(self.indice_key)
            if datas is not None:
                msg = "due to limitation of pytorch, you must provide same algo to layers share same indice key."
                assert algo == datas.algo, msg
                # algo = datas.algo
        profile_ctx = nullcontext()
        if input._timer is not None and self._sparse_unique_name:
            profile_ctx = input._timer.namespace(self._sparse_unique_name)
        with profile_ctx:
            if algo == ConvAlgo.Native:
                datas = input.find_indice_pair(self.indice_key)
                if datas is not None:
                    assert isinstance(datas, IndiceData)
                if self.inverse:
                    assert datas is not None and self.indice_key is not None
                    assert datas.is_subm is False, "inverse conv can only be used with standard conv and pool ops."

                    outids = datas.indices
                    indice_pairs = datas.indice_pairs
                    indice_pair_num = datas.indice_pair_num
                    out_spatial_shape = datas.spatial_shape
                    assert datas.ksize == self.kernel_size, "inverse conv must have same kernel size as its couple conv"
                else:
                    if self.indice_key is not None and datas is not None:
                        outids = datas.out_indices
                        indice_pairs = datas.indice_pairs
                        indice_pair_num = datas.indice_pair_num
                        assert self.subm, "only support reuse subm indices"
                        self._check_subm_reuse_valid(input, spatial_shape,
                                                     datas)
                    else:
                        if input.benchmark:
                            torch.cuda.synchronize()
                            t = time.time()
                        try:
                            outids, indice_pairs, indice_pair_num = ops.get_indice_pairs(
                                indices, batch_size, spatial_shape, algo,
                                self.kernel_size, self.stride, self.padding,
                                self.dilation, self.output_padding, self.subm,
                                self.transposed)
                        except Exception as e:
                            msg = "[Exception|native_pair]"
                            msg += f"indices={indices.shape},bs={batch_size},ss={spatial_shape},"
                            msg += f"algo={algo},ksize={self.kernel_size},stride={self.stride},"
                            msg += f"padding={self.padding},dilation={self.dilation},subm={self.subm},"
                            msg += f"transpose={self.transposed}"
                            print(msg, file=sys.stderr)
                            spconv_save_debug_data(indices)
                            raise e
                        if input.benchmark:
                            torch.cuda.synchronize()
                            interval = time.time() - t
                            out_tensor.benchmark_record[
                                self.name]["indice_gen_time"].append(interval)

                        indice_data = IndiceData(outids,
                                                 indices,
                                                 indice_pairs,
                                                 indice_pair_num,
                                                 spatial_shape,
                                                 out_spatial_shape,
                                                 is_subm=self.subm,
                                                 algo=algo,
                                                 ksize=self.kernel_size,
                                                 stride=self.stride,
                                                 padding=self.padding,
                                                 dilation=self.dilation)
                        if self.indice_key is not None:
                            msg = f"your indice key {self.indice_key} already exists in this sparse tensor."
                            assert self.indice_key not in indice_dict, msg
                            indice_dict[self.indice_key] = indice_data
                if input.benchmark:
                    torch.cuda.synchronize()
                    t = time.time()
                indice_pairs_calc = indice_pairs
                if indice_pairs.device != features.device:
                    indice_pairs_calc = indice_pairs.to(features.device)
                if self.subm:
                    out_features = Fsp.indice_subm_conv(
                        features, self.weight, indice_pairs_calc,
                        indice_pair_num, outids.shape[0], algo, input._timer)
                else:
                    if self.inverse:
                        out_features = Fsp.indice_inverse_conv(
                            features, self.weight, indice_pairs_calc,
                            indice_pair_num, outids.shape[0], algo)
                    else:
                        out_features = Fsp.indice_conv(features, self.weight,
                                                       indice_pairs_calc,
                                                       indice_pair_num,
                                                       outids.shape[0], algo,
                                                       input._timer)

            else:
                datas = input.find_indice_pair(self.indice_key)
                if datas is not None:
                    assert isinstance(datas, ImplicitGemmIndiceData)
                if self.inverse:
                    assert datas is not None and self.indice_key is not None
                    assert datas.is_subm is False, "inverse conv can only be used with standard conv and pool ops."
                    outids = datas.indices
                    pair_fwd = datas.pair_bwd
                    pair_bwd = datas.pair_fwd
                    pair_mask_fwd_splits = datas.pair_mask_bwd_splits
                    pair_mask_bwd_splits = datas.pair_mask_fwd_splits
                    mask_argsort_fwd_splits = datas.mask_argsort_bwd_splits
                    mask_argsort_bwd_splits = datas.mask_argsort_fwd_splits
                    masks = datas.masks
                    out_spatial_shape = datas.spatial_shape
                    assert datas.ksize == self.kernel_size, "inverse conv must have same kernel size as its couple conv"
                else:
                    if self.indice_key is not None and datas is not None:
                        outids = datas.out_indices
                        pair_fwd = datas.pair_fwd
                        pair_bwd = datas.pair_bwd
                        pair_mask_fwd_splits = datas.pair_mask_fwd_splits
                        pair_mask_bwd_splits = datas.pair_mask_bwd_splits
                        mask_argsort_fwd_splits = datas.mask_argsort_fwd_splits
                        mask_argsort_bwd_splits = datas.mask_argsort_bwd_splits
                        masks = datas.masks
                        assert self.subm, "only support reuse subm indices"
                        self._check_subm_reuse_valid(input, spatial_shape,
                                                     datas)
                    else:

                        with input._timer.namespace("gen_pairs"):
                            # we need to gen bwd indices for regular conv
                            # because it may be inversed.
                            try:
                                res = ops.get_indice_pairs_implicit_gemm(
                                    indices,
                                    batch_size,
                                    spatial_shape,
                                    algo,
                                    ksize=self.kernel_size,
                                    stride=self.stride,
                                    padding=self.padding,
                                    dilation=self.dilation,
                                    out_padding=self.output_padding,
                                    subm=self.subm,
                                    transpose=self.transposed,
                                    is_train=(not self.subm) or self.training,
                                    alloc=input.thrust_allocator,
                                    timer=input._timer)
                            except Exception as e:
                                msg = "[Exception|implicit_gemm_pair]"
                                msg += f"indices={indices.shape},bs={batch_size},ss={spatial_shape},"
                                msg += f"algo={algo},ksize={self.kernel_size},stride={self.stride},"
                                msg += f"padding={self.padding},dilation={self.dilation},subm={self.subm},"
                                msg += f"transpose={self.transposed}"
                                print(msg, file=sys.stderr)
                                spconv_save_debug_data(indices)
                                raise e

                        outids = res[0]
                        num_inds_per_loc = res[1]
                        pair_fwd = res[2]
                        pair_bwd = res[3]
                        pair_mask_fwd_splits = res[4]
                        pair_mask_bwd_splits = res[5]
                        mask_argsort_fwd_splits = res[6]
                        mask_argsort_bwd_splits = res[7]
                        masks = res[8]
                        if self.indice_key is not None:
                            indice_data = ImplicitGemmIndiceData(
                                outids,
                                indices,
                                pair_fwd,
                                pair_bwd,
                                pair_mask_fwd_splits=pair_mask_fwd_splits,
                                pair_mask_bwd_splits=pair_mask_bwd_splits,
                                mask_argsort_fwd_splits=mask_argsort_fwd_splits,
                                mask_argsort_bwd_splits=mask_argsort_bwd_splits,
                                masks=masks,
                                is_subm=self.subm,
                                spatial_shape=spatial_shape,
                                out_spatial_shape=out_spatial_shape,
                                algo=algo,
                                ksize=self.kernel_size,
                                stride=self.stride,
                                padding=self.padding,
                                dilation=self.dilation)
                            msg = f"your indice key {self.indice_key} already exists in this sparse tensor."
                            assert self.indice_key not in indice_dict, msg
                            indice_dict[self.indice_key] = indice_data
                if input.benchmark:
                    torch.cuda.synchronize()
                    t = time.time()
                num_activate_out = outids.shape[0]
                out_features = Fsp.implicit_gemm(
                    features, self.weight, pair_fwd, pair_bwd,
                    pair_mask_fwd_splits, pair_mask_bwd_splits,
                    mask_argsort_fwd_splits, mask_argsort_bwd_splits,
                    num_activate_out, masks, self.training, self.subm,
                    input._timer, self.fp32_accum)
                var_features = Fsp.implicit_gemm(
                    features, self.weight**2, pair_fwd, pair_bwd,
                    pair_mask_fwd_splits, pair_mask_bwd_splits,
                    mask_argsort_fwd_splits, mask_argsort_bwd_splits,
                    num_activate_out, masks, self.training, self.subm,
                    input._timer, self.fp32_accum)
                
        if self.bias is not None:
            out_features += self.bias
        if input.benchmark:
            torch.cuda.synchronize()
            interval = time.time() - t
            out_tensor.benchmark_record[self.name]["time"].append(interval)
            out_tensor.benchmark_record[self.name]["num_points"].append(
                features.shape[0])
            out_tensor.benchmark_record[self.name]["num_out_points"].append(
                out_features.shape[0])

        out_tensor = out_tensor.replace_feature(out_features)
        out_tensor.indices = outids
        out_tensor.indice_dict = indice_dict
        out_tensor.spatial_shape = out_spatial_shape
        
        # var_tensor = var_tensor.replace_feature(var.features)
        var_tensor = var_tensor.replace_feature(var_features)
        var_tensor.indices = outids
        var_tensor.indice_dict = indice_dict
        var_tensor.spatial_shape = out_spatial_shape
        
        return out_tensor, var_tensor
