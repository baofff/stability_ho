import torch
import torch.nn.init as init
import torch.nn.functional as F
import math
from .utils import iter_islast


def reset_parameters_linear(weight, bias):
    init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(bias, -bound, bound)


def create_mlp_params(mlp_shape, device, requires_grad):
    params = []
    for in_features, out_features in zip(mlp_shape, mlp_shape[1:]):
        weight = torch.zeros(out_features, in_features, requires_grad=requires_grad, device=device)
        bias = torch.zeros(out_features, requires_grad=requires_grad, device=device)
        reset_parameters_linear(weight, bias)
        params.extend([weight, bias])
    return params


def mlp_forward(inputs, params, af):
    for islast, (weight, bias) in iter_islast(zip(params[::2], params[1::2])):
        inputs = F.linear(inputs, weight, bias)
        if not islast:
            inputs = af(inputs)
    return inputs
