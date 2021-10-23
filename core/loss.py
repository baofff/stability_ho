import func
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import create_mlp_params, mlp_forward


class Loss(object):
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def loss_in(self, lamb, theta, z):
        raise NotImplementedError

    def loss_out(self, lamb, theta, z):
        raise NotImplementedError


class MLPFeatureLearning(Loss):
    def __init__(self, lamb_mlp_shape, theta_mlp_shape, task, af=nn.ELU()):
        super().__init__()
        self.lamb_mlp_shape = lamb_mlp_shape
        self.theta_mlp_shape = theta_mlp_shape
        self.af = af
        self.task = task
        self.init_theta_ = create_mlp_params(self.theta_mlp_shape, 'cpu', requires_grad=False)

    def loss(self, lamb, theta, z):
        x, y = z
        features = self.af(mlp_forward(x, lamb, self.af))
        outputs = mlp_forward(features, theta, self.af)
        if self.task == 'regression':
            return func.sos(y - outputs, 1)
        elif self.task == 'classification':
            return F.cross_entropy(outputs, y, reduction='none')
        else:
            raise ValueError

    def zero_one_loss(self, lamb, theta, z):
        x, y = z
        features = self.af(mlp_forward(x, lamb, self.af))
        outputs = mlp_forward(features, theta, self.af)
        _, predicted = outputs.max(dim=1)
        if y.dim() > 1:
            _, y = y.max(dim=1)
        return 1. - (predicted == y).float()

    def loss_out(self, lamb, theta, z):
        return self.loss(lamb, theta, z)

    def loss_in(self, lamb, theta, z):
        return self.loss(lamb, theta, z)

    def init_lamb(self, requires_grad):
        return create_mlp_params(self.lamb_mlp_shape, self.device, requires_grad=requires_grad)

    def init_theta(self, requires_grad):
        return [item.clone().detach().to(self.device).requires_grad_(requires_grad) for item in self.init_theta_]

    def lamb_gen(self, T, requires_grad):
        for _ in range(T):
            yield self.init_lamb(requires_grad)


class ReweightingMLP(Loss):
    def __init__(self, m_tr, theta_mlp_shape, af=nn.ELU()):
        super().__init__()
        self.m_tr = m_tr
        self.theta_mlp_shape = theta_mlp_shape
        self.af = af
        self.init_theta_ = create_mlp_params(self.theta_mlp_shape, 'cpu', requires_grad=False)

    def zero_one_loss(self, lamb, theta, z):
        x, y = z
        logits = mlp_forward(x, theta, self.af)
        _, predicted = logits.max(dim=1)
        return 1. - (predicted == y).float()

    def loss_out(self, lamb, theta, z):
        x, y = z
        logits = mlp_forward(x, theta, self.af)
        return F.cross_entropy(logits, y, reduction='none')

    def loss_in(self, lamb, theta, z):
        i, x, y = z
        used_lamb = lamb[i]
        logits = mlp_forward(x, theta, self.af)
        return used_lamb.sigmoid() * F.cross_entropy(logits, y, reduction='none')

    def init_lamb(self, requires_grad):
        return torch.zeros(self.m_tr, device=self.device, requires_grad=requires_grad)

    def init_theta(self, requires_grad):
        return [item.clone().detach().to(self.device).requires_grad_(requires_grad) for item in self.init_theta_]

    def lamb_gen(self, T, requires_grad):
        for _ in range(T):
            yield ((torch.rand(self.m_tr, device=self.device) - 0.5) * 10).requires_grad_(requires_grad)
