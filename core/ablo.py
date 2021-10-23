import torch
import torch.autograd as autograd
import torch.optim as optim
import math
import logging
import func as func
from .utils import infinite_loader, score_on_dataset


def sgd_step(theta, lamb, loss, z, lr, wd):
    r"""
    :return: (1 - alpha * wd) * theta - alpha * D_theta loss(lamb, theta, z)
    """
    with func.RequiresGradContext(theta, requires_grad=True):
        loss_ = loss(lamb, theta, z).mean()
        g = autograd.grad(loss_, theta, create_graph=True)
    if isinstance(theta, list) or isinstance(theta, tuple):
        return [(1. - lr * wd) * a - lr * b for a, b in zip(theta, g)], loss_.item()
    else:
        return (1. - lr * wd) * theta - lr * g[0], loss_.item()


def train(tr_dataset, val_dataset, te_dataset, mval_dataset, loss_cls, K, T, lr_l, wd_l, lr_h, wd_h, mm_h, tr_batch_size, val_batch_size, te_batch_size, mval_batch_size, device,
          record, print_every=10):
    lamb = loss_cls.init_lamb(requires_grad=True)
    train_dataset_loader = infinite_loader(tr_dataset, batch_size=tr_batch_size)
    val_dataset_loader = infinite_loader(val_dataset, batch_size=val_batch_size)

    opt = optim.SGD([lamb] if isinstance(lamb, torch.Tensor) else lamb, lr=lr_h, weight_decay=wd_h, momentum=mm_h)
    for it_h in range(T):
        theta = loss_cls.init_theta(requires_grad=False)
        for it_l in range(K):
            z_tr = [item.to(device) for item in next(train_dataset_loader)]
            theta, loss_tr_sgd = sgd_step(theta, lamb, loss_cls.loss_in, z_tr, lr_l, wd_l)
        z_val = [item.to(device) for item in next(val_dataset_loader)]
        loss_val = loss_cls.loss_out(lamb, theta, z_val).mean()

        opt.zero_grad()
        loss_val.backward()
        opt.step()

        if it_h % print_every == 0:
            logging.info("it_h: {}\tloss_val: {:.6f}".format(it_h, loss_val.item()))
        if "loss_tr_sgd" in record:
            record["loss_tr_sgd"].append((it_h, loss_tr_sgd))
        if "loss_tr" in record:
            loss_tr = score_on_dataset(tr_dataset, lambda z: loss_cls.loss_out(lamb, theta, z), tr_batch_size)
            record["loss_tr"].append((it_h, loss_tr.item()))
        if "zero_one_loss_tr" in record:
            zero_one_loss_tr = score_on_dataset(tr_dataset, lambda z: loss_cls.zero_one_loss(lamb, theta, z), tr_batch_size)
            record["zero_one_loss_tr"].append((it_h, zero_one_loss_tr.item()))
        if "loss_val" in record:
            record["loss_val"].append((it_h, loss_val.item()))
        if "zero_one_loss_val" in record:
            zero_one_loss_val = score_on_dataset(val_dataset, lambda z: loss_cls.zero_one_loss(lamb, theta, z), val_batch_size)
            record["zero_one_loss_val"].append((it_h, zero_one_loss_val.item()))
        if "loss_te" in record:
            loss_te = score_on_dataset(te_dataset, lambda z: loss_cls.loss_out(lamb, theta, z), te_batch_size)
            record["loss_te"].append((it_h, loss_te.item()))
        if "zero_one_loss_te" in record:
            zero_one_loss_te = score_on_dataset(te_dataset, lambda z: loss_cls.zero_one_loss(lamb, theta, z), te_batch_size)
            record["zero_one_loss_te"].append((it_h, zero_one_loss_te.item()))
        if "loss_mval" in record:
            loss_mval = score_on_dataset(mval_dataset, lambda z: loss_cls.loss_out(lamb, theta, z), mval_batch_size)
            record["loss_mval"].append((it_h, loss_mval.item()))
        if "zero_one_loss_mval" in record:
            zero_one_loss_mval = score_on_dataset(mval_dataset, lambda z: loss_cls.zero_one_loss(lamb, theta, z), mval_batch_size)
            record["zero_one_loss_mval"].append((it_h, zero_one_loss_mval.item()))
        if "gap" in record:
            record["gap"].append((it_h, loss_te.item() - loss_val.item()))

        if math.isnan(loss_val.item()):
            logging.info('nan at {}'.format(it_h))
            exit(0)
