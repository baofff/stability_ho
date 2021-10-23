from .utils import infinite_loader, score_on_dataset, detach
from .ablo import sgd_step
import math
import logging


def train(tr_dataset, val_dataset, te_dataset, loss_cls, lamb_gen, K, lr_l, wd_l, tr_batch_size, val_batch_size, te_batch_size,
          device, record, print_every=10):
    train_dataset_loader = infinite_loader(tr_dataset, batch_size=tr_batch_size)

    best_loss_val = float('inf')
    best_lamb = None
    best_theta = None
    for it_h, lamb in enumerate(lamb_gen):
        theta = loss_cls.init_theta(requires_grad=False)
        for it_l in range(K):
            z_tr = [item.to(device) for item in next(train_dataset_loader)]
            theta, loss_tr_sgd = sgd_step(theta, lamb, loss_cls.loss_in, z_tr, lr_l, wd_l)
            theta = detach(theta)
        loss_val = score_on_dataset(val_dataset, lambda z: loss_cls.loss_out(lamb, theta, z), val_batch_size).item()
        if loss_val < best_loss_val or it_h == 0:
            best_loss_val = loss_val
            best_lamb = lamb
            best_theta = theta

        if it_h % print_every == 0:
            logging.info("it_h: {}\tbest_loss_val: {:.6f}".format(it_h, best_loss_val))
        if "loss_tr_sgd" in record:
            record["loss_tr_sgd"].append((it_h, loss_tr_sgd))
        if "loss_tr" in record:
            loss_tr = score_on_dataset(tr_dataset, lambda z: loss_cls.loss_out(best_lamb, best_theta, z), tr_batch_size).item()
            record["loss_tr"].append((it_h, loss_tr))
        if "zero_one_loss_tr" in record:
            zero_one_loss_tr = score_on_dataset(tr_dataset, lambda z: loss_cls.zero_one_loss(best_lamb, best_theta, z), tr_batch_size).item()
            record["zero_one_loss_tr"].append((it_h, zero_one_loss_tr))
        if "loss_val" in record:
            record["loss_val"].append((it_h, best_loss_val))
        if "zero_one_loss_val" in record:
            zero_one_loss_val = score_on_dataset(val_dataset, lambda z: loss_cls.zero_one_loss(best_lamb, best_theta, z), val_batch_size).item()
            record["zero_one_loss_val"].append((it_h, zero_one_loss_val))
        if "loss_te" in record:
            loss_te = score_on_dataset(te_dataset, lambda z: loss_cls.loss_out(best_lamb, best_theta, z), te_batch_size).item()
            record["loss_te"].append((it_h, loss_te))
        if "zero_one_loss_te" in record:
            zero_one_loss_te = score_on_dataset(te_dataset, lambda z: loss_cls.zero_one_loss(best_lamb, best_theta, z), te_batch_size).item()
            record["zero_one_loss_te"].append((it_h, zero_one_loss_te))
        if "gap" in record:
            record["gap"].append((it_h, loss_te - best_loss_val))

        if math.isnan(loss_val):
            logging.info('nan at {}'.format(it_h))
            exit(0)
