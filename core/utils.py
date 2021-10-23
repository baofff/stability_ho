from torch.utils.data import DataLoader
import torch
import numpy as np
import random
import os
import pprint
import logging
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def infinite_loader(dataset, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    while True:
        for data in loader:
            yield data


def score_on_dataset(dataset, score_fn, batch_size):
    r"""
    Args:
        dataset: an instance of Dataset
        score_fn: a batch of data -> a batch of scalars
        batch_size: the batch size
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    total_score = 0.
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for z in dataloader:
        z = [item.to(device) for item in z]
        score = score_fn(z)
        total_score += score.sum().detach()
    mean_score = total_score / len(dataset)
    return mean_score


def backup_args(args, path):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "args.txt")
    s = pprint.pformat(args)
    with open(path, 'w') as f:
        f.write(s)


def detach(inputs):
    if isinstance(inputs, torch.Tensor):
        return inputs.detach()
    elif isinstance(inputs, list) or isinstance(inputs, tuple):
        return [item.detach() for item in inputs]
    else:
        raise TypeError


def iter_islast(iterable):
    it = iter(iterable)
    prev = next(it)
    for item in it:
        yield False, prev
        prev = item
    yield True, prev


def plot_record(record, path):
    os.makedirs(path, exist_ok=True)
    for curve_name, data in record.items():
        x = list(map(lambda z: z[0], data))
        y = list(map(lambda z: z[1], data))
        plt.plot(x, y, label="{}".format(curve_name))
        plt.title(curve_name)
        plt.savefig(os.path.join(path, "%s.png" % curve_name))
        plt.close()


def set_logger(fname):
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(fname, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)
