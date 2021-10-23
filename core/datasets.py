import torch
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
import random
from collections import defaultdict
from torch.utils.data import Dataset


class PMLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, label):
        onehot = F.one_hot(torch.tensor(label), num_classes=self.num_classes).float()
        return onehot * 2. - 1.


class Flatten(object):
    def __call__(self, tensor):
        return tensor.view(-1)


class QuickDataset(Dataset):
    def __init__(self, array):
        self.array = array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, item):
        return self.array[item]


class CorruptedMnist(object):
    def __init__(self, width, task, flatten):
        self.task = task
        _transform = [transforms.Resize(width), transforms.ToTensor()]
        if flatten:
            _transform.append(Flatten())
        target_transform = None
        if task == 'regression':
            target_transform = PMLabel(10)
        self.dst = datasets.MNIST('workspace/datasets/mnist', train=True, transform=transforms.Compose(_transform), target_transform=target_transform, download=True)
        self.idxes = list(range(len(self.dst)))
        random.shuffle(self.idxes)
        self.current_p = 0

    def _get_corrupted_data(self, m):
        assert m % 2 == 0
        data_lst = []
        for i, k in enumerate(range(self.current_p, self.current_p + m)):
            x, y = self.dst[self.idxes[k]]
            if k >= self.current_p + m // 2:
                y = random.randint(0, 9)
            data_lst.append((i, x, y))
        self.current_p += m
        return QuickDataset(data_lst)

    def _get_clean_data(self, m):
        data_lst = []
        for k in range(self.current_p, self.current_p + m):
            data_lst.append(self.dst[self.idxes[k]])
        self.current_p += m
        return QuickDataset(data_lst)

    def get_data(self, m_tr, m_val, m_te, m_mval, dim):
        if self.current_p + m_tr + m_val + m_te > len(self.dst):
            self.current_p = 0
            random.shuffle(self.idxes)
        return self._get_corrupted_data(m_tr), self._get_clean_data(m_val), self._get_clean_data(m_te), self._get_clean_data(m_mval)


class Omniglot(object):
    r"""
        964 classes in background, 659 classes in evaluation
        20 samples for each class
    """
    def __init__(self, width, flatten, num_classes):
        self.num_classes = num_classes
        _transform = [transforms.Resize(width), transforms.ToTensor()]
        if flatten:
            _transform.append(Flatten())
        dst = datasets.Omniglot('workspace/datasets/omniglot', transform=transforms.Compose(_transform), download=True)

        background_classes = list(range(964))
        random.shuffle(background_classes)
        chosen_classes = background_classes[:num_classes]
        class_idx_map = {cls: idx for idx, cls in enumerate(chosen_classes)}

        all_class_dct = defaultdict(list)
        for z in dst:
            all_class_dct[z[1]].append(z)

        self.idx_dct = {}
        for cls in chosen_classes:
            idx = class_idx_map[cls]
            self.idx_dct[idx] = [(1. - x, idx) for x, y in all_class_dct[cls]]

        self.pt = 0

    def shuffle_dataset(self):
        for idx, lst in self.idx_dct.items():
            random.shuffle(lst)

    def _get_data(self, m):
        assert m % self.num_classes == 0
        m_per_cls = m // self.num_classes
        if m_per_cls >= 20:
            raise ValueError
        if self.pt + m_per_cls > 20:
            self.shuffle_dataset()
            self.pt = 0
        res = []
        for idx in range(self.num_classes):
            res.extend(self.idx_dct[idx][self.pt: self.pt + m_per_cls])
        self.pt += m_per_cls
        return QuickDataset(res)

    def get_data(self, m_tr, m_val, m_te, m_mval, dim):
        return self._get_data(m_tr), self._get_data(m_val), self._get_data(m_te), self._get_data(m_mval)

