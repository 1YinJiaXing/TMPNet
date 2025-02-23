import os
import numpy as np
import torch
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class CustomDataset(Dataset):
    def __init__(self, root='/home/ubunone/YJX/fenge/point-transformer-master/data/Lianjian', npoints=2048, split='train', normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.normal_channel = normal_channel
        self.datapath = []

        self.classes = {0: 'Class0', 1: 'Class1', 2: 'Class2', 3: 'Class3', 4: 'Class4', 5: 'Class5', 6: 'Class6', 7: 'Class7' }
        self.num_classes = len(self.classes)

        # Assuming the split files are named as train.txt, val.txt, and test.txt
        split_file = os.path.join(self.root, f'{split}.txt')
        if not os.path.exists(split_file):
            raise ValueError(f"Split file {split_file} not found")

        with open(split_file, 'r') as f:
            filenames = f.read().splitlines()

        for fn in filenames:
            self.datapath.append(os.path.join(self.root, fn))

        self.cache = {}  # from index to (point_set, seg) tuple
        self.cache_size = 20000000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            data = np.loadtxt(fn).astype(np.float32)
            point_set = data[:, 0:3]  # xyz coordinates
            seg = data[:, 3].astype(np.int32)  # Class labels

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, seg)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, seg  # Ensure the return format is (coord, label)

    def __len__(self):
        return len(self.datapath)
