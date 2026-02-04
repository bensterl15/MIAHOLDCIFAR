# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License for
# CLD-SGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch.utils.data as data
import numpy as np
import lmdb
import os
import io
from PIL import Image


def num_samples(dataset, train):
    if dataset == 'celeba':
        return 27000 if train else 3000
    elif dataset == 'celeba64':
        return 162770 if train else 19867
    elif dataset == 'imagenet-oord':
        return 1281147 if train else 50000
    elif dataset == 'ffhq':
        return 63000 if train else 7000
    else:
        raise NotImplementedError('dataset %s is unknown' % dataset)


# lmdb_datasets.py

class LMDBDataset(data.Dataset):
    def __init__(self, root, name='', train=True, transform=None, is_encoded=False):
        self.train = train
        self.name = name
        self.transform = transform
        if self.train:
            lmdb_path = os.path.join(root, 'train.lmdb')
        else:
            lmdb_path = os.path.join(root, 'validation.lmdb')

        self.data_lmdb = lmdb.open(
            lmdb_path, readonly=True, max_readers=1,
            lock=False, readahead=False, meminit=False
        )
        self.is_encoded = is_encoded

        # Build list of keys once, in LMDB's sorted order
        with self.data_lmdb.begin(write=False) as txn:
            self.keys = [k for k, _ in txn.cursor()]
        print(f"{self.name} {'train' if self.train else 'val'}: {len(self.keys)} LMDB keys")

    def __getitem__(self, index):
        target = [0]
        with self.data_lmdb.begin(write=False, buffers=True) as txn:
            key = self.keys[index]
            data = txn.get(key)
            if data is None:
                raise IndexError(f"Missing key {key!r} at index {index}")

            if self.is_encoded:
                img = Image.open(io.BytesIO(data)).convert('RGB')
            else:
                img = np.frombuffer(data, dtype=np.uint8)
                size = int(np.sqrt(len(img) / 3))
                img = img.reshape(size, size, 3)
                img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.keys)
