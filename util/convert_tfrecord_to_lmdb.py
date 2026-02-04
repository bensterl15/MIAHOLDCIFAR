# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License for
# CLD-SGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import argparse
import os

import lmdb
import numpy as np
import torch
from tfrecord.torch.dataset import TFRecordDataset


def main(dataset, split, tfr_path, lmdb_root):
    assert split in {"train", "validation"}

    # create target directory
    os.makedirs(lmdb_root, exist_ok=True)

    if dataset == "celeba" and split in {"train", "validation"}:
        num_shards = {"train": 120, "validation": 40}[split]
        lmdb_path = os.path.join(lmdb_root, f"{split}.lmdb")
        tfrecord_path_template = os.path.join(
            tfr_path, "%s/%s-r08-s-%04d-of-%04d.tfrecords"
        )
    elif dataset == "imagenet-oord_32":
        num_shards = {"train": 2000, "validation": 80}[split]
        lmdb_path = os.path.join(lmdb_root, f"{split}.lmdb")
        tfrecord_path_template = os.path.join(
            tfr_path, "%s/%s-r05-s-%04d-of-%04d.tfrecords"
        )
    elif dataset == "imagenet-oord_64":
        num_shards = {"train": 2000, "validation": 80}[split]
        lmdb_path = os.path.join(lmdb_root, f"{split}.lmdb")
        tfrecord_path_template = os.path.join(
            tfr_path, "%s/%s-r06-s-%04d-of-%04d.tfrecords"
        )
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset}")

    # open LMDB once
    env = lmdb.open(lmdb_path, map_size=int(50e9))
    count = 0
    commit_interval = 1000  # commit every N samples

    txn = env.begin(write=True)
    try:
        for tf_ind in range(num_shards):
            tfrecord_path = tfrecord_path_template % (split, split, tf_ind, num_shards)
            index_path = None
            description = {"shape": "int", "data": "byte", "label": "int"}
            tf_ds = TFRecordDataset(tfrecord_path, index_path, description)
            loader = torch.utils.data.DataLoader(tf_ds, batch_size=1)

            for data in loader:
                raw_bytes = data["data"][0]  # bytes
                raw_im = np.frombuffer(raw_bytes, dtype=np.uint8)

                # CelebA Glow TFRecords are 256x256x3
                im = raw_im.reshape(256, 256, 3).astype(np.uint8)

                value = im.tobytes()  # raw HWC uint8
                key = str(count).encode("ascii")

                txn.put(key, value)
                count += 1

                if count % commit_interval == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                    print(f"{split}: processed {count}")

        # final commit
        txn.commit()
    finally:
        env.close()

    print(f"added {count} items to the LMDB dataset at {lmdb_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LMDB creator using TFRecords from GLOW.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet-oord_32",
        help="dataset name",
        choices=["imagenet-oord_32", "imagenet-oord_64", "celeba"],
    )
    parser.add_argument(
        "--tfr_path",
        type=str,
        default="/data1/datasets/imagenet-oord/mnt/host/imagenet-oord-tfr",
        help="location of TFRecords",
    )
    parser.add_argument(
        "--lmdb_path",
        type=str,
        default="/data1/datasets/imagenet-oord/imagenet-oord-lmdb_32",
        help="target location for storing lmdb files",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="training or validation split",
        choices=["train", "validation"],
    )
    args = parser.parse_args()
    main(args.dataset, args.split, args.tfr_path, args.lmdb_path)
