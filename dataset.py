import torch
import math
import random
import os
import pathlib
import pandas as pd
from torch.utils import data
import numpy as np
from tqdm import trange
import torch.distributed as dist
from typing import TypeVar, Iterator
from glob import glob
T_co = TypeVar('T_co', covariant=True)

def make_samples(directory:pathlib.Path, 
                extensions:list=['.npy']):
    '''
    Path composition:
    root / dataset / patient ID / Study Instance UID / Series Instance UID / {index}_{scale}.npy
    '''
    instances = []
    for ext in extensions:
        instances += list(directory.glob(f'**/*{ext}'))
    return instances
    
    # directory = os.path.expanduser(directory)
    # pos_cnt, neg_cnt = 0, 0
    # if (extensions is None and is_valid_file is None) \
    #     or (extensions is not None and is_valid_file is not None):
    #     raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    # if extensions is not None:
    #     def is_valid_file(x):
    #         # return has_file_allowed_extension(x, extensions)
    #         return x.lower().endswith(extensions)
    # # for target_class in sorted(class_to_idx.keys()):
    # for target_class in sorted(class_to_idx):
    #     target_dirs = glob(directory + '/%s'%target_class)
    #     # target_dir = os.path.join(directory, target_class)
    #     if debug:
    #         print(f"target class: {target_class}")
    #         print(f"directory: {directory}")
    #         print(f"target directories: {target_dirs}")
    #     for target_dir in target_dirs:
    #         if not os.path.isdir(target_dir):
    #             continue
    #         if class_index == 0:
    #             for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
    #                 for fname in sorted(fnames):
    #                     path = os.path.join(root, fname)
    #                     path_mask = path.replace(img_dir, mask_dir)
    #                     if is_valid_file(path):
    #                         item = path, path_mask, class_index
    #                         instances.append(item)
    #                         neg_cnt += 1
    #         else:
    #             for root, _, fnames in sorted(os.walk(target_dir.replace(img_dir,mask_dir), followlinks=True)):
    #                 for fname in sorted(fnames):
    #                     path_mask = os.path.join(root, fname)
    #                     path = path_mask.replace(mask_dir, img_dir)
    #                     if is_valid_file(path_mask):
    #                         item = path, path_mask, class_index
    #                         instances.append(item)
    #                         pos_cnt += 1

    # if debug:
    #     print(f"positive: {pos_cnt}, negative: {neg_cnt}")
    # return instances

class SegDataset(data.Dataset):
    def __init__(self, samples, img_dir, mask_dir, transform=None, only_path=False, on_memory=False):
        self.samples = samples
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.only_path = only_path
        self.on_memory = on_memory
        if on_memory:
            self.images = []
            for idx in range(len(samples)):
                self.images.append(np.load(self.samples[idx][0]))
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.only_path:
            crop = self.samples[idx][0]
        else:
            if self.on_memory:
                crop = self.images[idx]
            else:
                crop = np.load(self.samples[idx])
                mask = np.load(str(self.samples[idx]).replace(self.img_dir, self.mask_dir))
            if self.transform is not None:
                crop, mask = self.transform(crop, mask)
            crop = torch.from_numpy(crop.copy()).float()
            mask = torch.from_numpy(mask.copy()).float()

        return crop, mask

class DatasetDataframe(data.Dataset):
    def __init__(self, csv_path, data_path, transform=None, only_path=False, scales=[0.8, 0.9, 1.0, 1.1, 1.2]):
        self.transform = transform
        self.only_path = only_path
        self.df = pd.read_csv(csv_path)
        # self.df = self.df.iloc[:300]
        self.data_path = data_path
        self.scales = scales
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = int(self.df['class'][idx])

        if self.only_path:
            scale = random.sample(self.scales, 1)[0]
            crop = os.path.join(self.data_path, str(self.df['class'][idx]), self.df['seriesuid'][idx],
                                '_'.join([str(int(self.df['coordZ'][idx])),
                                          str(int(self.df['coordY'][idx])),
                                          str(int(self.df['coordX'][idx])), str(scale)])+'.npy')
        else:
            scale = random.sample(self.scales, 1)[0]
            path = os.path.join(self.data_path, str(self.df['class'][idx]), self.df['seriesuid'][idx],
                                '_'.join([str(int(self.df['coordZ'][idx])),
                                          str(int(self.df['coordY'][idx])),
                                          str(int(self.df['coordX'][idx])), str(scale)])+'.npy')
            crop = np.load(path)/255.

            if self.transform is not None:
                crop, label = self.transform(crop, label)
                if type(crop) == list:
                    crop = [torch.from_numpy(x).float() for x in crop]
                else:
                    crop = torch.from_numpy(crop).float()

        return crop, label

class BalancedDistributedSampler(data.sampler.Sampler[T_co]):
    def __init__(self, 
                dataset, 
                num_replicas = None,
                rank = None, 
                shuffle = True,
                seed = 0, 
                drop_last = False
                ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.class_dataset = dict()
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        print(f"Distributed rank: {rank}")

        # oversampling small class data
        tbar = trange(len(dataset))
        """
        self.dataset =
            {0: [indexes where label = 0],
            1: [indexes where label = 1]}
        """
        for idx in tbar:
            label = dataset[idx][1]
            tbar.set_postfix({'class':label})
            if label not in self.class_dataset:
                self.class_dataset[label] = list()
            self.class_dataset[label].append(idx)
            # self.balanced_max = len(self.dataset[label]) if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        
        ## len_list = [number of label 0, number of label 1]
        len_list = [len(self.class_dataset[cls]) for cls in self.class_dataset.keys()]
        len_dict = {cls: len(self.class_dataset[cls]) for cls in self.class_dataset.keys()}
        print(f"number of data in each class: {len_dict}")
        ## balanced_max: maximum number of labels
        self.balanced_max = max(len_list)
        for cls in self.class_dataset:
            while len(self.class_dataset[cls]) < self.balanced_max:
                self.class_dataset[cls].append(random.choice(self.class_dataset[cls]))

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        # if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
        if self.drop_last and len(self.class_dataset.keys()) * self.balanced_max % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.class_dataset.keys()) * self.balanced_max - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.class_dataset.keys()) * self.balanced_max / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        
    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            cur_class_dataset = {}
            for cls in self.class_dataset:
                cls_order = torch.randperm(len(self.class_dataset[cls]), generator=g).tolist()
                cur_class_dataset[cls] = [self.class_dataset[cls][idx] for idx in cls_order]
            # indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            cur_class_dataset = self.class_dataset
            # indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        indices = list()
        for i in trange(0,self.balanced_max,self.num_replicas, desc="Balance Batch Sampling..."):
            for cls in cur_class_dataset:
                indices += cur_class_dataset[cls][i:i+self.num_replicas]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch