# coding=utf-8
import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import random
import csv
import torch
import torch.autograd as autograd  # torch中自动计算梯度模块


class EHRDataset(BaseDataset):
    """"set EHR preprocessing steps"""

    def __init__(self, opt):
        """Initialize this dataset class.

          Parameters:
              opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
          """
        BaseDataset.__init__(self, opt)
        if opt.phase == 'val':
            dataset_now_orignal = list(csv.reader(
                open(opt.dataroot+'/val_now.csv', 'r', encoding='utf-8-sig')))
            dataset_last_orignal = list(csv.reader(
                open(opt.dataroot+'/val_last.csv', 'r', encoding='utf-8-sig')))
        elif opt.phase == 'test':
            dataset_now_orignal = list(csv.reader(
                open(opt.dataroot+'/test_now.csv', 'r', encoding='utf-8-sig')))
            dataset_last_orignal = list(csv.reader(
                open(opt.dataroot+'/test_last.csv', 'r', encoding='utf-8-sig')))
        else:
            dataset_now_orignal = list(csv.reader(
                open(opt.dataroot+'/train_now.csv', 'r', encoding='utf-8-sig')))
            dataset_last_orignal = list(csv.reader(
                open(opt.dataroot+'/train_last.csv', 'r', encoding='utf-8-sig')))

        # get the size of dataset A
        self.A_size = len(dataset_now_orignal)
        # get the size of dataset B
        self.B_size = len(dataset_last_orignal)
        self.dataset_now = prepare_dataset(dataset_now_orignal)
        # print(self.dataset_now)
        self.dataset_last = prepare_dataset(dataset_last_orignal)
        # self.transform = get_transform()

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
        """
        # apply image transformation
        # now = self.transform(index)
        # last = self.transform(index)
        now = autograd.Variable(
            torch.Tensor(self.dataset_now[index]))
        last = autograd.Variable(
            torch.Tensor(self.dataset_last[index]))

        return {'now': now, 'last': last}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)


def prepare_dataset(datafile):
    dataset = []
    for data in datafile:
        numbers = data[0].split(',')
        numbers = list(map(float, numbers))
        count = len(numbers)//10
        dataset.append([numbers[i*10:i*10+10]
                        for i in range(count)])
    return dataset
