import sys

import torch
import tensorflow as tf
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import h5py
import numpy as np
import collections
import numbers
import math
import pandas as pd


class KDD99Loader(object):
    def __init__(self, data_path, mode="train"):
        self.mode = mode
        data = np.load(data_path)

        labels = data["kdd"][:, -1]
        features = data["kdd"][:, :-1]
        N, D = features.shape

        normal_data = features[labels == 1]
        normal_labels = labels[labels == 1]

        attack_data = features[labels == 0]
        attack_labels = labels[labels == 0]

        N_attack = attack_data.shape[0]

        randIdx = np.arange(N_attack)
        np.random.shuffle(randIdx)
        N_train = N_attack // 2

        self.train = attack_data[randIdx[:N_train]]
        self.train_labels = attack_labels[randIdx[:N_train]]

        self.test = attack_data[randIdx[N_train:]]
        self.test_labels = attack_labels[randIdx[N_train:]]

        self.test = np.concatenate((self.test, normal_data), axis=0)
        self.test_labels = np.concatenate((self.test_labels, normal_labels),
                                          axis=0)


def get_dataset(data_path, batch_size, mode='train'):
    dataset = KDD99Loader(data_path, mode)

    if mode == 'train':
        train_data = tf.data.Dataset.from_tensor_slices(dataset.train)
        train_data = train_data.shuffle(dataset.train.shape[0])
        return train_data.batch(batch_size)

    else:
        test_data = tf.data.Dataset.from_tensor_slices(dataset.test)
        return test_data.batch(batch_size)
