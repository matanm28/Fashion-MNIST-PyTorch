import random
from typing import List, Tuple

import torch
from torchvision.transforms import Compose
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np


def get_data_from_file(file_path: str, num_of_rows: int = None):
    if num_of_rows is not None:
        data = np.loadtxt(file_path, max_rows=num_of_rows)
    else:
        data = get_data_set_from_file(file_path)
    return data


def get_data_set_from_file(training_set_examples_path: str):
    examples_list = []
    with open(training_set_examples_path, 'r') as training_examples_file:
        lines = training_examples_file.readlines()
        for line in lines:
            data = line.split()
            examples_list.append(data)
    return np.array(examples_list, dtype=np.float)


def get_data_answers_from_file(file_path: str, num_of_rows: int = None):
    if num_of_rows is not None:
        answers = np.loadtxt(file_path, max_rows=num_of_rows, dtype=np.int16)
    else:
        answers = np.loadtxt(file_path, dtype=np.int16)
    return answers


class FashionDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray) -> None:
        super(Dataset, self).__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transforms = None

    @classmethod
    def from_files(cls, train_x_path: str, train_y_path: str = None, num_of_rows=None,
                   validation_size: float = None) -> Tuple['FashionDataset', 'FashionDataset']:
        x_tmp = get_data_from_file(train_x_path, num_of_rows)
        if train_y_path is not None:
            y_tmp = get_data_answers_from_file(train_y_path, num_of_rows)
        else:
            y_tmp = np.zeros((x_tmp.shape[0]))
        if validation_size is None:
            return FashionDataset(x_tmp, y_tmp), None
        else:
            training, validation = cls.split_numpy_arrays(x_tmp, y_tmp, int(y_tmp.size * validation_size))
            return FashionDataset(training[0], training[1]), FashionDataset(validation[0], validation[1])

    @classmethod
    def split_numpy_arrays(cls, data: np.ndarray, labels: np.ndarray, validation_size: int):
        mask = torch.ones(data.shape[0], dtype=bool)
        false_values = 0
        while false_values < validation_size:
            idx = random.randint(0, data.shape[0] - 1)
            if mask[idx]:
                mask[idx] = False
                false_values += 1
        training_data = data[mask]
        training_labels = labels[mask]
        validation_data = data[~mask]
        validation_labels = labels[~mask]
        return (training_data, training_labels), (validation_data, validation_labels)

    def set_transforms(self, transforms: Compose):
        self.transforms = transforms

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx: List[int]) -> Tuple[Tensor, Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]
        labels = self.labels[idx]
        if self.transforms is not None:
            data, labels = self.transforms((data, labels))
        return data, labels

    @property
    def data_x(self) -> Tensor:
        return self.data

    def shape(self, i: int):
        return self.data_x.shape[i]


class StdNormalizer:
    def __init__(self, data_set: Tensor) -> None:
        self.mean = data_set.mean(axis=0)
        self.std_dev = data_set.std(axis=0)

    def __call__(self, data: Tensor) -> Tensor:
        inputs, labels = data
        x = ((inputs - self.mean) / self.std_dev)
        x[torch.isnan(x)] = 0
        return x, labels


class MinMaxNormalizer:
    def __init__(self, data_set: Tensor, normalized_max=1, normalized_min: int = 0) -> None:
        self.min = data_set.min(axis=0)
        self.max = data_set.max(axis=0)
        self.normalized_min = normalized_min
        self.normalized_max = normalized_max

    def __call__(self, data: Tensor):
        inputs, labels = data
        factor = (self.normalized_max - self.normalized_min)
        x = ((inputs - self.min) / (self.max - self.min)) * factor + self.normalized_min
        x[torch.isnan(x)] = 0
        return x, labels
