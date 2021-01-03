import math
import sys

import torch
from numpy import ndarray
from torchvision import transforms, datasets
import numpy as np

from fashion_dataset import StdNormalizer, FashionDataset
from model_a import ModelA
from model_wrapper import ModelWrapper
from torch.utils.data import random_split

MODELS = {
    'A': ModelA
}


def normalize_data_std(data_set: ndarray):
    mean = data_set.mean(axis=0)
    std_dev = data_set.std(axis=0)
    return np.nan_to_num((data_set - mean) / std_dev)


def normalize_data_min_max(data_set: ndarray, new_min: int = 0, new_max: int = 1):
    data_min = data_set.min(axis=0)
    data_max = data_set.max(axis=0)
    return ((data_set - data_min) / (data_max - data_min)) * (new_max - new_min) + new_min


def run_model(train_x_path: str, train_y_path: str, test_x_path: str, model_name: str, epochs: int, lr: float,
              batch_size: int, num_of_rows: int = None):
    training_set, validation_set = FashionDataset.from_files(train_x_path, train_y_path, num_of_rows, 0.2)
    train_transform_func = transforms.Compose([StdNormalizer(training_set.data)])
    training_set.set_transforms(train_transform_func)
    validation_set_transforms = transforms.Compose([StdNormalizer(validation_set.data)])
    validation_set.set_transforms(validation_set_transforms)
    online_validation = datasets.FashionMNIST('.\data', train=False, download=True)
    flat_data_tensor = online_validation.data.reshape((online_validation.data.shape[0], 784)).type(torch.float32)
    online_validation_set_transforms = transforms.Compose([StdNormalizer(flat_data_tensor)])
    online_validation_set = FashionDataset(flat_data_tensor.numpy(), online_validation.targets.numpy())
    online_validation_set.set_transforms(online_validation_set_transforms)

    test_set, _ = FashionDataset.from_files(test_x_path)
    test_transform_func = transforms.Compose([StdNormalizer(test_set.data)])
    test_set.set_transforms(test_transform_func)

    model = MODELS[model_name](training_set.data.shape[1], lr)
    model_wrapper = ModelWrapper(model, epochs, batch_size)

    loss_arr, accuracy_arr = model_wrapper.train(training_set)
    print(f'Total Loss: {loss_arr[-1]}')
    loss, accuracy = model_wrapper.test(validation_set)
    print(f'Accuracy: {accuracy}')
    predict = model_wrapper.predict(test_set)
    with open('test_y', 'w') as file:
        for pred in predict.numpy():
            file.write(f'{pred}\n')
    return


def main(train_x_path: str, train_y_path: str, test_x_path: str, num_of_rows: int = None):
    models_list = []
    run_model(train_x_path, train_y_path, test_x_path, 'A', 15, 0.1, 500, num_of_rows)


if __name__ == '__main__':
    LABELS = {
        0: 'T-shirt/top',
        1: 'Trousers',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }
    # diff_rows_list = check_files_equal()
    # print(diff_rows_list)
    NUM_OF_ROWS = None
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3], NUM_OF_ROWS)
