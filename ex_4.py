import concurrent
import math
import sys
import matplotlib.pyplot as plt
from concurrent.futures.thread import ThreadPoolExecutor

import torch
from numpy import ndarray
from torchvision import transforms, datasets
import numpy as np

from fashion_dataset import StdNormalizer, FashionDataset
from model_a import ModelA
from model_b import ModelB
from model_c import ModelC
from model_d import ModelD
from model_e import ModelE
from model_f import ModelF
from model_wrapper import ModelWrapper
from torch.utils.data import random_split

colors = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])

MODELS = {
    'A': ModelA,
    'B': ModelB,
    'C': ModelC,
    'D': ModelD,
    'E': ModelE,
    'F': ModelF,
}

MODELS_HYPER_PARAMS = {
    'A': {'lr': 0.01, 'batch_size': 64, 'epochs': 10},
    'B': {'lr': 0.01, 'batch_size': 80, 'epochs': 10},
    'C': {'lr': 0.001, 'batch_size': 64, 'epochs': 10},
    'D': {'lr': 0.01, 'batch_size': 64, 'epochs': 10},
    'E': {'lr': 0.01, 'batch_size': 64, 'epochs': 10},
    'F': {'lr': 0.01, 'batch_size': 250, 'epochs': 10},
}

MODELS_ADDITIONAL_PARAMS = {
    'C': {'dropout': [0.6, 0.4]}
}

FILE_NAME = 'test_y'


def normalize_data_std(data_set: ndarray):
    mean = data_set.mean(axis=0)
    std_dev = data_set.std(axis=0)
    return np.nan_to_num((data_set - mean) / std_dev)


def normalize_data_min_max(data_set: ndarray, new_min: int = 0, new_max: int = 1):
    data_min = data_set.min(axis=0)
    data_max = data_set.max(axis=0)
    return ((data_set - data_min) / (data_max - data_min)) * (new_max - new_min) + new_min


def run_model(training_set: FashionDataset, validation_set: FashionDataset, online_validation_set: FashionDataset,
              model_name: str, epochs: int, lr: float, batch_size: int):
    print(f'Running with model {model_name}')
    if model_name in MODELS_ADDITIONAL_PARAMS.keys():
        model = MODELS[model_name](training_set.data.shape[1], lr, MODELS_ADDITIONAL_PARAMS[model_name])
    else:
        model = MODELS[model_name](training_set.data.shape[1], lr)
    model_wrapper = ModelWrapper(model, epochs, batch_size)

    training_loss_arr, training_accuracy_arr = model_wrapper.train(training_set)
    print(f'Total Loss: {training_loss_arr[-1]}')
    validation_loss, validation_accuracy = model_wrapper.test(validation_set)
    online_validation_loss, online_validation_accuracy = model_wrapper.test(online_validation_set)
    avg_validation_loss = (validation_loss + online_validation_loss) / 2
    avg_validation_accuracy = (validation_accuracy + online_validation_accuracy) / 2
    return model_wrapper, (training_loss_arr, training_accuracy_arr), (avg_validation_loss, avg_validation_accuracy)


def load_test_set(test_x_path):
    test_set, _ = FashionDataset.from_files(test_x_path)
    test_transform_func = transforms.Compose([StdNormalizer(test_set.data)])
    test_set.set_transforms(test_transform_func)
    return test_set


def write_predictions_to_file(file_name: str, predictions: ndarray):
    with open(file_name, 'w') as file:
        file.write('\n'.join([str(i) for i in predictions.tolist()]))


def get_best_model_with_threads(train_x_path: str, train_y_path: str, num_of_rows: int = None):
    models = {}
    training_set, validation_set, online_validation_set = load_fashion_datasets(num_of_rows, train_x_path, train_y_path)
    futures = {}
    with ThreadPoolExecutor(6) as executor:
        for name in MODELS:
            params = MODELS_HYPER_PARAMS[name]
            futures[name] = executor.submit(run_model, training_set, validation_set, online_validation_set, name,
                                            params['epochs'], params['lr'], params['batch_size'])

        for name, future in zip(futures.keys(), concurrent.futures.as_completed(futures.values())):
            model_wrapper, training, validation = future.result()
            torch.save(model_wrapper.model.state_dict(), f'saved_models/{name}.pt')
            models[name] = {'model': model_wrapper, 'training': training, 'validation': validation}
    best_model_name = 'A'
    for name in models:
        if name == best_model_name:
            continue
        if models[name]['validation'][1] > models[best_model_name]['validation'][1]:
            best_model_name = name
    return models, best_model_name


def load_fashion_datasets(num_of_rows, train_x_path, train_y_path):
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
    return training_set, validation_set, online_validation_set


def main(train_x_path: str, train_y_path: str, test_x_path: str, num_of_rows: int = None):
    best_model, best_model_name = get_best_model_with_threads(train_x_path, train_y_path, num_of_rows)
    print(f'Best model is {best_model_name}')
    test_data = load_test_set(test_x_path)
    predictions = best_model[best_model_name]['model'].predict(test_data)
    write_predictions_to_file(FILE_NAME, predictions.numpy())


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
