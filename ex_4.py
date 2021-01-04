import math
import sys
import matplotlib.pyplot as plt
import torch
from numpy import ndarray
from scipy.io import wavfile
from torchvision import transforms, datasets
import numpy as np

from fashion_dataset import StdNormalizer, FashionDataset
from model_a import ModelA
from model_wrapper import ModelWrapper
from torch.utils.data import random_split

colors = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])

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
    return (loss_arr, accuracy_arr), (loss, accuracy)


def generate_report(train_x_path: str, train_y_path: str, test_x_path: str, num_of_rows: int = None):
    model_list = list(MODELS.keys())
    epochs = 15
    accuracy_training = []
    loss_training = []
    accuracy_validation = []
    loss_validation = []
    mode = ['loss', 'accuracy']
    # training graphs
    for model in model_list:
        training, validation = run_model(train_x_path, train_y_path, test_x_path, model, epochs, 0.1, 500, num_of_rows)
        loss_training.append(np.array(training[0]))
        accuracy_training.append(np.array(training[1]))
        loss_validation.append(validation[0])
        accuracy_validation.append(validation[1])
    train = [loss_training, accuracy_training]
    vali = [loss_validation, accuracy_validation]
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), dpi=500)
    axs = axs.flatten()
    fig.suptitle('Results', fontsize=20)
    # validation graphs
    for i in range(2):
        axs[i].set_title(f'$Training$ ${mode[i]}$')
        axs[i].set_xlabel('$Iterations$')
        axs[i].set_ylabel(f'$Average$ ${mode[i]}$')
        for model_index, color in zip(range(len(model_list)), colors):
            axs[i].plot(train[i][model_index], c=color, lw=2, label=model_list[model_index])
        axs[i].legend(ncol=4, fontsize='large', loc='best')
    for i in range(2, 4):
        axs[i].set_title(f'$Validation$ ${mode[i % 2]}$')
        axs[i].bar(model_list, vali[i % 2])
    plt.savefig('plot.jpeg')


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
        # main(sys.argv[1], sys.argv[2], sys.argv[3], NUM_OF_ROWS)
        generate_report(sys.argv[1], sys.argv[2], sys.argv[3], NUM_OF_ROWS)
