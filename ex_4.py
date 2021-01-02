import math
import sys
from numpy import ndarray
from torchvision import transforms
import numpy as np

from fashion_dataset import StdNormalizer, FashionDataset
from model_a import ModelA
from model_wrapper import ModelWrapper


def normalize_data_std(data_set: ndarray):
    mean = data_set.mean(axis=0)
    std_dev = data_set.std(axis=0)
    return np.nan_to_num((data_set - mean) / std_dev)


def normalize_data_min_max(data_set: ndarray, new_min: int = 0, new_max: int = 1):
    data_min = data_set.min(axis=0)
    data_max = data_set.max(axis=0)
    return ((data_set - data_min) / (data_max - data_min)) * (new_max - new_min) + new_min


def main(train_x_path: str, train_y_path: str, test_x_path: str, num_of_rows: int = None):
    training_set, validation_set = FashionDataset.from_files(train_x_path, train_y_path, num_of_rows,
                                                             0.2)
    test_set, _ = FashionDataset.from_files(test_x_path)
    train_transform_func = transforms.Compose([StdNormalizer(training_set.data_x)])
    training_set.set_transforms(train_transform_func)
    validation_set_transforms = transforms.Compose([StdNormalizer(validation_set.data_x)])
    validation_set.set_transforms(validation_set_transforms)
    test_transform_func = transforms.Compose([StdNormalizer(test_set.data_x)])
    test_set.set_transforms(test_transform_func)
    model = ModelA(training_set.shape(1))
    model_wrapper = ModelWrapper(model, 20)
    batch_size = 500
    total_loss = model_wrapper.train(training_set, batch_size=batch_size)
    print(f'Total Loss: {total_loss}')
    accuracy = model_wrapper.test(validation_set, batch_size=batch_size)
    print(f'Accuracy: {accuracy}')
    predict = model_wrapper.predict(test_set)
    with open('test_y', 'w') as file:
        for pred in predict.numpy():
            file.write(f'{pred}\n')
    return predict


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
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
