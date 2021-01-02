from typing import Callable

import torch
from torch.utils.data import DataLoader
from torch import Tensor
import torch.nn as nn

from fashion_dataset import FashionDataset


class ModelWrapper:
    def __init__(self, model: nn.Module, epochs: int):
        self.model = model
        self.epochs = epochs
        self.optimizer = model.optimizer
        self.loss_function = model.loss_function

    def train(self, training_data, batch_size: int) -> float:
        train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=4)
        self.model.train()
        running_loss = 0.0
        for i in range(self.epochs):
            running_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f'Epoch: {i + 1} loss: {running_loss}')
        return running_loss

    def train_model(self, training_data, batch_size: int = 10):
        last_loss = self.train(training_data, batch_size)
        for i in range(1):
            if last_loss <= 5:
                break
            last_loss = self.train(training_data,batch_size)
        return last_loss

    def test(self, data, batch_size: int = 10) -> float:
        test_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=4)
        total = 0
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                predictions = torch.argmax(outputs.data, 1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
        return 100 * correct / total

    def predict(self, data: FashionDataset) -> Tensor:
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data.data_x)
            predictions = torch.argmax(outputs.data, 1)
            return predictions
