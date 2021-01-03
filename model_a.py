import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim


class ModelA(nn.Module):
    def __init__(self, image_size: int, lr: float = 0.1, batch_size: int = 500):
        super(ModelA, self).__init__()
        self.image_size = image_size
        self.hidden_layer_0 = nn.Linear(image_size, 100)
        self.hidden_layer_1 = nn.Linear(100, 50)
        self.output_layer = nn.Linear(50, 10)
        self.lr = lr

    @property
    def optimizer(self):
        return optim.SGD(self.parameters(), lr=self.lr)

    @property
    def loss_function(self):
        return F.nll_loss

    def forward(self, data: Tensor):
        data = data.view(-1, self.image_size)
        data = F.relu(self.hidden_layer_0(data))
        data = F.relu(self.hidden_layer_1(data))
        data = F.relu(self.output_layer(data))
        return F.log_softmax(data, dim=1)
