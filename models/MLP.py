import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, num_features=49, hidden1_size=64, hidden2_size=64, num_classes=6):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(num_features, hidden1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
