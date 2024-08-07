import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 30)
        # self.fc3 = nn.Linear(80, 60)
        # self.fc4 = nn.Linear(60, 40)
        # self.fc5 = nn.Linear(40, 30)
        self.fce = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        # x = F.relu(self.fc2(x))
        # x = F.dropout(x, training=self.training)
        # x = F.relu(self.fc3(x))
        # x = F.dropout(x, training=self.training)
        # x = F.relu(self.fc4(x))
        # x = F.dropout(x, training=self.training)
        # x = F.relu(self.fc5(x))
        # x = F.dropout(x, training=self.training)
        x = self.fce(x)

        return F.softmax(x)