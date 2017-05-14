import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class DQN(nn.Module):
	def __init__(self, shape):
        super(DQN, self).__init__()

        in_, out_, stride, kernel = ([] for i in range(3))
        for i in shape:
        	in_, out_, stride, kernel = shape[i]

       	self.layer1 = nn.Sequential(
            nn.Conv2d(in_[0], out_[0], kernel[0], stride=stride[0]),
            nn.BatchNorm2d(out_[0]),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_[1], out_[1], kernel[1], stride=stride[1]),
            nn.BatchNorm2d(out_[1]),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_[2], out_[2], kernel[2], stride=stride[2]),
            nn.BatchNorm2d(out_[2]),
            nn.ReLU(),
            nn.MaxPool2d(2))

        # self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)

		x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x


def create_model(shape):
	dqn = DQN(shape)
	optimizer = torch.optim.Adam(dqn.parameters(), lr=learning_rate)

	def select_action(state):
		state = torch.from_numpy(state).float().unsqueeze(0)
		probs = dqn(Variable(state))
		action = probs.multinomial()
		return action.data

	