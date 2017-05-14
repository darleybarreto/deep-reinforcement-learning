import random
import pythonautogui
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

opt =   {
            "Adam": torch.optim.Adam,\
            "RMSprop": torch.optim.RMSprop,\
            "SGD": torch.optim.SGD 
        }

class DQN(nn.Module):
	def __init__(self, shape, kernel=5):
        super(DQN, self).__init__()

        self.layers = {}

        # that there are no pooling layers
        # pooling layers buys translation invariance:
        # the network becomes insensitive to the location of an object in the image.
        # That makes perfectly sense for a classification task like ImageNet.
        # But for games the location of an object is crucial in determining the potential reward

        for l, i in enumerate(shape):
        	in_, out_, stride = shape[i]
            self.layers["layer%s"%l] = nn.Sequential(
                nn.Conv2d(in_, out_, kernel_size=kernel, stride=stride),
                nn.BatchNorm2d(out_),
                nn.ReLU())

        # self.fc1 = nn.Linear(7*7*32, 10)

    def forward(self, x):
        for l in self.layers:
            x = self.layers[l](x)

		x = x.view(x.size(0), -1)
        # x = self.fc1(x)
        return x


def create_model(shape, learning_rate, opt_='Adam', **kwargs):
	
    BATCH_SIZE  = kwargs.get("BATCH_SIZE",128)
    GAMMA       = kwargs.get("GAMMA", 0.999)
    EPS_START   = kwargs.get("EPS_START", 0.9)
    EPS_END     = kwargs.get("EPS_END", 0.05)
    EPS_DECAY   = kwargs.get("EPS_DECAY", 200)

    dqn = DQN(shape)
	optimizer = opt[opt_](dqn.parameters(), lr=learning_rate)
    steps_done = 0
    sample = random.random()

	def select_action(state):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        
        if sample > eps_threshold:
    		state = torch.from_numpy(state).float().unsqueeze(0)
    		probs = dqn(Variable(state))
    		action = probs.multinomial()
            return action.data.max(1)[1].cpu() # get the index of the max log-probability

        else:
            return torch.LongTensor([[random.randrange(2)]])

    def perform_action(possible_actions, action):
        pyautogui.press(possible_actions[action])

    def optimize():
        pass

    return dqn, select_action, perform_action, optimize
	