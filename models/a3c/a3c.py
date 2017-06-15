import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import namedtuple
# from torchvision.utils import save_image

class A3C(nn.Module):
    def __init__(self, actions, shape, fc):
        super(DQN, self).__init__()

        # that there are no pooling layers
        # pooling layers buys translation invariance:
        # the network becomes insensitive to the location of an object in the image.
        # That makes perfectly sense for a classification task like ImageNet.
        # But for games the location of an object is crucial in determining the potential reward

        # The BatchNorm layer immediately after fully connected layers (or convolutional layers), and before non-linearities.
        # It has become a very common practice to use Batch Normalization in neural networks. 
        # In practice networks that use Batch Normalization are significantly more robust to bad initialization. 
        # Additionally, batch normalization can be interpreted as doing preprocessing at every layer of the network, 
        # but integrated into the network itself in a differentiable manner
        # 
        # Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift paper
        # https://arxiv.org/abs/1502.03167
        
        in_, out_, kernel, stride = (list(zip(*shape))[i] for i in range(len(shape) + 1))
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_[0], out_[0], kernel_size=kernel[0], stride=stride[0]), 
            nn.BatchNorm2d(out_[0]),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_[1], out_[1], kernel_size=kernel[1], stride=stride[1]), 
            nn.BatchNorm2d(out_[1]),
            nn.ReLU())

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_[2], out_[2], kernel_size=kernel[2], stride=stride[2]), 
            nn.BatchNorm2d(out_[2]),
            nn.ReLU())

        self.fc1 = nn.Linear(fc[0], fc[1])
        self.fc2 = nn.Linear(fc[1], actions)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # save_image(x.data,"3_conv.png")

        # x.size(0) get the 0 component of its size
        # x.view() is basically to reshape the tensor
        # the -1 means that for a given number of rows
        # we want to Pytorch find the best number of columns
        # that fits our data
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

######################################################################
# Replay Memory
# -------------
#
# We'll be using experience replay memory for training our DQN. It stores
# the transitions that the agent observes, allowing us to reuse this data
# later. By sampling from it randomly, the transitions that build up a
# batch are decorrelated. It has been shown that this greatly stabilizes
# and improves the DQN training procedure.
#
# For this, we're going to need two classses:
#
# - Transition - representing a single transition in our environment
# - ReplayMemory - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a .sample()
#    method for selecting a random batch of transitions for training.
######################################################################

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        [stack_x, action, st, reward] = args
        
        stack_x = torch.from_numpy(stack_x.astype(float).reshape((1, *stack_x.shape)))
        st = torch.from_numpy(st.astype(float).reshape(1, *st.shape))
        
        self.memory[self.position] = Transition(*[stack_x, \
                                                action,\
                                                st,\
                                                torch.Tensor([reward])])

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def create_model(actions, shape, fully_connected, learning_rate=1e-2, opt_='RMSprop', **kwargs): 

    # if path: dqn.load_state_dict(torch.load(path))

    values = []
    log_probs = []
    entropies = []
	
    def select_action(state, hx, cx, isTrain=True):

        value, logit, (hx, cx) = model((Variable(state.unsqueeze(0), volatile=True), (hx, cx)))

        prob = F.softmax(logit)
        
        if not isTrain:
            action = prob.max(1)[1].data.numpy()
        
        else:
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)
            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))

            values.append(value)
            log_probs.append(log_prob)

        return action, hx, cx # get the index of the max log-probability


    def perform_action(f_action, possible_actions, action):
        # reward
        return f_action(possible_actions[action])

    def optimize():
        pass

    def save_model(path):
        if path: torch.save(dqn.state_dict(),path)
 
    def push_to_memory(*args):
        memory.push(*args)

    return push_to_memory, select_action, perform_action, optimize, save_model
	