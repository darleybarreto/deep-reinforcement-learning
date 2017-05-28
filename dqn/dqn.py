import math
import random
import pyautogui
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import namedtuple

opt =   {
            "Adam": torch.optim.Adam,\
            "RMSprop": torch.optim.RMSprop,\
            "SGD": torch.optim.SGD 
        }

class DQN(nn.Module):
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
        print("At forward method",x.size())
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
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

        self.memory[self.position] = Transition(*[torch.from_numpy(stack_x), \
                                                action,\
                                                torch.from_numpy(st),\
                                                torch.Tensor([reward])])

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def create_model(actions, shape, fully_connected, learning_rate=1e-2, opt_='RMSprop', **kwargs):
	
    BATCH_SIZE  = kwargs.get("BATCH_SIZE",128)
    GAMMA       = kwargs.get("GAMMA", 0.999)
    EPS_START   = kwargs.get("EPS_START", 0.9)
    EPS_END     = kwargs.get("EPS_END", 0.05)
    EPS_DECAY   = kwargs.get("EPS_DECAY", 200)
    path        = kwargs.get("path", None)
    memory      = kwargs.get("memory", ReplayMemory(10000))
    
    dqn = DQN(actions, shape, fully_connected)

    if path: load_model(path)
	
    optimizer = opt[opt_](dqn.parameters(), lr=learning_rate)
    steps_done = 0
    last_sync = 0

    def select_greddy_action(state):
        nonlocal steps_done

        sample = random.random()

        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        
        if sample > eps_threshold:
            state = torch.from_numpy(state).float().unsqueeze(0)
            probs = dqn(Variable(state, volatile=True))
            
            return probs.data.max(1)[1].cpu() # get the index of the max log-probability

        else:
            return torch.LongTensor([[random.randrange(actions)]])

    def perform_action(possible_actions, action):
        pyautogui.press(possible_actions[action])

    def optimize():
        ### Perform experience replay and train the network.
        nonlocal last_sync

        if len(memory) < BATCH_SIZE:
            return

        transitions = memory.sample(BATCH_SIZE)
        # Use the replay buffer to sample a batch of transitions
        
        batch = Transition(*zip(*transitions))
        # batch.state is a tuple of states
        # batch.action is a tuple os actions
        # batch.reward is a tuple of rewards
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]),volatile=True)
        
        state_batch = state_batch.view(1, *state_batch.size())
        
        # Compute current Q value, takes only state and output value for every state-action pair
        # We choose Q based on action taken.
        state_action_values = dqn(state_batch).gather(1, action_batch)
        # Compute next Q value based on which action gives max Q values
        next_state_values = Variable(torch.zeros(BATCH_SIZE))
        next_state_values[non_final_mask] = dqn(non_final_next_states).max(1)[0]

        next_state_values.volatile = False
        # Compute the target of the current Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # same as SmoothL1Loss
        # Creates a criterion that uses a squared term if 
        # the absolute element-wise error falls below 1 and an L1 term otherwise.
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Clears the gradients of all optimized Variable
        optimizer.zero_grad()

        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Variables with requires_grad=True.
        # After this call w1.grad and w2.grad will be Variables holding the gradient
        # of the loss with respect to w1 and w2 respectively.
        loss.backward()
        
        for param in dqn.parameters():
            param.grad.data.clamp_(-1, 1)
            
        optimizer.step()

    def save_model(path):
        if path: torch.save(dqn.state_dict(),path)

    def load_model(path): 
        dqn.load_state_dict(torch.load(path))

    def push_to_memory(*args):
        memory.push(*args)

    return push_to_memory, select_greddy_action, perform_action, optimize, save_model
	