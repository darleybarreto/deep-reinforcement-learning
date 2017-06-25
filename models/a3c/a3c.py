import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from torchvision.utils import save_image

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

class A3C(nn.Module):
    def __init__(self, shape, fully, lstm, num_outputs):
        super(A3C, self).__init__()

        in_, out_, kernel, stride, padding = (list(zip(*shape))[i] for i in range(len(shape) + 1))
        
        self.conv1 = nn.Conv2d(in_[0], out_[0], kernel_size=kernel[0], stride=stride[0], padding=padding[0])
        self.conv2 = nn.Conv2d(in_[1], out_[1], kernel_size=kernel[1], stride=stride[1], padding=padding[1])
        self.conv3 = nn.Conv2d(in_[2], out_[2], kernel_size=kernel[2], stride=stride[2], padding=padding[2])
        self.conv4 = nn.Conv2d(in_[3], out_[3], kernel_size=kernel[3], stride=stride[3], padding=padding[3])

        self.lstm = nn.LSTMCell(*lstm)
        self.critic_linear = nn.Linear(*fully)
        self.actor_linear = nn.Linear(fully[0], num_outputs)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()
    
    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        # save_image(inputs.data[0, 0, :, :], "before_conv.png")
        x = F.relu(F.max_pool2d(self.conv1(inputs), kernel_size=2, stride=2))
        # save_image(x.data[0, 0, :, :], "conv1.png")
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        # save_image(x.data[0, 0, :, :], "conv2.png")
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2, stride=2))
        # save_image(x.data[0, 0, :, :], "conv3.png")
        x = F.relu(F.max_pool2d(self.conv4(x), kernel_size=2, stride=2))
        # save_image(x.data[0, 0, :, :], "conv4.png")
        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))

        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)


def create_model(model_conf): 
    a3cmodel = A3C(*model_conf)
    # if path: dqn.load_state_dict(torch.load(path))
    def select_action(state, hx, cx, model, isTrain):
        info_dict = {}
        state = torch.from_numpy(state).unsqueeze(0).float()
        # print("TYPE >>> ",type(state))
        value, logit, (hx, cx) = model((Variable(state), (hx, cx)))

        prob = F.softmax(logit)
        
        if not isTrain:
            action = prob.max(1)[1].data.numpy()
        
        else:
            info_dict["log_probs"] = F.log_softmax(logit)
            entropy = -(info_dict["log_probs"] * prob).sum(1)
            info_dict["entropies"] = entropy
            action = prob.multinomial().data
            info_dict["log_probs"] = info_dict["log_probs"].gather(1, Variable(action))

            info_dict["values"] = value

        return action, hx, cx, info_dict # get the index of the max log-probability


    def perform_action(f_action, possible_actions, action):
        # reward
        return f_action(possible_actions[action[0][0]])

    def save_model(model, path):
        if path and model: torch.save(model.state_dict(),path)

    return select_action, perform_action, a3cmodel,save_model
	