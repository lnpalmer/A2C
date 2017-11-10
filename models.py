import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def normalized_columns(shape):
    weights = torch.randn(shape)
    weights *= 1. / torch.sqrt(weights.pow(2).sum(1, keepdim=True))
    return weights

def initializer(module):
    """ Parameter initializer for AtariNet

    Initializes Linear, Conv2d, and LSTMCell weights
    """
    classname = module.__class__.__name__

    if classname == 'Linear':
        weight_shape = module.weight.data.size()
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        module.weight.data.uniform_(-w_bound, w_bound)
        module.bias.data.zero_()

    elif classname == 'Conv2d':
        weight_shape = module.weight.data.size()
        fan_in = weight_shape[0] * weight_shape[2] * weight_shape[3]
        fan_out = weight_shape[0] * weight_shape[1]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        module.weight.data.uniform_(-w_bound, w_bound)
        module.bias.data.zero_()

    elif classname == 'LSTMCell':
        module.bias_ih.data.zero_()
        module.bias_hh.data.zero_()

class AtariNet(nn.Module):
    """ Basic convolutional+LSTM network for Atari environments

    Equivalent to OpenAI's universe-starter-agent
    """

    def __init__(self, policy_size, lstm_size=256):
        super().__init__()

        self.conv = nn.Sequential(nn.Conv2d(1, 32, 3, stride=2, padding=1),
                                  nn.ELU(inplace=True),
                                  nn.Conv2d(32, 32, 3, stride=2, padding=1),
                                  nn.ELU(inplace=True),
                                  nn.Conv2d(32, 32, 3, stride=2, padding=1),
                                  nn.ELU(inplace=True),
                                  nn.Conv2d(32, 32, 3, stride=2, padding=1),
                                  nn.ELU(inplace=True))

        self.lstm = nn.LSTMCell(32 * 3 * 3, lstm_size)

        self.policy = nn.Linear(lstm_size, policy_size)
        self.value = nn.Linear(lstm_size, 1)

        self.policy_size = policy_size
        self.lstm_size = lstm_size

        # initialize parameters
        self.apply(initializer)
        self.policy.weight.data = \
            normalized_columns(self.policy.weight.size()) * 0.01

        self.value.weight.data = normalized_columns(self.value.weight.size())

    def forward(self, conv_in, lstm_state_in):
        """ PyTorch forward pass """
        conv_out = self.conv(conv_in)

        lstm_in = conv_out.view(-1, 3 * 3 * 32)
        lstm_state_out = self.lstm(lstm_in, lstm_state_in)
        lstm_out, _ = lstm_state_out

        policy = self.policy(lstm_out)
        value = self.value(lstm_out)

        return policy, value, lstm_state_out

    def make_state(self, count=1):
        """ Makes an initial network state """
        return (Variable(torch.zeros(count, self.lstm_size)),
                Variable(torch.zeros(count, self.lstm_size)))
