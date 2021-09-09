import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MacroAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MacroAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape

        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_subgoals)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        v = self.fc2(h)
        return v, h