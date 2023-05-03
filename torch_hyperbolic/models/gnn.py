"""Example Implementation of a small hyperbolic graph neural network"""

import torch.nn as nn
import torch
import torch_geometric as pyg

class GNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, dropout=0, act="ELU", trainable_curvature=True, init_curvature=1, gcn_kwargs={}):
        super(GNN, self).__init__()
        n_layers = 4  # one linear input layer, 2 GNN layers, 1 linear output layer

        requires_grad = trainable_curvature
        init_val = float(init_curvature) if init_curvature is not None else 1
        self.curvatures = [nn.Parameter(torch.Tensor([init_val]), requires_grad=requires_grad) for _ in range(n_layers)]

        self.input_lin = pyg.nn.Linear(in_channels=in_channels, out_channels=hidden_dim)
        self.act0 = getattr(torch.nn, act)()
        self.dropout0 = nn.Dropout(p=dropout)

        self.gnn1 = pyg.nn.GCNConv(in_channels=hidden_dim, out_channels=hidden_dim, **gcn_kwargs)
        self.act1 = getattr(torch.nn, act)()
        self.dropout1 = nn.Dropout(p=dropout)

        self.gnn2 = pyg.nn.GCNConv(in_channels=hidden_dim, out_channels=hidden_dim, **gcn_kwargs)
        self.act2 = getattr(torch.nn, act)()
        self.dropout2 = nn.Dropout(p=dropout)

        self.output_lin = pyg.nn.Linear(in_channels=hidden_dim, out_channels=out_channels)

    def forward(self, x, adj):

        # pass through input layers
        x = self.dropout0(self.act0(self.input_lin(x)))

        # pass through GNN
        x = self.dropout1(self.act1(self.gnn1(x, adj)))
        x = self.dropout2(self.act2(self.gnn2(x, adj)))

        # pass through classifier
        x = self.output_lin(x)

        return x