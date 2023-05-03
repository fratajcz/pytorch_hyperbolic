"""Example Implementation of a small hyperbolic graph neural network"""

import torch.nn as nn
import torch
import torch_hyperbolic.nn as hypnn

class HGNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, manifold="PoincareBall", dropout=0, act=torch.nn.ELU(), trainable_curvature=True, init_curvature=1, gcn_kwargs={}):
        super(HGNN, self).__init__()
        n_layers = 4  # one linear input layer, 2 GNN layers, 1 linear output layer

        requires_grad = trainable_curvature
        init_val = float(init_curvature) if init_curvature is not None else 1
        self.curvatures = [nn.Parameter(torch.Tensor([init_val]), requires_grad=requires_grad) for _ in range(n_layers)]

        if manifold == "Hyperboloid":
            in_channels += 1
            hidden_dim += 1
            out_channels += 1

        self.encoder = hypnn.HyperbolicEncoder(manifold=manifold, curvature=self.curvatures[0])
        self.input_lin = hypnn.HypLinear(manifold=manifold, in_channels=in_channels, out_channels=hidden_dim, c=self.curvatures[0])
        self.act0 = hypnn.HypAct(act, manifold=manifold, c_in=self.curvatures[0], c_out=self.curvatures[1])
        self.dropout0 = nn.Dropout(p=dropout)

        self.gnn1 = hypnn.HGCNConv(in_channels=hidden_dim, out_channels=hidden_dim, manifold=manifold, c=self.curvatures[1], dropout=dropout, **gcn_kwargs)
        self.act1 = hypnn.HypAct(act, manifold=manifold, c_in=self.curvatures[1], c_out=self.curvatures[2])
        self.dropout1 = nn.Dropout(p=dropout)

        self.gnn2 = hypnn.HGCNConv(in_channels=hidden_dim, out_channels=hidden_dim, manifold=manifold, c=self.curvatures[2], dropout=dropout, **gcn_kwargs)
        self.act2 = hypnn.HypAct(act, manifold=manifold, c_in=self.curvatures[2], c_out=self.curvatures[3])
        self.dropout2 = nn.Dropout(p=dropout)

        self.output_lin = hypnn.HypLinear(manifold=manifold, in_channels=hidden_dim, out_channels=out_channels, c=self.curvatures[3])
        self.decoder = hypnn.HyperbolicDecoder(manifold=manifold, curvature=self.curvatures[3])

    def forward(self, x, adj):
        # bring x into hyperbolic space
        x = self.encoder(x)

        # pass through input layers
        x = self.dropout0(self.act0(self.input_lin(x)))

        # pass through GNN
        x = self.dropout1(self.act1(self.gnn1(x, adj)))
        x = self.dropout2(self.act2(self.gnn2(x, adj)))

        # pass through classifier
        x = self.output_lin(x)

        # bring back into euclidean space
        x = self.decoder(x)

        return x