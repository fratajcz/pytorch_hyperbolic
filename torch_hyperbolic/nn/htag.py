import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor, SparseTensor, PairTensor
from .hlinear import HypLinear
import torch_hyperbolic.manifolds as manifolds
from torch import Tensor

class HTAGConv(MessagePassing):
    """
    Hyperbolic graph convolution layer.

    It assumes that the input is already on the manifold and outputs the feature matrix on the manifold.

    Implementation based on https://github.com/HazyResearch/hgcn/blob/master/layers/hyp_layers.py 
    but implemented for the MessagePassing framework using the GCN template from https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html#implementing-the-gcn-layer
    """

    def __init__(self, in_channels, out_channels, c, manifold="PoincareBall", dropout: int = 0, K: int = 3, use_bias=True, aggr="add", normalize=True, local_agg=False):
        super().__init__(aggr=aggr)
        self.c = c
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.manifold = getattr(manifolds, manifold)()
        self.lins = torch.nn.ModuleList([
            HypLinear(in_channels, out_channels, c, manifold=manifold, dropout=dropout, use_bias=False) for _ in range(K + 1)
        ])

        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        self.normalize = normalize
        self.local_agg = local_agg
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight: OptTensor = None):
        """ Assumes that x is already on the manifold, i.e. that features are hyperbolic """
        # Step 1: Add self-loops to the adjacency matrix.
        #edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        out = self.lins[0](x)

        for lin in self.lins[1:]:
            # Step 3: Project Feature Matrix into Tangent Space.
            if not self.local_agg:
                x = self.manifold.logmap0(x, c=self.c)
            
            # Step 4: Start propagating messages.
            
            x_new = self.propagate(edge_index, x=x)

            # Step 5: Project Feature Map back in Hyperbolic Space.
            if self.local_agg:
                x = self.manifold.proj(self.manifold.expmap(x_new, x, c=self.c), c=self.c)
            else:
                x = self.manifold.proj(self.manifold.expmap0(x_new, c=self.c), c=self.c)

            out = self.manifold.proj(self.manifold.mobius_add(lin(x), out, c=self.c), c=self.c)


        # Step 6: Apply a final bias vector.
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            out = self.manifold.mobius_add(out, hyp_bias, c=self.c)
            out = self.manifold.proj(out, self.c)

        return out

    def message(self, x_i, x_j):
        """ If we use local aggregation, x_i and x_j are still on the manifold, else they are in tangent space of origin """
        # x_j has shape [E, out_channels]
        
        if self.local_agg:
            # use features projected into local tangent space of center node x_i
            x_j = self.manifold.logmap(x_j, x_i, c=self.c)

        return x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, dropout={self.dropout}, c={self.c}')
