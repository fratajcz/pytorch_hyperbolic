import torch
import copy
import torch.nn as nn

from typing import Callable, Optional, Tuple, Union

from torch_geometric.nn.dense.linear import Linear
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from .hact import HypAct
from .hlinear import HypLinear
import torch_hyperbolic.manifolds as manifolds
from torch_geometric.typing import PairTensor
from torch import Tensor

from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)

from typing import Callable, Optional


class HFiLMConv(MessagePassing):
    """
    Hyperbolic graph convolution layer.

    It assumes that the input is already on the manifold and outputs the feature matrix on the manifold.

    Implementation based on https://github.com/HazyResearch/hgcn/blob/master/layers/hyp_layers.py 
    but implemented for the MessagePassing framework using the GCN template from https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html#implementing-the-gcn-layer
    """

    def __init__(self, in_channels, out_channels, c, num_relations: int = 1, nn: Optional[Callable] = None, act: Optional[Callable] = nn.ReLU(), manifold="PoincareBall", dropout=0, aggr="mean", local_agg=False):
        super().__init__(aggr=aggr)
        self.num_relations = num_relations
        self.act = HypAct(act=act, c_in=c, c_out=c)
        self.c = c
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.manifold = getattr(manifolds, manifold)()

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lins = torch.nn.ModuleList()
        self.films = torch.nn.ModuleList()

        for _ in range(num_relations):
            self.lins.append(HypLinear(in_channels[0], out_channels, c=c, use_bias=False))
            if nn is None:
                film = Linear(in_channels[1], 2 * out_channels)
            else:
                film = copy.deepcopy(nn)
            self.films.append(film)

        self.lin_skip = HypLinear(in_channels[1], self.out_channels, c=c, use_bias=False)

        if nn is None:
            self.film_skip = Linear(in_channels[1], 2 * self.out_channels, bias=False)
        else:
            self.film_skip = nn.copy.deepcopy(nn)

        self.reset_parameters()
        self.local_agg = local_agg
        self.dropout = dropout

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_type: OptTensor = None) -> Tensor:
        """ Assumes that x is already on the manifold, i.e. that features are hyperbolic """
        
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        x_tangent = self.manifold.proj(self.manifold.expmap0(x[1], c=self.c), c=self.c)

        beta, gamma = self.film_skip(x_tangent).split(self.out_channels, dim=-1)

        out = gamma * self.lin_skip(x[1])

        flag = torch.any(torch.isnan(out))

        beta = self.manifold.proj_tan0(beta, self.c)
        hyp_beta = self.manifold.expmap0(beta, self.c)
        hyp_beta = self.manifold.proj(hyp_beta, self.c)
        out = self.manifold.mobius_add(out, hyp_beta, c=self.c)
        out = self.manifold.proj(out, self.c)

        flag = torch.any(torch.isnan(out))

        if self.act is not None:
            out = self.act(out)
        
        out = self.manifold.logmap0(out, self.c)
        flag = torch.any(torch.isnan(out))
        # propagate_type: (x: Tensor, beta: Tensor, gamma: Tensor)
        if self.num_relations <= 1:
            beta, gamma = self.films[0](x_tangent).split(self.out_channels, dim=-1)
            beta = self.manifold.proj_tan0(beta, self.c)
            hyp_beta = self.manifold.expmap0(beta, self.c)
            hyp_beta = self.manifold.proj(hyp_beta, self.c)
            out = out + self.propagate(edge_index, x=self.lins[0](x[0]),
                                       beta=hyp_beta, gamma=gamma, size=None)
            
            flag = torch.any(torch.isnan(out))
        else:
            for i, (lin, film) in enumerate(zip(self.lins, self.films)):
                beta, gamma = film(x_tangent).split(self.out_channels, dim=-1)
                beta = self.manifold.proj_tan0(beta, self.c)
                hyp_beta = self.manifold.expmap0(beta, self.c)
                hyp_beta = self.manifold.proj(hyp_beta, self.c)
                if isinstance(edge_index, SparseTensor):
                    edge_type = edge_index.storage.value()
                    assert edge_type is not None
                    mask = edge_type == i
                    adj_t = torch_sparse.masked_select_nnz(
                        edge_index, mask, layout='coo')
                    out = out + self.propagate(adj_t, x=lin(x[0]), beta=hyp_beta,
                                               gamma=gamma, edge_type=i, size=None)
                else:
                    assert edge_type is not None
                    mask = edge_type == i
                    out = out + self.propagate(edge_index[:, mask], x=lin(
                        x[0]), beta=hyp_beta, gamma=gamma, edge_type=i, size=None)
                flag = torch.any(torch.isnan(out))

        return out

    def message(self, x_i, x_j, gamma_i, beta_i, edge_type):
        """ x_i and x_j are always on the manifold, beta_i is on the manifold too """
        # x_j has shape [E, out_channels]
        
        if self.local_agg:
            # use features projected into local tangent space of center node x_i
            x_j = self.manifold.proj(self.manifold.logmap(x_j, x_i, c=self.c), c=self.c)
            beta, gamma = self.films[edge_type](x_j).split(self.out_channels, dim=-1)
        
        out = gamma_i * x_j

        out = self.manifold.mobius_add(out, beta_i, c=self.c)
        out = self.manifold.proj(out, self.c)

        if self.act is not None:
            out = self.act(out)

        #now we have to bring features into euclidean for aggregation:
        if not self.local_agg:

            out = self.manifold.logmap0(x_j, c=self.c)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, dropout={self.dropout}, c={self.c}')
