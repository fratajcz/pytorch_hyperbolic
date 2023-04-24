import torch.nn as nn
import torch
from torch.nn.modules.module import Module
from torch_hyperbolic import manifolds


class HyperbolicDecoder(Module):
    def __init__(self, manifold: str = "PoincareBall", curvature=None):
        """ The decode() method of the LinearDecoder from https://github.com/HazyResearch/hgcn/edit/master/models/decoders.py as an explicit class
            This implementation does NOT include an additional Linear layer, so it is intended to be used before the final Linear layer """
        super(HyperbolicDecoder, self).__init__()
        self.curvature = nn.Parameter(torch.Tensor([1.])) if curvature is None else curvature
        self.manifold = getattr(manifolds, manifold)()

    def forward(self, x):
        """ Projects x from hyperbolic back into euclidean space """
        x = self.manifold.logmap0(x, c=self.curvature)
        if isinstance(self.manifold, manifolds.Hyperboloid):
            x = x[:, 1:]
        return x

    def extra_repr(self):
        return 'c={}'.format(self.curvature)
