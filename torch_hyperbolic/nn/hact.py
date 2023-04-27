"""Hyperbolic layers."""
import torch.nn as nn
from torch.nn.modules.module import Module


class HypAct(Module):
    def __init__(self, act, c_in=None, c_out=None, manifold="PoincareBall"):
        """
        Hyperbolic activation layer.

        Assumes input features are on a manifold with curvature c_in, brings them into euclidean space, applies the activation,
        and transforms the output into hyperbolic space with curvature c_out. 
        
        In case only translation between two curvatures is desired, torch.nn.Identity() can be passed as activation.
        """
        super(HypAct, self).__init__()
        from torch_hyperbolic.nn import HyperbolicDecoder, HyperbolicEncoder
        self.decoder = HyperbolicDecoder(manifold=manifold, curvature=c_in) if c_in is not None else c_in
        self.act = act
        self.encoder = HyperbolicEncoder(manifold=manifold, curvature=c_out) if c_out is not None else c_out

    def forward(self, x):
        if self.decoder is not None:
            x = self.decoder(x)

        x = self.act(x)

        if self.encoder is not None:
            x = self.encoder(x)

        return x