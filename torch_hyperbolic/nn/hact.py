"""Hyperbolic layers."""
import torch.nn as nn
from torch.nn.modules.module import Module


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, act, c_in=None, c_out=None, manifold="PoincareBall"):
        super(HypAct, self).__init__()
        from torch_hyperbolic.nn import HyperbolicDecoder, HyperbolicEncoder
        #self.manifold = getattr(manifolds, manifold)()
        self.act = act
        #self.c_in = c_in
        #self.c_out = c_out
        self.decoder = HyperbolicDecoder(manifold=manifold, curvature=c_in) if c_in is not None else c_in
        self.encoder = HyperbolicEncoder(manifold=manifold, curvature=c_out) if c_out is not None else c_in

    def forward(self, x):
        if self.decoder is not None:
            x = self.decoder(x)

        x = self.act(x)

        if self.encoder is not None:
            x = self.encoder(x)

        return x

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)