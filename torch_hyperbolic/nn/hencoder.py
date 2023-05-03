import torch.nn as nn
import torch
from torch.nn.modules.module import Module
from torch_hyperbolic import manifolds


class HyperbolicEncoder(Module):
    def __init__(self, manifold: str = "PoincareBall", curvature=None):
        """ The encode() method of the HGCN and HNN from https://github.com/HazyResearch/hgcn/edit/master/models/encoders.py as an explicit class. 
        This layer does not include any linear layers, it only translates the features from euclidean space onto the manifold with curvature c."""
        super(HyperbolicEncoder, self).__init__()
        self.curvature = nn.Parameter(torch.Tensor([1.])) if curvature is None else curvature
        self.manifold = getattr(manifolds, manifold)()

    def forward(self, x):
        """ Projects x into hyperbolic space:

        .. math::

            \mathbf{X}^{\prime} = \textrm{exp}_\mathbf{o}^c \left( \mathbf{X} \right)

        where exp() :math:`\textrm{exp} \left( \right)` is given as

        .. math::

            \textrm{exp}_\mathbf{o}^c \left( \mathbf{v} \right) = \mathbf{0} \oplus_c \left( \textrm{tanh} \left( \sqrt{|c|} \frac{\lambda_\mathbf{o}^c || \mathbf{v} || _{2}}{2} \frac{\mathbf{v}}{\sqrt{|c| \mathbf{v} || _{2}}} \right) \right)
        
        for PoincareBall Manifold and

        .. math::

            \textrm{exp}_\mathbf{o}^c \left( \mathbf{v} \right) = \textrm{cosh} \left( \sqrt{|c|} || \mathbf{v} || _{\mathcal{L}} \right) \mathbf{0} + \mathbf{v} \frac{\textrm{sinh} \left( \sqrt{|c|} || \mathbf{v} || _{\mathcal{L}} \right) }{\sqrt{|c| || \mathbf{v} || _{\mathcal{L}}}}

        for Hyperboloid Manifold (Lorentz Model)

        In case the manifold is a hyperoloid (Lorentz model), the output will have n+1 dimensions"""

        if isinstance(self.manifold, manifolds.Hyperboloid):
            x = torch.cat((torch.zeros_like(x)[:, 0:1], x), dim=-1)

        x_tan = self.manifold.proj_tan0(x, self.curvature)

        x_hyp = self.manifold.expmap0(x_tan, c=self.curvature)
 
        x_hyp = self.manifold.proj(x_hyp, c=self.curvature)

        return x_hyp

    def extra_repr(self):
        return 'c={}'.format(self.curvature)
