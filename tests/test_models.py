import unittest
import torch

from torch_hyperbolic.models.hgnn import HGNN

class HGNNTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_dim = 10
        self.out_dim = 1
        self.n_nodes = 5

        self.x_input = torch.rand((self.n_nodes, self.in_dim))
        self.adj = torch.LongTensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])

    def test_init_poincare(self):
        _ = HGNN(in_channels=10, out_channels=1, hidden_dim=5)

    def test_init_hyperboloid(self):
        _ = HGNN(in_channels=10, out_channels=1, hidden_dim=5, manifold="Hyperboloid")

    def test_forward_poincare(self):
        model = HGNN(in_channels=self.in_dim, out_channels=self.out_dim, hidden_dim=5)
        out = model(self.x_input, self.adj)
        self.assertEqual(out.shape, (self.n_nodes, self.out_dim))

    def test_forward_hyperboloid(self):
        model = HGNN(in_channels=self.in_dim, out_channels=self.out_dim, hidden_dim=5)
        out = model(self.x_input, self.adj)
        self.assertEqual(out.shape, (self.n_nodes, self.out_dim))
