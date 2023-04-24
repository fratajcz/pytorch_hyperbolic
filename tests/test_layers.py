
import unittest
import torch

from torch_hyperbolic.nn import HypAct, HypLinear, HyperbolicEncoder, HyperbolicDecoder, HGATConv, HGCNConv

class HypLinearTest(unittest.TestCase):

    def test_init_poincare(self):
        hlin = HypLinear(in_channels=10, out_channels=10, c=1.5)

    def test_init_hyperboloid(self):
        hlin = HypLinear(in_channels=10, out_channels=10, manifold="Hyperboloid", c=1.5)

    def test_forward_poincare(self):
        x_input = torch.rand((5, 10))

        hlin = HypLinear(in_channels=10, out_channels=8, c=1.5)
        x = hlin.forward(hlin.manifold.expmap0(x_input, hlin.c))
        self.assertEqual(x.shape, (5, 8))

        # check if implicitely calling forward works too
        x_direct = hlin(hlin.manifold.expmap0(x_input, hlin.c))
        self.assertTrue(torch.allclose(x_direct, x))

    def test_forward_hyperboloid(self):
        x_input = torch.rand((5, 10))

        hlin = HypLinear(in_channels=10, out_channels=10, manifold="Hyperboloid", c=1.5)
        x = hlin.forward(hlin.manifold.expmap0(x_input, hlin.c))
        self.assertTrue(not torch.allclose(x, x_input))

        # check if implicitely calling forward works too
        x_direct = hlin(hlin.manifold.expmap0(x_input, c=1.5))
        self.assertTrue(torch.allclose(x_direct, x))

    def test_forward_poincare_single_output(self):
        x_input = torch.rand((5, 10))

        hlin = HypLinear(in_channels=10, out_channels=1, manifold="PoincareBall", c=1.5)
        x = hlin.forward(x_input)

        # check if implicitely calling forward works too
        x_direct = hlin(x_input)
        self.assertTrue(torch.allclose(x_direct, x))

    def test_forward_hyperboloid_single_output(self):
        x_input = torch.rand((5, 10))

        hlin = HypLinear(in_channels=10, out_channels=1, manifold="Hyperboloid", c=1.5)
        x = hlin.forward(x_input)

        # check if implicitely calling forward works too
        x_direct = hlin(x_input)
        self.assertTrue(torch.allclose(x_direct, x))


class InitLayerTest(unittest.TestCase):

    def test_init(self):
        _ = HyperbolicEncoder(curvature=1.5)
        _ = HyperbolicDecoder(curvature=1.5)

    def test_encode_and_decode_poincare(self):
        values = torch.rand(5, 10)
        encoder = HyperbolicEncoder(curvature=1.5)
        decoder = HyperbolicDecoder(curvature=1.5)

        hyp = encoder(values)
        euclid = decoder(hyp)

        self.assertTrue(torch.allclose(values, euclid))

    def test_encode_and_decode_lorentz(self):

        values = torch.rand(5, 10)
        encoder = HyperbolicEncoder(curvature=1.5, manifold="Hyperboloid")
        decoder = HyperbolicDecoder(curvature=1.5, manifold="Hyperboloid")

        hyp = encoder(values)
        print(hyp.shape)
        euclid = decoder(hyp)

        self.assertTrue(torch.allclose(values, euclid))


class HypActTest(unittest.TestCase):

    def test_init(self):
        _ = HypAct(act=torch.nn.ELU(), c_in=1.5, c_out=1.5)

    def test_forward(self):
        x_input = torch.rand((5, 10))

        hact = HypAct(act=torch.nn.ELU(), c_in=1.5, c_out=0.5)
        x = hact.forward(hact.manifold.expmap0(x_input, c=1.5))
        self.assertTrue(not torch.allclose(x, x_input))

        # check if implicitely calling forward works too
        x_direct = hact(hact.manifold.expmap0(x_input, c=1.5))
        self.assertTrue(torch.allclose(x_direct, x))

    def test_forward_hyperboloid(self):
        x_input = torch.rand((5, 10))

        hact = HypAct(act=torch.nn.ELU(), c_in=1.5, c_out=1.5, manifold="Hyperboloid")
        x = hact.forward(x_input)
        self.assertTrue(not torch.allclose(x, x_input))

        # check if implicitely calling forward works too
        x_direct = hact(x_input)
        self.assertTrue(torch.allclose(x_direct, x))

class HGATConvTest(unittest.TestCase):

    def test_init(self):
        hgcn = HGATConv(in_channels=10, out_channels=10, c=1.5)

    def test_forward(self):
        edges = torch.LongTensor([[0, 1], [0, 2], [2, 3], [2, 4]])
        x_input = torch.rand((5, 10))

        hgcn = HGATConv(in_channels=10, out_channels=10, c=1.5)
        x = hgcn.forward(x_input, edges.T.long())
        self.assertTrue(not torch.allclose(x, x_input))

         # check if implicitely calling forward works too
        x_direct = hgcn(x_input, edges.T.long())
        self.assertTrue(torch.allclose(x_direct, x))

    def test_local_agg_poincare(self):
        edges = torch.LongTensor([[0, 1], [0, 2], [2, 3], [2, 4]])
        x_input = torch.rand((5, 10))

        hgcn0 = HGATConv(in_channels=10, out_channels=8, c=1.5)
        hgcnlocal = HGATConv(in_channels=10, out_channels=8, c=1.5, local_agg=True)

        hgcnlocal.lin = hgcn0.lin
        hgcnlocal.lin_src = hgcn0.lin_src
        hgcnlocal.lin_dst = hgcn0.lin_dst
        x0 = hgcn0.forward(hgcn0.manifold.expmap0(x_input, hgcn0.c),  edges.T.long())
        xlocal = hgcnlocal.forward(hgcnlocal.manifold.expmap0(x_input, hgcnlocal.c),  edges.T.long())
        self.assertTrue(not torch.allclose(x0, xlocal))


class HGCNConvTest(unittest.TestCase):

    def test_init(self):
        hgcn = HGCNConv(in_channels=10, out_channels=10, c=1.5)

    def test_forward(self):
        edges = torch.LongTensor([[0, 1], [0, 2], [2, 3], [2, 4]])
        x_input = torch.rand((5, 10))

        hgcn = HGCNConv(in_channels=10, out_channels=10, c=1.5)
        x = hgcn.forward(x_input, edges.T.long())
        self.assertTrue(not torch.allclose(x, x_input))

         # check if implicitely calling forward works too
        x_direct = hgcn(x_input, edges.T.long())
        self.assertTrue(torch.allclose(x_direct, x))

    def test_local_agg_poincare(self):
        edges = torch.LongTensor([[0, 1], [0, 2], [2, 3], [2, 4]])
        x_input = torch.rand((5, 10))

        hgcn0 = HGCNConv(in_channels=10, out_channels=8, c=1.5)
        hgcnlocal = HGCNConv(in_channels=10, out_channels=8, c=1.5, local_agg=True)

        hgcnlocal.lin = hgcn0.lin
        x0 = hgcn0.forward(hgcn0.manifold.expmap0(x_input, hgcn0.c),  edges.T.long())
        xlocal = hgcnlocal.forward(hgcnlocal.manifold.expmap0(x_input, hgcnlocal.c),  edges.T.long())
        self.assertTrue(not torch.allclose(x0, xlocal))

    def test_use_att_poincare(self):
        edges = torch.LongTensor([[0, 1], [0, 2], [2, 3], [2, 4]])
        x_input = torch.rand((5, 10))

        hgcn0 = HGCNConv(in_channels=10, out_channels=8, c=1.5)
        hgcnlocal = HGCNConv(in_channels=10, out_channels=8, c=1.5, local_agg=True, use_att=True)

        hgcnlocal.lin = hgcn0.lin
        x0 = hgcn0.forward(hgcn0.manifold.expmap0(x_input, hgcn0.c),  edges.T.long())
        xlocal = hgcnlocal.forward(hgcnlocal.manifold.expmap0(x_input, hgcnlocal.c),  edges.T.long())
        self.assertTrue(not torch.allclose(x0, xlocal))


if __name__ == '__main__':
    unittest.main()