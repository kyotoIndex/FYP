import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hetsa_research import transport_private_b, shared_a_update_round, suffix_layers


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.a = torch.randn(4, 9, dtype=torch.float64)
        self.b = torch.randn(7, 4, dtype=torch.float64)

    def test_identity_and_zero_strength_preserve_parameters(self):
        self.assertTrue(torch.equal(transport_private_b(self.a,self.a,self.b), self.b))
        self.assertTrue(torch.equal(transport_private_b(self.a,2*self.a,self.b,0), self.b))

    def test_rotated_basis_restores_function(self):
        q, _ = torch.linalg.qr(torch.randn(4,4,dtype=torch.float64))
        new_a = q @ self.a
        new_b = transport_private_b(self.a,new_a,self.b,ridge_relative=1e-9)
        torch.testing.assert_close(new_b @ new_a,self.b @ self.a,rtol=1e-6,atol=1e-6)

    def test_ridge_solution_satisfies_stationarity(self):
        new_a = torch.randn_like(self.a)
        new_b = transport_private_b(self.a,new_a,self.b)
        ridge = 1e-3 * (new_a @ new_a.T).trace()/4 + 1e-8
        gradient = (new_b @ new_a-self.b @ self.a) @ new_a.T+ridge*(new_b-self.b)
        torch.testing.assert_close(gradient,torch.zeros_like(gradient),rtol=0,atol=1e-10)
        def objective(b):
            return (b@new_a-self.b@self.a).square().sum()+ridge*(b-self.b).square().sum()
        self.assertLessEqual(objective(new_b),objective(self.b)+1e-10)

    def test_degenerate_new_a_is_finite(self):
        new_b = transport_private_b(self.a,torch.zeros_like(self.a),self.b)
        torch.testing.assert_close(new_b,self.b)

    def test_partial_interpolates_and_schedule_keeps_budget(self):
        new_a = torch.randn_like(self.a)
        full = transport_private_b(self.a,new_a,self.b)
        torch.testing.assert_close(transport_private_b(self.a,new_a,self.b,.5),.5*(self.b+full))
        self.assertEqual(suffix_layers(12,6),list(range(6,12)))
        self.assertEqual([r for r in range(1,10) if shared_a_update_round('slow_a',r)],[1,2,3,6,9])


if __name__ == '__main__':
    unittest.main()
