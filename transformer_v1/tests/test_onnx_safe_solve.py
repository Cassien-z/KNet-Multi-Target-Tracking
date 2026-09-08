import unittest

import torch

from src.model import solve_3x3_onnx_safe


class OnnxSafeSolveTests(unittest.TestCase):
    def test_matches_torch_solve_for_regularized_information_matrices(self):
        generator = torch.Generator().manual_seed(20260907)
        base = torch.randn(4, 100, 3, 3, generator=generator)
        matrix = base.transpose(-2, -1) @ base
        matrix = matrix + torch.eye(3).view(1, 1, 3, 3) * 1e-3
        vector = torch.randn(4, 100, 3, generator=generator)

        expected = torch.linalg.solve(matrix, vector.unsqueeze(-1)).squeeze(-1)
        actual = solve_3x3_onnx_safe(matrix, vector)

        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)

    def test_zero_information_case_stays_finite(self):
        matrix = torch.eye(3).view(1, 1, 3, 3) * 1e-6
        vector = torch.zeros(1, 1, 3)
        actual = solve_3x3_onnx_safe(matrix, vector)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))


if __name__ == "__main__":
    unittest.main()
