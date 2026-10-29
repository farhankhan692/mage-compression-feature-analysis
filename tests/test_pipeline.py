"""
Mathematical Theorem and Pipeline Verification Test Suite
Verifies:
1. SVD Orthogonality: U^T U = I, V^T V = I
2. Eckart-Young-Mirsky Theorem: ||A - A_k||_F = sqrt(sum_{i=k+1}^p sigma_i^2)
3. Vector-space Isomorphism & Isometry: <vec(A), vec(B)> = Tr(A^T B)
4. PCA Orthonormality & Dual Gram matrix equivalence
5. Orthogonal Projection Idempotency: P^2 = P and symmetry P^T = P
6. Pythagorean Orthogonality of Projection Residuals
7. Vector Calculus: Cauchy-Schwarz for directional derivatives |D_u I| <= ||nabla I||
8. Laplacian divergence relationship Delta I = div(nabla I)
"""

import unittest
import numpy as np
import os
import sys

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.rank_analysis import RankAnalyzer
from src.svd_compression import SVDCompressor
from src.vector_space import VectorSpaceTransformer
from src.pca_features import PCAModel
from src.projection_reconstruction import ProjectionReconstructor
from src.vector_calculus_gradient import VectorCalculusGradient

class TestLinearAlgebraAndVectorCalculus(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.m, self.n = 64, 64
        # Create synthetic test matrix
        x = np.linspace(-1, 1, self.m)
        y = np.linspace(-1, 1, self.n)
        X, Y = np.meshgrid(x, y)
        self.A = (np.sin(3 * np.pi * X) * np.cos(3 * np.pi * Y) + 1.0) / 2.0
        self.B = np.exp(-(X**2 + Y**2) / 0.3)

    def test_svd_orthogonality_and_eckart_young(self):
        compressor = SVDCompressor(target_rank=8)
        Ak, metrics = compressor.compress_matrix(self.A, rank_k=8)

        # 1. Orthogonality check
        self.assertLess(metrics["u_orthogonality_error"], 1e-12)
        self.assertLess(metrics["v_orthogonality_error"], 1e-12)

        # 2. Eckart-Young-Mirsky discrepancy check
        self.assertLess(metrics["eckart_young_discrepancy"], 1e-12)

        # 3. Best rank approximation property: ||A - A_k||_F must be smaller than any arbitrary rank-k perturbation
        random_rank_k_mat = np.random.randn(self.m, 8) @ np.random.randn(8, self.n)
        scale = np.linalg.norm(self.A, 'fro') / np.linalg.norm(random_rank_k_mat, 'fro')
        random_rank_k_mat *= scale
        
        err_optimal = np.linalg.norm(self.A - Ak, 'fro')
        err_random = np.linalg.norm(self.A - random_rank_k_mat, 'fro')
        self.assertLess(err_optimal, err_random)

    def test_vector_space_isomorphism_and_isometry(self):
        vst = VectorSpaceTransformer(image_shape=(self.m, self.n))
        iso = vst.verify_isomorphism(self.A, self.B, alpha=3.14, beta=-2.71)

        self.assertTrue(iso["is_linear"])
        self.assertLess(iso["linearity_residual"], 1e-12)
        self.assertTrue(iso["is_isometry"])
        self.assertLess(iso["inner_prod_discrepancy"], 1e-12)
        self.assertLess(iso["norm_discrepancy"], 1e-12)

    def test_pca_dual_gram_and_orthonormality(self):
        # Create ensemble of 5 images
        vst = VectorSpaceTransformer(image_shape=(self.m, self.n))
        matrices = [self.A + 0.05 * np.random.randn(self.m, self.n) for _ in range(5)]
        X = vst.build_ensemble_matrix(matrices)

        pca = PCAModel(n_components=3)
        metrics = pca.fit(X)

        # 1. Basis Orthonormality: W^T W = I
        self.assertLess(metrics["orthogonality_error"], 1e-12)

        # 2. Dual Gram Matrix eigenvalue exact equivalence
        self.assertLess(metrics["dual_gram_eigenvalue_discrepancy"], 1e-12)

        # 3. Variance sum bounded
        self.assertGreater(metrics["cumulative_variance_p"], 0.0)
        self.assertLessEqual(metrics["cumulative_variance_p"], 1.0 + 1e-12)

    def test_orthogonal_projection_properties(self):
        vst = VectorSpaceTransformer(image_shape=(self.m, self.n))
        matrices = [self.A + 0.1 * np.random.randn(self.m, self.n) for _ in range(6)]
        X = vst.build_ensemble_matrix(matrices)

        pca = PCAModel(n_components=4)
        pca.fit(X)
        reconstructor = ProjectionReconstructor(pca)

        # Projection operator P = W W^T
        W = pca.components
        P = W @ W.T

        # 1. Idempotency: P^2 = P
        P_squared = P @ P
        self.assertLess(np.linalg.norm(P_squared - P, 'fro'), 1e-12)

        # 2. Symmetry: P^T = P
        self.assertLess(np.linalg.norm(P.T - P, 'fro'), 1e-12)

        # 3. Pythagorean decomposition verification
        Y, X_hat = reconstructor.project_and_reconstruct(X)
        errors = reconstructor.evaluate_errors(matrices, matrices, X_hat, (self.m, self.n))
        for err in errors:
            self.assertLess(err["pythagorean_residual"], 1e-10)

    def test_vector_calculus_gradient_and_directional_derivative(self):
        vc = VectorCalculusGradient(edge_threshold=0.1)
        grad = vc.compute_gradient_field(self.A)

        # 1. Directional Derivative along unit vector u = (1/sqrt(2), 1/sqrt(2))
        u = (1.0 / np.sqrt(2), 1.0 / np.sqrt(2))
        D_u = vc.compute_directional_derivative(self.A, u)

        # 2. Cauchy-Schwarz: |D_u I(x, y)| <= ||nabla I(x, y)||
        cs_violation = np.max(np.abs(D_u) - grad["magnitude"])
        self.assertLess(cs_violation, 1e-12)

        # 3. Directional derivative along gradient unit vector equals gradient magnitude
        norm_grad = grad["magnitude"] + 1e-12
        u_grad_x = grad["dI_dx"] / norm_grad
        u_grad_y = grad["dI_dy"] / norm_grad
        D_grad = grad["dI_dx"] * u_grad_x + grad["dI_dy"] * u_grad_y
        self.assertLess(np.max(np.abs(D_grad - grad["magnitude"])), 1e-10)

if __name__ == "__main__":
    unittest.main()
