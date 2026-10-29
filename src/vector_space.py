"""
Vector-Space Representation Module
Establishes the isomorphism between the matrix vector space M_{m x n}(R) and
the Euclidean vector space R^{mn}. Implements vectorization, inner product
preservation (Frobenius isometry), and ensemble matrix assembly.
"""

from typing import List, Tuple, Dict, Any
import numpy as np

class VectorSpaceTransformer:
    """
    Handles linear vector space operations, vectorization isomorphism,
    and ensemble matrix construction.
    """
    def __init__(self, image_shape: Tuple[int, int] = (64, 64)):
        self.m, self.n = image_shape
        self.dimension = self.m * self.n

    def vectorize(self, A: np.ndarray) -> np.ndarray:
        """
        Vectorization operator: vec: R^{m x n} -> R^{mn}.
        Flattens 2D matrix into a 1D column vector.
        """
        assert A.shape == (self.m, self.n), f"Shape mismatch: {A.shape} vs {(self.m, self.n)}"
        return A.flatten()

    def devectorize(self, x: np.ndarray) -> np.ndarray:
        """
        Inverse isomorphism: mat: R^{mn} -> R^{m x n}.
        Reshapes a 1D vector back to 2D matrix.
        """
        assert x.size == self.dimension, f"Vector size mismatch: {x.size} vs {self.dimension}"
        return x.reshape((self.m, self.n))

    def build_ensemble_matrix(self, matrices: List[np.ndarray]) -> np.ndarray:
        """
        Assembles a collection of N images into a data matrix X in R^{D x N},
        where each column is a vectorized image x_i in R^D.
        """
        N = len(matrices)
        X = np.zeros((self.dimension, N), dtype=np.float64)
        for i, A in enumerate(matrices):
            X[:, i] = self.vectorize(A)
        return X

    def verify_isomorphism(self, A: np.ndarray, B: np.ndarray, alpha: float = 2.5, beta: float = -1.2) -> Dict[str, Any]:
        """
        Verifies vector space axioms and Frobenius inner product isometry:
            1. Linearity: vec(alpha*A + beta*B) == alpha*vec(A) + beta*vec(B)
            2. Isometry: <vec(A), vec(B)>_2 == <A, B>_Frobenius = Tr(A^T B)
            3. Norm preservation: ||vec(A)||_2 == ||A||_Frobenius
        """
        vec_A = self.vectorize(A)
        vec_B = self.vectorize(B)

        # 1. Linearity Check
        lhs_lin = self.vectorize(alpha * A + beta * B)
        rhs_lin = alpha * vec_A + beta * vec_B
        linearity_error = float(np.linalg.norm(lhs_lin - rhs_lin))

        # 2. Inner Product Preservation (Frobenius Isometry)
        inner_prod_vec = float(np.dot(vec_A, vec_B))
        inner_prod_mat = float(np.trace(A.T @ B))
        inner_prod_discrepancy = abs(inner_prod_vec - inner_prod_mat)

        # 3. Norm Preservation
        norm_vec = float(np.linalg.norm(vec_A))
        norm_mat = float(np.linalg.norm(A, 'fro'))
        norm_discrepancy = abs(norm_vec - norm_mat)

        return {
            "is_linear": linearity_error < 1e-12,
            "linearity_residual": linearity_error,
            "is_isometry": inner_prod_discrepancy < 1e-12,
            "inner_prod_discrepancy": inner_prod_discrepancy,
            "norm_discrepancy": norm_discrepancy,
            "dimension": self.dimension
        }
