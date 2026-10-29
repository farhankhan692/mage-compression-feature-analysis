"""
Principal Component Analysis (PCA) Module
Computes empirical mean vector, sample covariance matrix, eigen-decomposition,
dual Gram matrix theorem (Sirovich-Kirby / Turk-Pentland method for D >> N),
principal basis vectors (eigen-features), and explained variance spectrum.
"""

from typing import Tuple, Dict, Any, List
import numpy as np

class PCAModel:
    """
    Performs Principal Component Analysis on high-dimensional image vectors.
    """
    def __init__(self, n_components: int = 5):
        self.n_components = n_components
        self.mean_vector: np.ndarray = None       # mu in R^D
        self.components: np.ndarray = None        # W in R^{D x p} (orthonormal columns)
        self.eigenvalues: np.ndarray = None       # lambda_j in R^p
        self.total_variance: float = 0.0
        self.explained_variance_ratio: np.ndarray = None

    def fit(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Fits PCA model on ensemble matrix X in R^{D x N}.
        
        Parameters:
            X: Matrix where each of the N columns is a vectorized image x_i in R^D
            
        Mathematical Formulation:
            1. Empirical Mean Vector:
               mu = (1 / N) * sum_{i=1}^N x_i in R^D
            2. Mean-Centered Data Matrix:
               X_tilde = X - mu * 1_N^T in R^{D x N}
            3. Sample Covariance Matrix:
               C = (1 / (N - 1)) * X_tilde * X_tilde^T in R^{D x D}
            4. Dual Gram Matrix Equivalence (Sirovich-Kirby Method):
               Since D >> N, compute Gram matrix K = (1 / (N - 1)) * X_tilde^T * X_tilde in R^{N x N}.
               If K * v_j = lambda_j * v_j, then C * (X_tilde * v_j) = lambda_j * (X_tilde * v_j).
               Normalized eigenvector: w_j = (X_tilde * v_j) / ||X_tilde * v_j||_2.
        """
        D, N = X.shape
        if N < 2:
            raise ValueError("PCA requires at least N >= 2 image samples.")

        # 1. Compute empirical mean
        self.mean_vector = np.mean(X, axis=1, keepdims=True)  # shape (D, 1)

        # 2. Mean centering
        X_tilde = X - self.mean_vector  # shape (D, N)

        # 3. Dual Method via Economy SVD of scaled centered data
        # Let A_scaled = X_tilde / sqrt(N - 1).
        # SVD: A_scaled = W * S_pca * V^T
        # Then C = A_scaled * A_scaled^T = W * S_pca^2 * W^T
        # So columns of W are eigenvectors of C, and eigenvalues are S_pca^2.
        A_scaled = X_tilde / np.sqrt(N - 1.0)
        
        # Economy SVD: W in R^{D x min(D, N)}, S in R^{min(D, N)}, Vt in R^{min(D, N) x N}
        W, S_pca, Vt = np.linalg.svd(A_scaled, full_matrices=False)

        eigenvalues_all = S_pca**2
        self.total_variance = float(np.sum(eigenvalues_all))

        # Select top p components
        p = min(self.n_components, len(eigenvalues_all))
        self.components = W[:, :p]  # shape (D, p)
        self.eigenvalues = eigenvalues_all[:p]

        if self.total_variance > 0:
            self.explained_variance_ratio = self.eigenvalues / self.total_variance
            cumulative_variance = np.cumsum(eigenvalues_all) / self.total_variance
        else:
            self.explained_variance_ratio = np.zeros(p)
            cumulative_variance = np.zeros(len(eigenvalues_all))

        # 4. Orthogonality check: W_p^T * W_p = I_p
        ortho_err = float(np.linalg.norm(self.components.T @ self.components - np.eye(p), 'fro'))

        # 5. Dual Gram vs Direct Covariance Mathematical Verification
        # Verify on the small N x N Gram matrix
        K = (X_tilde.T @ X_tilde) / (N - 1.0)
        eigvals_K, eigvecs_K = np.linalg.eigh(K)
        # Sort descending
        idx = np.argsort(eigvals_K)[::-1]
        eigvals_K = eigvals_K[idx]
        gram_discrepancy = float(np.max(np.abs(eigvals_K[:len(S_pca)] - eigenvalues_all)))

        return {
            "dimension_D": D,
            "sample_size_N": N,
            "components_extracted": p,
            "eigenvalues": self.eigenvalues,
            "explained_variance_ratio": self.explained_variance_ratio,
            "cumulative_variance_p": float(np.sum(self.explained_variance_ratio)),
            "all_cumulative_variance": cumulative_variance,
            "orthogonality_error": ortho_err,
            "dual_gram_eigenvalue_discrepancy": gram_discrepancy,
            "total_variance": self.total_variance
        }

    def get_eigen_images(self, shape: Tuple[int, int]) -> List[np.ndarray]:
        """
        Reshapes principal component basis vectors w_j in R^D back into
        eigen-images (eigenfaces/eigen-features) in R^{m x n}.
        """
        m, n = shape
        assert m * n == self.mean_vector.shape[0]
        eigen_images = []
        for j in range(self.components.shape[1]):
            w_j = self.components[:, j]
            # Reshape to 2D
            img_j = w_j.reshape((m, n))
            eigen_images.append(img_j)
        return eigen_images
