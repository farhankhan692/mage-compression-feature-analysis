"""
Projection, Reconstruction, and Error Analysis Module
Implements orthogonal projection onto the principal subspace, reconstruction
via the projection matrix P = W_p W_p^T, and comprehensive error analysis
(MSE, PSNR, Frobenius residual, Pythagorean orthogonal decomposition).
"""

from typing import Tuple, Dict, Any, List
import numpy as np

class ProjectionReconstructor:
    """
    Handles orthogonal projection, reconstruction, and quantitative error analysis.
    """
    def __init__(self, pca_model):
        self.pca = pca_model

    def project(self, X: np.ndarray) -> np.ndarray:
        """
        Projects centered data matrix (or single vector) onto principal subspace.
        y_i = W_p^T * (x_i - mu) in R^p
        
        Returns:
            Y in R^{p x N}: Coordinate representation in principal subspace
        """
        if self.pca.components is None or self.pca.mean_vector is None:
            raise ValueError("PCA model has not been fitted yet.")
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_centered = X - self.pca.mean_vector
        Y = self.pca.components.T @ X_centered
        return Y

    def reconstruct(self, Y: np.ndarray) -> np.ndarray:
        """
        Reconstructs image vectors from principal coordinates:
        x_hat = mu + W_p * y_i in R^D
        
        Returns:
            X_hat in R^{D x N}: Reconstructed vectors
        """
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        X_hat = self.pca.mean_vector + self.pca.components @ Y
        return X_hat

    def project_and_reconstruct(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies end-to-end projection and reconstruction.
        """
        Y = self.project(X)
        X_hat = self.reconstruct(Y)
        return Y, X_hat

    def evaluate_errors(
        self,
        original_matrices: List[np.ndarray],
        svd_matrices: List[np.ndarray],
        reconstructed_vectors: np.ndarray,
        shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """
        Computes detailed error metrics across all stages:
            - Original -> SVD compressed (Stage 1 error)
            - SVD compressed -> PCA reconstructed (Stage 2 error)
            - Original -> Final reconstructed (End-to-End error)
            - Pythagorean orthogonality residual verification:
              ||x - mu||^2 = ||x_hat - mu||^2 + ||x - x_hat||^2
        """
        m, n = shape
        D = m * n
        N = len(original_matrices)
        results = []

        for i in range(N):
            A_orig = original_matrices[i]
            A_svd = svd_matrices[i]
            
            x_hat_vec = reconstructed_vectors[:, i]
            A_final = np.clip(x_hat_vec.reshape((m, n)), 0.0, 1.0)
            x_svd_vec = A_svd.flatten()
            x_orig_vec = A_orig.flatten()

            # 1. Stage 1 Error: Original vs SVD
            diff_svd = A_orig - A_svd
            fro_err_svd = float(np.linalg.norm(diff_svd, 'fro'))
            mse_svd = float(np.mean(diff_svd**2))
            psnr_svd = float(10.0 * np.log10(1.0 / (mse_svd + 1e-15)))

            # 2. Stage 2 Error: SVD vs Final PCA Reconstruction
            diff_pca = A_svd - A_final
            fro_err_pca = float(np.linalg.norm(diff_pca, 'fro'))
            mse_pca = float(np.mean(diff_pca**2))
            psnr_pca = float(10.0 * np.log10(1.0 / (mse_pca + 1e-15)))

            # 3. End-to-End Error: Original vs Final PCA Reconstruction
            diff_total = A_orig - A_final
            fro_err_total = float(np.linalg.norm(diff_total, 'fro'))
            norm_orig = float(np.linalg.norm(A_orig, 'fro'))
            rel_fro_error = fro_err_total / (norm_orig + 1e-15)
            mse_total = float(np.mean(diff_total**2))
            psnr_total = float(10.0 * np.log10(1.0 / (mse_total + 1e-15)))

            # 4. Pythagorean Orthogonality Verification for PCA Subspace
            # ||x_svd - mu||^2 == ||x_hat - mu||^2 + ||x_svd - x_hat||^2
            mu = self.pca.mean_vector.ravel()
            total_dev = np.linalg.norm(x_svd_vec - mu)**2
            proj_dev = np.linalg.norm(x_hat_vec - mu)**2
            ortho_dev = np.linalg.norm(x_svd_vec - x_hat_vec)**2
            pythagorean_discrepancy = abs(total_dev - (proj_dev + ortho_dev))

            results.append({
                "image_idx": i,
                "reconstructed_matrix": A_final,
                "svd_fro_error": fro_err_svd,
                "svd_mse": mse_svd,
                "svd_psnr_db": psnr_svd,
                "pca_fro_error": fro_err_pca,
                "pca_mse": mse_pca,
                "pca_psnr_db": psnr_pca,
                "total_fro_error": fro_err_total,
                "total_relative_error": rel_fro_error,
                "total_mse": mse_total,
                "total_psnr_db": psnr_total,
                "pythagorean_residual": float(pythagorean_discrepancy)
            })

        return results
