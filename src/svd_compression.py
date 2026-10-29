"""
SVD Compression Module
Applies Singular Value Decomposition (SVD) for low-rank image matrix compression.
Demonstrates the Eckart-Young-Mirsky theorem, computes theoretical vs empirical
Frobenius error, and evaluates compression ratios and PSNR.
"""

from typing import Tuple, Dict, Any, List
import numpy as np

class SVDCompressor:
    """
    Implements SVD decomposition and low-rank approximation for 2D image matrices.
    """
    def __init__(self, target_rank: int = 10):
        self.target_rank = target_rank

    def compress_matrix(self, A: np.ndarray, rank_k: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Decomposes A in R^{m x n} via SVD and reconstructs rank-k approximation A_k.
        
        Mathematical Formulation:
            A = U Sigma V^T = sum_{i=1}^r sigma_i u_i v_i^T
            A_k = sum_{i=1}^k sigma_i u_i v_i^T = U_k Sigma_k V_k^T
            
        Returns:
            A_k: Low-rank approximated matrix (same dimensions m x n)
            metrics: Dictionary of compression metrics and theorem verifications
        """
        m, n = A.shape
        max_rank = min(m, n)
        k = rank_k if rank_k is not None else self.target_rank
        k = max(1, min(k, max_rank))

        # Perform Full SVD: U in R^{m x m}, S in R^{p}, Vt in R^{n x n}
        U, S, Vt = np.linalg.svd(A, full_matrices=False)

        # Truncation to rank k
        Uk = U[:, :k]
        Sk = S[:k]
        Vtk = Vt[:k, :]

        # Reconstruct rank-k approximation
        Ak = Uk @ np.diag(Sk) @ Vtk
        Ak_clipped = np.clip(Ak, 0.0, 1.0)

        # 1. Eckart-Young-Mirsky Theorem Verification:
        # ||A - A_k||_F = sqrt(sum_{i=k+1}^p sigma_i^2)
        tail_singular_values = S[k:]
        theoretical_frobenius_error = float(np.sqrt(np.sum(tail_singular_values**2)))
        empirical_frobenius_error = float(np.linalg.norm(A - Ak, 'fro'))
        frobenius_error_discrepancy = abs(theoretical_frobenius_error - empirical_frobenius_error)

        # 2. Quality and Compression Metrics
        norm_orig = float(np.linalg.norm(A, 'fro'))
        relative_frobenius_error = empirical_frobenius_error / (norm_orig + 1e-15)

        mse = float(np.mean((A - Ak_clipped)**2))
        psnr = float(10.0 * np.log10(1.0 / (mse + 1e-15))) if mse > 0 else float('inf')

        # Storage Analysis
        uncompressed_values = m * n
        compressed_values = k * (m + n + 1)  # U_k (m*k) + Sigma_k (k) + V_k^T (k*n)
        compression_ratio = compressed_values / uncompressed_values
        space_saving_pct = max(0.0, (1.0 - compression_ratio) * 100.0)

        # Orthogonality checks: U_k^T U_k = I_k and V_k V_k^T = I_k
        u_orthogonality_error = float(np.linalg.norm(Uk.T @ Uk - np.eye(k), 'fro'))
        v_orthogonality_error = float(np.linalg.norm(Vtk @ Vtk.T - np.eye(k), 'fro'))

        metrics = {
            "rank_k": k,
            "max_rank": max_rank,
            "uncompressed_values": uncompressed_values,
            "compressed_values": compressed_values,
            "compression_ratio": compression_ratio,
            "space_saving_pct": space_saving_pct,
            "mse": mse,
            "psnr_db": psnr,
            "relative_frobenius_error": relative_frobenius_error,
            "theoretical_fro_error": theoretical_frobenius_error,
            "empirical_fro_error": empirical_frobenius_error,
            "eckart_young_discrepancy": frobenius_error_discrepancy,
            "u_orthogonality_error": u_orthogonality_error,
            "v_orthogonality_error": v_orthogonality_error,
            "singular_values": S,
            "energy_retained_pct": float(np.sum(Sk**2) / (np.sum(S**2) + 1e-15) * 100.0)
        }

        return Ak_clipped, metrics

    def compress_collection(self, matrices: List[np.ndarray], rank_k: int = None) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """
        Compresses an entire collection of image matrices.
        Returns:
            reduced_matrices: List of SVD-compressed matrices A_i^{(k)}
            all_metrics: List of metric dicts for each image
        """
        reduced_matrices = []
        all_metrics = []
        for mat in matrices:
            Ak, m = self.compress_matrix(mat, rank_k)
            reduced_matrices.append(Ak)
            all_metrics.append(m)
        return reduced_matrices, all_metrics
