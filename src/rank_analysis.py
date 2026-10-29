"""
Rank Analysis Module
Performs matrix rank determination, condition number analysis, singular value
spectrum computation, and energy concentration evaluation for image matrices.
"""

from typing import Dict, Any, List
import numpy as np

class RankAnalyzer:
    """
    Analyzes mathematical rank properties of 2D image matrices.
    """
    def __init__(self, tolerance: float = None):
        self.tolerance = tolerance

    def analyze_matrix(self, A: np.ndarray, name: str = "Image") -> Dict[str, Any]:
        """
        Analyzes a single image matrix A in R^{m x n}.
        Computes:
            - Singular values (spectrum)
            - Exact / Numerical rank (using machine precision threshold)
            - Condition number
            - Roy-Vetterli effective rank (spectral entropy)
            - Ranks required for 90%, 95%, 99% energy capture
        """
        m, n = A.shape
        # Compute SVD singular values
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        sigma_max = S[0] if len(S) > 0 and S[0] > 0 else 1.0

        # Numerical tolerance for rank
        eps = np.finfo(A.dtype).eps
        tol = self.tolerance if self.tolerance is not None else eps * max(m, n) * sigma_max

        numerical_rank = int(np.sum(S > tol))
        
        # Condition number: sigma_max / sigma_min (for non-zero singular values)
        non_zero_s = S[S > tol]
        cond_num = float(sigma_max / non_zero_s[-1]) if len(non_zero_s) > 0 else np.inf

        # Total energy (Frobenius norm squared = sum of singular values squared)
        energy_spectrum = S**2
        total_energy = np.sum(energy_spectrum)
        cumulative_energy = np.cumsum(energy_spectrum) / (total_energy + 1e-15)

        # Ranks needed for energy thresholds
        k_90 = int(np.searchsorted(cumulative_energy, 0.90) + 1)
        k_95 = int(np.searchsorted(cumulative_energy, 0.95) + 1)
        k_99 = int(np.searchsorted(cumulative_energy, 0.99) + 1)

        # Spectral entropy and effective rank (Roy & Vetterli, 2007)
        p = S / (np.sum(S) + 1e-15)
        p = p[p > 0]
        entropy = -np.sum(p * np.log(p))
        erank = float(np.exp(entropy))

        return {
            "name": name,
            "shape": (m, n),
            "max_possible_rank": min(m, n),
            "numerical_rank": numerical_rank,
            "condition_number": cond_num,
            "effective_rank": erank,
            "k_for_90_energy": min(k_90, min(m, n)),
            "k_for_95_energy": min(k_95, min(m, n)),
            "k_for_99_energy": min(k_99, min(m, n)),
            "singular_values": S,
            "cumulative_energy": cumulative_energy,
            "total_frobenius_norm": float(np.sqrt(total_energy)),
            "tolerance_used": float(tol)
        }

    def analyze_collection(self, matrices: List[np.ndarray], names: List[str]) -> List[Dict[str, Any]]:
        """
        Analyzes a full collection of image matrices.
        """
        results = []
        for mat, name in zip(matrices, names):
            res = self.analyze_matrix(mat, name)
            results.append(res)
        return results
