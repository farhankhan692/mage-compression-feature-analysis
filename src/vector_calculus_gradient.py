"""
Vector Calculus & Gradient Module (Module 4)
Treats 2D image matrices as discrete scalar intensity fields I(x, y).
Computes spatial partial derivatives, the gradient vector field nabla I,
gradient magnitude, orientation angle, directional derivatives, Laplacian
(divergence of gradient), and produces edge maps.
Evaluates preservation of vector calculus features across compression stages.
"""

from typing import Tuple, Dict, Any
import numpy as np

class VectorCalculusGradient:
    """
    Computes vector calculus operators on 2D scalar intensity fields.
    """
    def __init__(self, edge_threshold: float = 0.15):
        self.edge_threshold = edge_threshold
        # Standard Sobel convolution kernels
        self.K_x = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=np.float64) / 8.0
        
        self.K_y = np.array([[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]], dtype=np.float64) / 8.0

    def compute_gradient_field(self, I: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Computes the gradient vector field nabla I = (dI/dx, dI/dy)^T,
        gradient magnitude ||nabla I||, and orientation angle theta(x, y).
        
        Mathematical Formulation:
            I: Omega subset R^2 -> [0, 1] (Scalar Field)
            nabla I(x, y) = [ dI/dx, dI/dy ]^T (Gradient Vector Field)
            ||nabla I(x, y)|| = sqrt((dI/dx)^2 + (dI/dy)^2) (Scalar Magnitude Field)
            theta(x, y) = atan2(dI/dy, dI/dx) in [-pi, pi] (Orientation Field)
        """
        # Central difference approximation using convolution
        from scipy.signal import convolve2d
        
        dI_dx = convolve2d(I, self.K_x, mode='same', boundary='symm')
        dI_dy = convolve2d(I, self.K_y, mode='same', boundary='symm')

        magnitude = np.sqrt(dI_dx**2 + dI_dy**2)
        orientation = np.arctan2(dI_dy, dI_dx)

        return {
            "dI_dx": dI_dx,
            "dI_dy": dI_dy,
            "magnitude": magnitude,
            "orientation": orientation
        }

    def compute_directional_derivative(self, I: np.ndarray, unit_vector: Tuple[float, float]) -> np.ndarray:
        """
        Computes the directional derivative D_u I = nabla I . u along unit vector u = (u_x, u_y).
        By Cauchy-Schwarz: |D_u I| <= ||nabla I||, with equality when u is parallel to nabla I.
        """
        u_x, u_y = unit_vector
        norm_u = np.sqrt(u_x**2 + u_y**2)
        if norm_u == 0:
            raise ValueError("Direction vector cannot be zero.")
        u_x, u_y = u_x / norm_u, u_y / norm_u

        grad = self.compute_gradient_field(I)
        D_u = grad["dI_dx"] * u_x + grad["dI_dy"] * u_y
        return D_u

    def compute_laplacian(self, I: np.ndarray) -> np.ndarray:
        """
        Computes the Laplacian scalar field:
            Delta I = div(nabla I) = d^2 I / dx^2 + d^2 I / dy^2
        Represents the flux density of the gradient vector field.
        """
        from scipy.signal import convolve2d
        # Discrete 5-point discrete Laplacian stencil
        laplacian_kernel = np.array([[ 0,  1,  0],
                                     [ 1, -4,  1],
                                     [ 0,  1,  0]], dtype=np.float64)
        return convolve2d(I, laplacian_kernel, mode='same', boundary='symm')

    def generate_edge_map(self, I: np.ndarray, threshold: float = None) -> np.ndarray:
        """
        Extracts binary edge map E(x, y) = 1 if ||nabla I(x, y)|| >= tau else 0.
        """
        tau = threshold if threshold is not None else self.edge_threshold
        grad = self.compute_gradient_field(I)
        edge_map = (grad["magnitude"] >= tau).astype(np.uint8)
        return edge_map

    def evaluate_edge_preservation(self, I_orig: np.ndarray, I_recon: np.ndarray, threshold: float = None) -> Dict[str, Any]:
        """
        Compares gradient vector field and edge maps between original and reconstructed image:
            - Cosine similarity of gradient vectors
            - Gradient magnitude correlation
            - Jaccard Index (Intersection over Union) of edge maps:
              J(E_orig, E_recon) = |E_orig cap E_recon| / |E_orig cup E_recon|
            - Dice similarity coefficient
        """
        tau = threshold if threshold is not None else self.edge_threshold
        grad_orig = self.compute_gradient_field(I_orig)
        grad_recon = self.compute_gradient_field(I_recon)

        mag_orig = grad_orig["magnitude"]
        mag_recon = grad_recon["magnitude"]

        # 1. Gradient Magnitude Correlation (Pearson r)
        cov_mag = np.cov(mag_orig.ravel(), mag_recon.ravel())[0, 1]
        std_orig = np.std(mag_orig)
        std_recon = np.std(mag_recon)
        corr_mag = float(cov_mag / (std_orig * std_recon + 1e-15))

        # 2. Vector Cosine Similarity
        dot_product = grad_orig["dI_dx"] * grad_recon["dI_dx"] + grad_orig["dI_dy"] * grad_recon["dI_dy"]
        denom = (mag_orig * mag_recon) + 1e-15
        cos_sim = np.mean(dot_product / denom)

        # 3. Binary Edge Maps & Jaccard / Dice Metrics
        edge_orig = (mag_orig >= tau)
        edge_recon = (mag_recon >= tau)

        intersection = np.sum(edge_orig & edge_recon)
        union = np.sum(edge_orig | edge_recon)
        jaccard = float(intersection / (union + 1e-15))
        dice = float(2.0 * intersection / (np.sum(edge_orig) + np.sum(edge_recon) + 1e-15))

        return {
            "edge_threshold": tau,
            "original_edge_pixels": int(np.sum(edge_orig)),
            "reconstructed_edge_pixels": int(np.sum(edge_recon)),
            "edge_intersection_pixels": int(intersection),
            "jaccard_iou": jaccard,
            "dice_coefficient": dice,
            "gradient_magnitude_correlation": corr_mag,
            "mean_cosine_similarity": float(cos_sim)
        }
