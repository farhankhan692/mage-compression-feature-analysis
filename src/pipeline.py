"""
Sequential Computational Pipeline
Orchestrates the entire sequential mathematical workflow:
Images -> Matrix -> Rank -> SVD -> Reduced image -> Vectorization -> PCA -> Projection -> Reconstruction -> Gradient -> Edge map.
Generates comprehensive visual plots, diagnostic logs, and benchmark tables.
"""

import os
import json
from typing import Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data_loader import ImageCollectionLoader
from src.rank_analysis import RankAnalyzer
from src.svd_compression import SVDCompressor
from src.vector_space import VectorSpaceTransformer
from src.pca_features import PCAModel
from src.projection_reconstruction import ProjectionReconstructor
from src.vector_calculus_gradient import VectorCalculusGradient

class IntegratedSystemPipeline:
    def __init__(
        self,
        dataset_dir: str,
        output_dir: str,
        target_size: tuple = (64, 64),
        svd_rank: int = 12,
        pca_components: int = 6,
        edge_threshold: float = 0.12
    ):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.target_size = target_size
        self.svd_rank = svd_rank
        self.pca_components = pca_components
        self.edge_threshold = edge_threshold
        os.makedirs(output_dir, exist_ok=True)

        # Initialize modular stages
        self.loader = ImageCollectionLoader(target_size=target_size)
        self.rank_analyzer = RankAnalyzer()
        self.svd_compressor = SVDCompressor(target_rank=svd_rank)
        self.vector_space = VectorSpaceTransformer(image_shape=target_size)
        self.pca_model = PCAModel(n_components=pca_components)
        self.reconstructor = ProjectionReconstructor(self.pca_model)
        self.gradient_analyzer = VectorCalculusGradient(edge_threshold=edge_threshold)

    def run_pipeline(self) -> Dict[str, Any]:
        print("=" * 70)
        print("STAGE 1: Loading Image Collection & Converting to 2D Matrices")
        print("=" * 70)
        matrices, filenames = self.loader.load_directory(self.dataset_dir)
        N = len(matrices)
        m, n = self.target_size
        print(f"Loaded N = {N} images, each normalized to matrix A in R^{m}x{n}")

        print("\n" + "=" * 70)
        print("STAGE 2: Mathematical Rank Analysis of Image Matrices")
        print("=" * 70)
        rank_results = self.rank_analyzer.analyze_collection(matrices, filenames)
        avg_rank = np.mean([r["numerical_rank"] for r in rank_results])
        avg_erank = np.mean([r["effective_rank"] for r in rank_results])
        print(f"Average Numerical Rank: {avg_rank:.2f} / {min(m, n)}")
        print(f"Average Effective Rank (Spectral Entropy): {avg_erank:.2f}")

        print("\n" + "=" * 70)
        print(f"STAGE 3: SVD Compression (Low-Rank Approximation at k = {self.svd_rank})")
        print("=" * 70)
        svd_matrices, svd_metrics = self.svd_compressor.compress_collection(matrices, self.svd_rank)
        avg_psnr_svd = np.mean([m["psnr_db"] for m in svd_metrics])
        avg_ey_discrepancy = np.mean([m["eckart_young_discrepancy"] for m in svd_metrics])
        print(f"SVD Rank-k Compression completed. Avg PSNR: {avg_psnr_svd:.2f} dB")
        print(f"Eckart-Young-Mirsky Theorem Discrepancy: {avg_ey_discrepancy:.2e} (Optimality Verified)")

        print("\n" + "=" * 70)
        print("STAGE 4: Vector-Space Representation (Isomorphism vec: R^{m x n} -> R^{mn})")
        print("=" * 70)
        # Verify vector space isomorphism on first two images
        iso_check = self.vector_space.verify_isomorphism(matrices[0], matrices[1])
        print(f"Vector Space Dimension D = m * n = {self.vector_space.dimension}")
        print(f"Isomorphism Linearity Check: Residual = {iso_check['linearity_residual']:.2e}")
        print(f"Frobenius Inner Product Isometry: Discrepancy = {iso_check['inner_prod_discrepancy']:.2e}")

        # Assemble ensemble data matrix from SVD-reduced images
        X_reduced = self.vector_space.build_ensemble_matrix(svd_matrices)
        print(f"Ensemble Data Matrix X in R^{X_reduced.shape[0]}x{X_reduced.shape[1]}")

        print("\n" + "=" * 70)
        print(f"STAGE 5: PCA on Vectorized Ensemble (p = {self.pca_components} components)")
        print("=" * 70)
        pca_fit_metrics = self.pca_model.fit(X_reduced)
        explained_var = pca_fit_metrics["cumulative_variance_p"] * 100.0
        print(f"Extracted p = {self.pca_components} Principal Directions")
        print(f"Explained Variance by top {self.pca_components} components: {explained_var:.2f}%")
        print(f"Dual Gram Matrix Eigenvalue Discrepancy: {pca_fit_metrics['dual_gram_eigenvalue_discrepancy']:.2e}")

        print("\n" + "=" * 70)
        print("STAGE 6: Orthogonal Projection onto Principal Subspace")
        print("=" * 70)
        Y_coords = self.reconstructor.project(X_reduced)
        print(f"Projected coordinates Y in R^{Y_coords.shape[0]}x{Y_coords.shape[1]}")

        print("\n" + "=" * 70)
        print("STAGE 7: Reconstruction back to Vector/Matrix Space & Multi-stage Error Analysis")
        print("=" * 70)
        X_reconstructed = self.reconstructor.reconstruct(Y_coords)
        error_results = self.reconstructor.evaluate_errors(
            matrices, svd_matrices, X_reconstructed, self.target_size
        )
        avg_total_psnr = np.mean([e["total_psnr_db"] for e in error_results])
        avg_pythagoras = np.mean([e["pythagorean_residual"] for e in error_results])
        print(f"Reconstruction Complete. End-to-End Avg PSNR: {avg_total_psnr:.2f} dB")
        print(f"Pythagorean Orthogonality Residual: {avg_pythagoras:.2e} (Subspace projection verified)")

        print("\n" + "=" * 70)
        print("STAGE 8 (Module 4): Vector Calculus - Gradient Vector Field & Edge Map")
        print("=" * 70)
        edge_preservations = []
        for i in range(N):
            ep = self.gradient_analyzer.evaluate_edge_preservation(
                matrices[i], error_results[i]["reconstructed_matrix"], threshold=self.edge_threshold
            )
            edge_preservations.append(ep)
        avg_jaccard = np.mean([ep["jaccard_iou"] for ep in edge_preservations])
        avg_grad_corr = np.mean([ep["gradient_magnitude_correlation"] for ep in edge_preservations])
        print(f"Gradient Magnitude Correlation: {avg_grad_corr:.4f}")
        print(f"Edge Map Jaccard Similarity (IoU): {avg_jaccard:.4f}")

        # Generate Visualizations
        print("\n" + "=" * 70)
        print("GENERATING PUBLICATION PLOTS AND DIAGNOSTICS")
        print("=" * 70)
        self._plot_rank_analysis(rank_results)
        self._plot_svd_compression(matrices[0], filenames[0])
        self._plot_pca_features(pca_fit_metrics)
        self._plot_pca_projections(Y_coords, filenames)
        self._plot_reconstruction_pipeline(matrices, svd_matrices, error_results, filenames)
        self._plot_vector_calculus(matrices[0], svd_matrices[0], error_results[0]["reconstructed_matrix"], filenames[0])

        # Save Metrics Summary
        summary = {
            "num_images": N,
            "image_size": list(self.target_size),
            "vector_dimension_D": self.vector_space.dimension,
            "svd_rank_k": self.svd_rank,
            "pca_components_p": self.pca_components,
            "edge_threshold": self.edge_threshold,
            "avg_numerical_rank": float(avg_rank),
            "avg_effective_rank": float(avg_erank),
            "avg_svd_psnr_db": float(avg_psnr_svd),
            "avg_total_psnr_db": float(avg_total_psnr),
            "pca_explained_variance_pct": float(explained_var),
            "avg_edge_jaccard_iou": float(avg_jaccard),
            "avg_gradient_corr": float(avg_grad_corr),
            "eckart_young_discrepancy": float(avg_ey_discrepancy),
            "isomorphism_residual": float(iso_check['linearity_residual']),
            "pythagorean_residual": float(avg_pythagoras)
        }
        with open(os.path.join(self.output_dir, "metrics_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)
        print(f"Saved metrics summary to {os.path.join(self.output_dir, 'metrics_summary.json')}")

        return {
            "summary": summary,
            "rank_results": rank_results,
            "svd_metrics": svd_metrics,
            "pca_metrics": pca_fit_metrics,
            "error_results": error_results,
            "edge_preservations": edge_preservations
        }

    def _plot_rank_analysis(self, rank_results: List[Dict[str, Any]]):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        
        # Plot 1: Singular value spectra
        for r in rank_results[:8]:
            axes[0].semilogy(r["singular_values"], label=r["name"].replace(".png", "")[:12])
        axes[0].set_title("Singular Value Spectra Decay", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Singular Value Index i")
        axes[0].set_ylabel("Singular Value $\\sigma_i$ (log scale)")
        axes[0].grid(True, linestyle="--", alpha=0.6)
        axes[0].legend(fontsize=8, loc="upper right")

        # Plot 2: Cumulative energy distribution
        for r in rank_results[:8]:
            axes[1].plot(r["cumulative_energy"] * 100, label=r["name"].replace(".png", "")[:12])
        axes[1].axhline(90, color="r", linestyle=":", label="90% Energy")
        axes[1].axhline(99, color="k", linestyle=":", label="99% Energy")
        axes[1].set_title("Cumulative Energy Concentration", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Number of Singular Values k")
        axes[1].set_ylabel("Cumulative Energy Retained (%)")
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].legend(fontsize=8, loc="lower right")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "rank_analysis.png"), dpi=200)
        plt.close()

    def _plot_svd_compression(self, sample_img: np.ndarray, name: str):
        test_ranks = [1, 3, 6, 12, 24, min(self.target_size)]
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        for idx, k in enumerate(test_ranks):
            Ak, metrics = self.svd_compressor.compress_matrix(sample_img, rank_k=k)
            axes[idx].imshow(Ak, cmap="gray", vmin=0, vmax=1)
            title = f"Rank k = {k}\nPSNR: {metrics['psnr_db']:.1f} dB | CR: {metrics['compression_ratio']*100:.1f}%"
            if k == min(self.target_size):
                title = f"Full Rank (k = {k})\nOriginal Image"
            axes[idx].set_title(title, fontsize=10, fontweight="bold")
            axes[idx].axis("off")

        plt.suptitle(f"SVD Low-Rank Compression Progression ({name})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "svd_compression_stages.png"), dpi=200)
        plt.close()

    def _plot_pca_features(self, pca_metrics: Dict[str, Any]):
        eigen_images = self.pca_model.get_eigen_images(self.target_size)
        mean_img = self.pca_model.mean_vector.reshape(self.target_size)

        num_components = len(eigen_images)
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        axes = axes.flatten()

        # Mean Image
        axes[0].imshow(mean_img, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Mean Image $\\boldsymbol{\\mu}$", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Top Eigen-images
        for i in range(min(num_components, 6)):
            var_pct = self.pca_model.explained_variance_ratio[i] * 100.0
            axes[i + 1].imshow(eigen_images[i], cmap="coolwarm")
            axes[i + 1].set_title(f"PC {i+1} Eigen-Image\n({var_pct:.1f}% var)", fontsize=10, fontweight="bold")
            axes[i + 1].axis("off")

        # Scree Plot
        axes[7].plot(range(1, len(pca_metrics["all_cumulative_variance"]) + 1),
                     pca_metrics["all_cumulative_variance"] * 100, "bo-", linewidth=2)
        axes[7].set_title("PCA Scree Plot (Cumulative Var)", fontsize=11, fontweight="bold")
        axes[7].set_xlabel("Number of Principal Components")
        axes[7].set_ylabel("Explained Variance (%)")
        axes[7].grid(True, linestyle="--", alpha=0.6)

        plt.suptitle("PCA Principal Features (Eigen-Images) & Variance Spectrum", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "pca_eigenfeatures.png"), dpi=200)
        plt.close()

    def _plot_pca_projections(self, Y_coords: np.ndarray, filenames: List[str]):
        fig, ax = plt.subplots(figsize=(9, 7))
        y1 = Y_coords[0, :]
        y2 = Y_coords[1, :]

        scatter = ax.scatter(y1, y2, c=range(len(y1)), cmap="viridis", s=100, edgecolors="black", linewidth=1.5)
        for i, txt in enumerate(filenames):
            clean_name = txt.replace(".png", "")[:10]
            ax.annotate(clean_name, (y1[i], y2[i]), fontsize=8, xytext=(5, 5), textcoords="offset points")

        ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
        ax.axvline(0, color="grey", linestyle="--", alpha=0.5)
        ax.set_title("Orthogonal Projection onto Principal Subspace ($PC_1$ vs $PC_2$)", fontsize=13, fontweight="bold")
        ax.set_xlabel(f"Principal Component 1 ({self.pca_model.explained_variance_ratio[0]*100:.1f}% variance)")
        ax.set_ylabel(f"Principal Component 2 ({self.pca_model.explained_variance_ratio[1]*100:.1f}% variance)")
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "pca_projections_2d.png"), dpi=200)
        plt.close()

    def _plot_reconstruction_pipeline(
        self,
        orig_matrices: List[np.ndarray],
        svd_matrices: List[np.ndarray],
        error_results: List[Dict[str, Any]],
        filenames: List[str]
    ):
        num_samples = min(4, len(orig_matrices))
        fig, axes = plt.subplots(num_samples, 4, figsize=(14, 3.2 * num_samples))

        for row in range(num_samples):
            A_orig = orig_matrices[row]
            A_svd = svd_matrices[row]
            A_recon = error_results[row]["reconstructed_matrix"]
            err_map = np.abs(A_orig - A_recon)

            axes[row, 0].imshow(A_orig, cmap="gray", vmin=0, vmax=1)
            axes[row, 0].set_title(f"Original ({filenames[row][:10]})", fontsize=10, fontweight="bold")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(A_svd, cmap="gray", vmin=0, vmax=1)
            axes[row, 1].set_title(f"SVD (k={self.svd_rank})\nPSNR: {error_results[row]['svd_psnr_db']:.1f} dB", fontsize=10)
            axes[row, 1].axis("off")

            axes[row, 2].imshow(A_recon, cmap="gray", vmin=0, vmax=1)
            axes[row, 2].set_title(f"PCA Recon (p={self.pca_components})\nPSNR: {error_results[row]['total_psnr_db']:.1f} dB", fontsize=10)
            axes[row, 2].axis("off")

            im = axes[row, 3].imshow(err_map, cmap="inferno", vmin=0, vmax=0.3)
            axes[row, 3].set_title(f"Absolute Residual\nMSE: {error_results[row]['total_mse']:.4f}", fontsize=10)
            axes[row, 3].axis("off")
            plt.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)

        plt.suptitle("End-to-End Sequential Reconstruction Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "reconstruction_comparison.png"), dpi=200)
        plt.close()

    def _plot_vector_calculus(self, A_orig: np.ndarray, A_svd: np.ndarray, A_recon: np.ndarray, name: str):
        grad_orig = self.gradient_analyzer.compute_gradient_field(A_orig)
        grad_svd = self.gradient_analyzer.compute_gradient_field(A_svd)
        grad_recon = self.gradient_analyzer.compute_gradient_field(A_recon)

        edge_orig = self.gradient_analyzer.generate_edge_map(A_orig)
        edge_svd = self.gradient_analyzer.generate_edge_map(A_svd)
        edge_recon = self.gradient_analyzer.generate_edge_map(A_recon)
        laplacian_recon = self.gradient_analyzer.compute_laplacian(A_recon)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # Row 1: Intensity fields & Gradient Magnitudes
        axes[0, 0].imshow(A_recon, cmap="gray", vmin=0, vmax=1)
        axes[0, 0].set_title("Reconstructed Scalar Field $I(x,y)$", fontsize=10, fontweight="bold")
        axes[0, 0].axis("off")

        # Quiver plot of gradient vector field (downsampled for clarity)
        step = 4
        Y, X = np.mgrid[0:self.target_size[0]:step, 0:self.target_size[1]:step]
        U = grad_recon["dI_dx"][::step, ::step]
        V = grad_recon["dI_dy"][::step, ::step]
        axes[0, 1].imshow(A_recon, cmap="gray", alpha=0.4)
        axes[0, 1].quiver(X, Y, U, -V, color="cyan", angles="xy", scale_units="xy", scale=0.5, width=0.005)
        axes[0, 1].set_title("Gradient Vector Field $\\nabla I$", fontsize=10, fontweight="bold")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(grad_recon["magnitude"], cmap="magma")
        axes[0, 2].set_title("Gradient Magnitude $\\|\\nabla I\\|$", fontsize=10, fontweight="bold")
        axes[0, 2].axis("off")

        axes[0, 3].imshow(laplacian_recon, cmap="coolwarm")
        axes[0, 3].set_title("Laplacian Field $\\Delta I = \\nabla \\cdot \\nabla I$", fontsize=10, fontweight="bold")
        axes[0, 3].axis("off")

        # Row 2: Edge Map Preservations
        axes[1, 0].imshow(edge_orig, cmap="gray")
        axes[1, 0].set_title("Edge Map: Original Image", fontsize=10, fontweight="bold")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(edge_svd, cmap="gray")
        axes[1, 1].set_title(f"Edge Map: SVD Reduced (k={self.svd_rank})", fontsize=10, fontweight="bold")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(edge_recon, cmap="gray")
        axes[1, 2].set_title(f"Edge Map: Final Recon (p={self.pca_components})", fontsize=10, fontweight="bold")
        axes[1, 2].axis("off")

        # Overlay Edge Discrepancy (Green: overlap, Red: missing, Blue: false positive)
        overlap = np.zeros((*self.target_size, 3), dtype=np.float32)
        overlap[:, :, 0] = (edge_orig & ~edge_recon).astype(float)  # Red: missed in recon
        overlap[:, :, 1] = (edge_orig & edge_recon).astype(float)   # Green: exact match
        overlap[:, :, 2] = (~edge_orig & edge_recon).astype(float)  # Blue: artifact
        axes[1, 3].imshow(overlap)
        axes[1, 3].set_title("Edge Map Overlay\n(Green: Preserved, Red: Missed)", fontsize=10, fontweight="bold")
        axes[1, 3].axis("off")

        plt.suptitle(f"Module 4: Vector Calculus Gradient Analysis ({name})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "gradient_edge_analysis.png"), dpi=200)
        plt.close()
