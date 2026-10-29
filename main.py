"""
Main Entry Point
Integrated Image Compression and Feature Analysis System
Linear Algebra & Vector Calculus Mini Project (#56)
"""

import os
import sys
import argparse

# Ensure src is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.dataset_generator import generate_synthetic_dataset
from src.pipeline import IntegratedSystemPipeline

def main():
    parser = argparse.ArgumentParser(
        description="Integrated Image Compression and Feature Analysis System (Linear Algebra & Vector Calculus)"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.path.join(current_dir, "dataset"),
        help="Directory containing input images (default: dataset/)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(current_dir, "output"),
        help="Directory to save generated plots and metrics (default: output/)"
    )
    parser.add_argument(
        "--svd-rank",
        type=int,
        default=12,
        help="Truncated rank k for SVD compression stage (default: 12)"
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=6,
        help="Number of principal components p for PCA projection (default: 6)"
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.12,
        help="Gradient magnitude threshold for edge map extraction (default: 0.12)"
    )
    parser.add_argument(
        "--generate-dataset",
        action="store_true",
        help="Force regeneration of synthetic test images in dataset-dir"
    )

    args = parser.parse_args()

    # Generate dataset if empty or requested
    if args.generate_dataset or not os.path.exists(args.dataset_dir) or len(os.listdir(args.dataset_dir)) == 0:
        print(f"Generating synthetic dataset in {args.dataset_dir}...")
        generate_synthetic_dataset(args.dataset_dir)

    print("=" * 80)
    print("STARTING INTEGRATED IMAGE COMPRESSION AND FEATURE ANALYSIS SYSTEM")
    print("Sequential Mathematical Process:")
    print("Images -> Matrix -> Rank -> SVD -> Reduced image -> Vectorization -> PCA -> Projection -> Reconstruction")
    print("Module 4 (Vector Calculus Extension): Reconstructed image -> Scalar intensity field -> Gradient -> Edge map")
    print("=" * 80)

    pipeline = IntegratedSystemPipeline(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        target_size=(64, 64),
        svd_rank=args.svd_rank,
        pca_components=args.pca_components,
        edge_threshold=args.edge_threshold
    )

    results = pipeline.run_pipeline()

    print("\n" + "=" * 80)
    print("EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"All artifacts, diagnostic plots, and metrics summary generated in: {args.output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
