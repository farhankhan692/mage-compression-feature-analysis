"""
Data Loader Module
Loads image collections, converts them to grayscale, normalizes intensities
to [0, 1], and represents them as mathematical matrices A in R^{m x n}.
"""

import os
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image

class ImageCollectionLoader:
    """
    Handles loading, preprocessing, and matrix conversion of image datasets.
    """
    def __init__(self, target_size: Tuple[int, int] = (64, 64)):
        self.target_size = target_size  # (height m, width n)

    def load_image(self, file_path: str) -> np.ndarray:
        """
        Loads a single image, converts to grayscale, resizes, and normalizes to [0, 1].
        Returns matrix A in R^{m x n} with float64 precision.
        """
        img = Image.open(file_path).convert('L')
        if img.size != (self.target_size[1], self.target_size[0]):
            img = img.resize((self.target_size[1], self.target_size[0]), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float64) / 255.0
        return arr

    def load_directory(self, dir_path: str) -> Tuple[List[np.ndarray], List[str]]:
        """
        Loads all supported images from a directory.
        Returns:
            matrices: List of 2D numpy arrays A_i in R^{m x n}
            filenames: List of file names
        """
        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        files = sorted([
            f for f in os.listdir(dir_path)
            if os.path.splitext(f)[1].lower() in valid_exts
        ])
        if not files:
            raise FileNotFoundError(f"No valid image files found in {dir_path}")

        matrices = []
        filenames = []
        for f in files:
            full_path = os.path.join(dir_path, f)
            try:
                mat = self.load_image(full_path)
                matrices.append(mat)
                filenames.append(f)
            except Exception as e:
                print(f"Warning: Failed to load {f}: {e}")

        return matrices, filenames
