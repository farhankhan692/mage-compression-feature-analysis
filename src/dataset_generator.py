"""
Dataset Generator for Image Compression and Feature Analysis System.
Generates a diverse set of 16 grayscale images (64x64) with controlled geometric,
facial, and harmonic features to demonstrate SVD compression, rank analysis,
PCA feature extraction, and gradient edge detection.
"""

import os
import numpy as np
from PIL import Image

def generate_synthetic_dataset(output_dir: str, image_size: int = 64) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    images = []
    
    x = np.linspace(-1, 1, image_size)
    y = np.linspace(-1, 1, image_size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)

    # 1. Concentric circles / ripple
    img1 = (np.cos(8 * np.pi * R) * np.exp(-R**2) + 1) / 2
    
    # 2. Diagonal sinusoidal grating
    img2 = (np.sin(6 * np.pi * (X + Y)) + 1) / 2
    
    # 3. Horizontal stripe grating
    img3 = (np.sin(10 * np.pi * Y) + 1) / 2
    
    # 4. Vertical stripe grating
    img4 = (np.sin(10 * np.pi * X) + 1) / 2
    
    # 5. Checkerboard pattern
    img5 = ((np.sign(np.sin(4 * np.pi * X)) * np.sign(np.sin(4 * np.pi * Y)) + 1) / 2)
    
    # 6. Gaussian blob / spot
    img6 = np.exp(-(X**2 + Y**2) / 0.15)
    
    # 7. Twin Gaussian blobs
    img7 = 0.5 * np.exp(-((X - 0.4)**2 + Y**2) / 0.1) + 0.5 * np.exp(-((X + 0.4)**2 + Y**2) / 0.1)
    
    # 8. Spiral pattern
    img8 = (np.sin(5 * Theta + 6 * np.pi * R) + 1) / 2
    
    # 9-16. Synthetic Face-like Archetypes (varying eyes, smile, angle)
    def make_face(eye_y=0.25, eye_span=0.35, smile_curve=0.4, mouth_open=0.05, tilt=0.0):
        # Rotated coordinates
        c, s = np.cos(tilt), np.sin(tilt)
        Xr = c * X - s * Y
        Yr = s * X + c * Y
        
        # Head contour (ellipse)
        head = np.exp(-(Xr**2 / 0.45 + Yr**2 / 0.6))
        
        # Eyes
        eye_left = np.exp(-((Xr + eye_span)**2 / 0.015 + (Yr - eye_y)**2 / 0.01))
        eye_right = np.exp(-((Xr - eye_span)**2 / 0.015 + (Yr - eye_y)**2 / 0.01))
        pupil_left = np.exp(-((Xr + eye_span)**2 / 0.003 + (Yr - eye_y)**2 / 0.003))
        pupil_right = np.exp(-((Xr - eye_span)**2 / 0.003 + (Yr - eye_y)**2 / 0.003))
        
        # Nose
        nose = 0.6 * np.exp(-(Xr**2 / 0.01 + (Yr + 0.05)**2 / 0.05))
        
        # Mouth: parabola curve
        mouth_y_center = -0.35 + smile_curve * (Xr**2)
        mouth = np.exp(-(mouth_y_center - Yr)**2 / (mouth_open**2 + 0.002)) * (np.abs(Xr) < 0.25)
        
        face = 0.7 * head - 0.5 * (eye_left - pupil_left*0.5) - 0.5 * (eye_right - pupil_right*0.5) + nose - 0.6 * mouth
        face = (face - face.min()) / (face.max() - face.min() + 1e-8)
        return face

    face_variants = [
        make_face(smile_curve=0.6, mouth_open=0.03, tilt=0.0),    # Neutral / slight smile
        make_face(smile_curve=1.2, mouth_open=0.08, tilt=0.0),    # Wide smile
        make_face(smile_curve=-0.5, mouth_open=0.02, tilt=0.0),   # Frown
        make_face(smile_curve=0.4, mouth_open=0.04, tilt=0.15),   # Tilted left
        make_face(smile_curve=0.4, mouth_open=0.04, tilt=-0.15),  # Tilted right
        make_face(eye_y=0.32, eye_span=0.4, smile_curve=0.5),     # Wide eyes
        make_face(eye_y=0.20, eye_span=0.28, smile_curve=-0.2),   # Narrow eyes
        make_face(smile_curve=0.0, mouth_open=0.12, tilt=0.0),    # Open mouth surprised
    ]

    all_patterns = [
        ("concentric_circles", img1),
        ("diagonal_grating", img2),
        ("horizontal_stripes", img3),
        ("vertical_stripes", img4),
        ("checkerboard", img5),
        ("gaussian_blob", img6),
        ("twin_blobs", img7),
        ("spiral", img8),
    ] + [(f"face_variant_{i+1:02d}", fv) for i, fv in enumerate(face_variants)]

    saved_paths = []
    for name, arr in all_patterns:
        # Normalize strictly to [0, 255]
        norm = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
        img = Image.fromarray(norm, mode='L')
        fpath = os.path.join(output_dir, f"{name}.png")
        img.save(fpath)
        saved_paths.append(fpath)
        
    return saved_paths

if __name__ == "__main__":
    out = os.path.expanduser("~/.gemini/antigravity/scratch/image_compression_feature_analysis/dataset")
    paths = generate_synthetic_dataset(out)
    print(f"Generated {len(paths)} images in {out}")
