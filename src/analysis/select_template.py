import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.linalg import svd
from scipy.stats import rankdata
from tqdm import tqdm
import albumentations as A
import pandas as pd

def min_max_scale(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max == arr_min:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)

def collect_image_paths(target_dir):
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    image_paths = []
    for root, _, files in os.walk(target_dir):
        for filename in sorted(files):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                image_paths.append(os.path.join(root, filename))
    return sorted(image_paths)

def load_removal_ids(removal_files):
    removal_ids = set()
    for r in removal_files:
        if os.path.exists(r):
            removal_ids.update(pd.read_csv(r)["id"].astype(str).values)
    return removal_ids

def build_resize_transform(width, height):
    return A.Compose([A.Resize(height=height, width=width)])

def load_image_as_vector(filepath, transform):
    img = Image.open(filepath).convert('L')
    img_np = np.array(img)
    img_resized = transform(image=img_np)['image']
    return img_resized.astype(np.float32).flatten()

def main():
    parser = argparse.ArgumentParser(
        description="Rank candidate registration templates using PCA distance and reconstruction error."
    )
    parser.add_argument('--target_dir', type=str, required=True,
                        help="Folder containing the radiographs eligible for template selection.")
    parser.add_argument('--removal', nargs='+', default=[])
    parser.add_argument('--save_dir', type=str, default="./svd_results",
                        help="Path to save SVD results (graphs, eigenvector images)")
    parser.add_argument('--width', type=int, default=300, help="Width to resize")
    parser.add_argument('--height', type=int, default=300, help="Height to resize")
    parser.add_argument('--target_energy', type=float, default=0.9,
                        help="Cumulative energy for manifold recognition (default: 0.9)")
    parser.add_argument('--num_vis', type=int, default=10,
                        help="Number of top principal component vector images to save (default: 10)")
    parser.add_argument('--query_image', type=str, default=None,
                        help="Optional image to check for average proximity")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Loading images from [{args.target_dir}] directory (including subfolders)...")

    all_filepaths = collect_image_paths(args.target_dir)
    print(f"  Number of image files found: {len(all_filepaths)}")

    transform = build_resize_transform(args.width, args.height)

    image_paths = []
    images_data = []

    removal_ids = load_removal_ids(args.removal)

    for filepath in tqdm(all_filepaths, leave=True):
        try:
            if os.path.basename(filepath) in removal_ids:
                continue
            vec = load_image_as_vector(filepath, transform)
            images_data.append(vec)
            rel_path = os.path.relpath(filepath, args.target_dir)
            image_paths.append(rel_path)
        except Exception as e:
            print(f"Warning: Failed to read {filepath} - {e}")

    if not images_data:
        print("Error: No valid images to process.")
        return

    X = np.vstack(images_data)
    N, D = X.shape
    print(f"Data matrix X created: {N} images, {D} dimensions")

    print("Centering data and performing SVD...")
    mu = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mu

    U, S, Vh = svd(X_centered, full_matrices=False)

    variances = S ** 2
    total_variance = np.sum(variances)
    explained_variance_ratio = variances / total_variance
    cumulative_energy = np.cumsum(explained_variance_ratio)

    K = np.searchsorted(cumulative_energy, args.target_energy) + 1
    print(f"Number of dimensions (K) to reach target energy {args.target_energy * 100:.1f}%: {K} / {min(N, D)}")

    print("Saving cumulative energy graph...")
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_energy, marker='', linestyle='-', color='b', linewidth=2)
    plt.axvline(x=K - 1, color='r', linestyle='--', label=f'K={K}')
    plt.axhline(y=args.target_energy, color='r', linestyle='--', label=f'Energy={args.target_energy}')
    plt.title('Cumulative Explained Variance (Energy)')
    plt.xlabel('Principal Component Index')
    plt.ylabel('Cumulative Energy Ratio')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'cumulative_energy.png'), dpi=300)
    plt.close()

    print(f"Saving top {args.num_vis} eigenvector images...")
    for i in range(min(args.num_vis, len(Vh))):
        vector = Vh[i, :]
        img_array = vector.reshape((args.height, args.width))
        scaled_img_array = min_max_scale(img_array) * 255.0
        img = Image.fromarray(scaled_img_array.astype(np.uint8), mode='L')
        img.save(os.path.join(args.save_dir, f"{i + 1:03d}.png"))

    print("Calculating singular-value-scaled PCA distance and reconstruction error...")
    W = Vh[:K, :].T
    Z = np.dot(X_centered, W)

    Z_normalized = Z / S[:K]
    latent_distances = np.linalg.norm(Z_normalized, axis=1)

    X_reconstructed = np.dot(Z, W.T) + mu
    reconstruction_errors = np.linalg.norm(X - X_reconstructed, axis=1)

    rank_latent = rankdata(latent_distances)
    rank_error = rankdata(reconstruction_errors)
    total_scores = rank_latent + rank_error
    ranked_indices = np.argsort(total_scores)

    top_n = min(10, N)
    print("\n==================================================")
    print(f"Most 'average' hand images Top {top_n}")
    print("==================================================")
    for rank in range(top_n):
        idx = ranked_indices[rank]
        print(f"Rank {rank + 1}: {image_paths[idx]}")
        print(f"    - Total Score (Sum of Ranks) : {total_scores[idx]:.1f}")
        print(f"    - Scaled PCA distance : {latent_distances[idx]:.4f} (Rank: {rank_latent[idx]:.0f})")
        print(f"    - Recon Error : {reconstruction_errors[idx]:.4f} (Rank: {rank_error[idx]:.0f})")
        print("-" * 50)

    if args.query_image is not None:
        print("\n==================================================")
        print(f"[Query Image Average Proximity Analysis]")
        print(f"  Path: {args.query_image}")
        print("==================================================")

        if not os.path.isfile(args.query_image):
            print("  Error: File not found.")
        else:
            try:
                q_vec = load_image_as_vector(args.query_image, transform) 
                q_centered = q_vec - mu.squeeze()                          

                q_z = q_centered @ W                                        
                q_z_norm = q_z / S[:K]
                q_latent_dist = float(np.linalg.norm(q_z_norm))

                q_reconstructed = q_z @ W.T + mu.squeeze()                 
                q_recon_error = float(np.linalg.norm(q_vec - q_reconstructed))

                pct_latent = float(np.mean(latent_distances > q_latent_dist)) * 100
                pct_recon  = float(np.mean(reconstruction_errors > q_recon_error)) * 100

                q_rank_latent = float(np.sum(latent_distances < q_latent_dist)) + 1
                q_rank_recon  = float(np.sum(reconstruction_errors < q_recon_error)) + 1
                q_total_score = q_rank_latent + q_rank_recon

                print(f"  Scaled PCA distance       : {q_latent_dist:.4f}")
                print(f"    Rank {q_rank_latent:.0f} out of {N} (closer to average than {pct_latent:.1f}%)")
                print(f"  Reconstruction Error      : {q_recon_error:.4f}")
                print(f"    Rank {q_rank_recon:.0f} out of {N} (closer to average than {pct_recon:.1f}%)")
                print(f"  Total Score (Sum of Ranks)     : {q_total_score:.1f}  (Lower is closer to average)")

                overall_rank = int(np.sum(total_scores < q_total_score)) + 1
                print(f"  Overall Rank                 : {overall_rank} / {N}  "
                      f"({'Top' if overall_rank <= N // 2 else 'Bottom'} {min(overall_rank, N - overall_rank + 1) / N * 100:.1f}%)")
            except Exception as e:
                print(f"  Error: Failed to process query image - {e}")

if __name__ == "__main__":
    main()
