"""
Unsupervised Evaluation Script for DINOv3 Fine-tuned Models
============================================================
"""

import torch
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import argparse
import warnings
from plotly import express as px

# Suppress annoying terminal warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="umap")
warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden.*")
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️  umap-learn not installed. UMAP plots will be skipped.")

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  plotly not installed. 3D UMAP will fall back to static PNG.")


# ============================================================================
# 1. MODEL LOADING
# ============================================================================

def load_model(model_id: str, model_path=None, device="cuda"):
    """Loads either the base DINOv3 model or a fine-tuned checkpoint."""
    print(f"\nLoading base model: {model_id}")
    try:
        model = AutoModel.from_pretrained(model_id)
    except Exception as e:
        if "401" in str(e) or "gated" in str(e).lower() or "Unauthorized" in str(e):
            print("\n❌ HF GATED REPO ERROR: DINOv3 requires you to accept their license.")
            print("  1. Go to: https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m")
            print("  2. Accept the user agreement.")
            print("  3. Run `huggingface-cli login` in your terminal and paste your token.\n")
            exit(1)
        else:
            raise e

    if model_path:
        print(f"Loading custom weights from: {model_path}")
        try:
            ckpt = torch.load(model_path, map_location=device)

            # Extract state dict safely
            state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))

            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("_orig_mod.", "")
                # Only strip strictly unnecessary wrappers, LEAVE "model." intact
                if k.startswith("backbone."): k = k.replace("backbone.", "")
                if k.startswith("net."): k = k.replace("net.", "")
                # FIX: Strip the "model." prefix causing the mismatch
                if k.startswith('model.'):
                    k = k[6:] # Removes the first 6 characters ('model.')
                elif k.startswith('student.backbone.'):
                    k = k.replace('student.backbone.', '')
                new_state_dict[k] = v

            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            if missing:
                print(f"  ⚠️  Missing keys  ({len(missing)}): {missing[:3]} ...")
            if unexpected:
                # 14 unexpected keys (dino_head.mlp) is normal and expected for SSL models
                print(f"  ⚠️  Unexpected keys ({len(unexpected)}): {unexpected[:3]} ...")
            print("  ✓ Custom weights loaded successfully.")
        except Exception as e:
            print(f"  ❌ Could not load checkpoint: {e}\n  Falling back to base weights.")
    else:
        print("  Using base (unmodified) model weights.")

    model = model.to(device).eval()
    return model



def load_processor(model_id: str):
    print(f"Loading image processor for: {model_id}")
    return AutoImageProcessor.from_pretrained(model_id)


# ============================================================================
# 2. FEATURE EXTRACTION
# ============================================================================

@torch.no_grad()
def extract_features(model, image_paths, processor, device, batch_size=32):
    model.eval()
    all_features = []
    valid_paths =[]

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        current_valid =[]

        for path in batch_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
                current_valid.append(path)
            except Exception as e:
                print(f"  ⚠️  Skipping corrupt/unreadable image {path}: {e}")

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt").to(device)
        outputs = model(**inputs)

        cls_features = outputs.last_hidden_state[:, 0, :]
        all_features.append(cls_features.cpu().float().numpy())
        valid_paths.extend(current_valid)

    return np.vstack(all_features), valid_paths


# ============================================================================
# 3. NEAREST NEIGHBOURS EVALUATION
# ============================================================================

@torch.no_grad()
def get_nearest_neighbors(query_path, feature_bank, valid_paths, model, processor, device, k=5):
    query_img = Image.open(query_path).convert("RGB")
    inputs = processor(images=[query_img], return_tensors="pt").to(device)
    query_feat = model(**inputs).last_hidden_state[:, 0, :].cpu().float().numpy()

    sims = cosine_similarity(query_feat, feature_bank).flatten()
    top_k_indices = np.argsort(sims)[::-1]

    neighbors =[]
    for idx in top_k_indices:
        if Path(valid_paths[idx]) != Path(query_path):
            neighbors.append((valid_paths[idx], sims[idx]))
        if len(neighbors) == k:
            break
    return neighbors


def plot_neighbors(query_path, neighbor_data, title, output_dir):
    k = len(neighbor_data)
    fig, axes = plt.subplots(1, k + 1, figsize=(4 * (k + 1), 5))

    axes[0].imshow(Image.open(query_path))
    axes[0].set_title("Query", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    for i, (path, sim) in enumerate(neighbor_data):
        axes[i + 1].imshow(Image.open(path))
        axes[i + 1].set_title(f"#{i+1}  sim={sim:.3f}", fontsize=10)
        axes[i + 1].axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    filename = Path(output_dir) / f"{title.lower().replace(' ', '_')}_neighbors.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {filename.name}")
    plt.close()


# ============================================================================
# 4. COLOUR & PLOT HELPERS
# ============================================================================

def _build_colour_map(unique_labels):
    cmap = plt.get_cmap("tab20")
    colours = {}
    colour_idx = 0
    for label in sorted(unique_labels):
        if label == "Other Class":
            colours[label] = ("lightgray", 0.25)
        else:
            colours[label] = (cmap(colour_idx % 20), 0.75)
            colour_idx += 1
    return colours


def plot_scatter_2d(ax, features_2d, labels, title):
    unique_labels = sorted(set(labels))
    if "Other Class" in unique_labels:
        unique_labels.remove("Other Class")
        unique_labels.append("Other Class")

    colours = _build_colour_map(unique_labels)

    for label in unique_labels:
        idx =[j for j, l in enumerate(labels) if l == label]
        colour, alpha = colours[label]
        zorder = 1 if label == "Other Class" else 2
        ax.scatter(
            features_2d[idx, 0], features_2d[idx, 1],
            label=label, color=colour, alpha=alpha, s=14, zorder=zorder,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, markerscale=2)


def visualize_tsne(features, labels, title, output_dir="."):
    print(f"  Running t-SNE …")
    tsne = TSNE(n_components=2, random_state=0, perplexity=30, max_iter=1000)
    features_norm = sk_normalize(features, norm="l2")
    features_2d = tsne.fit_transform(features_norm)

    fig, ax = plt.subplots(figsize=(12, 9))
    plot_scatter_2d(ax, features_2d, labels, f"t-SNE — {title}")
    ax.set_xlabel("t-SNE Component 1"); ax.set_ylabel("t-SNE Component 2")

    filename = Path(output_dir) / f"{title.lower().replace(' ', '_')}_tsne.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {filename.name}")
    plt.close()


def visualize_umap(features, labels, title, output_dir="."):
    if not UMAP_AVAILABLE: return
    print(f"  Running 2-D UMAP (cosine metric) …")

    features_norm = sk_normalize(features, norm="l2")
    n_neighbors = int(np.clip(len(features) // 100, 15, 50))

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, metric="cosine", random_state=0)
    features_2d = reducer.fit_transform(features_norm)

    fig, ax = plt.subplots(figsize=(12, 9))
    plot_scatter_2d(ax, features_2d, labels, f"UMAP — {title}")
    ax.set_xlabel("UMAP Component 1"); ax.set_ylabel("UMAP Component 2")

    filename = Path(output_dir) / f"{title.lower().replace(' ', '_')}_umap.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {filename.name}")
    plt.close()


def visualize_umap_3d(features, labels, title, output_dir="."):
    if not UMAP_AVAILABLE: return
    print(f"  Running 3-D UMAP (cosine metric) …")

    features_norm = sk_normalize(features, norm="l2")
    n_neighbors = int(np.clip(len(features) // 100, 15, 50))

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=3, metric="cosine", random_state=0)
    features_3d = reducer.fit_transform(features_norm)

    if PLOTLY_AVAILABLE:
        fig = px.scatter_3d(
            x=features_3d[:, 0], y=features_3d[:, 1], z=features_3d[:, 2],
            color=labels, title=f"3-D UMAP — {title}", opacity=0.7, height=800
        )
        fig.update_traces(marker=dict(size=3))
        filename = Path(output_dir) / f"{title.lower().replace(' ', '_')}_umap_3d.html"
        fig.write_html(str(filename))
        print(f"  ✓ Saved interactive 3D UMAP: {filename.name}")
    else:
        from mpl_toolkits.mplot3d import Axes3D
        colours = _build_colour_map(sorted(set(labels)))
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        for label in sorted(set(labels)):
            idx = [j for j, l in enumerate(labels) if l == label]
            colour, alpha = colours[label]
            ax.scatter(features_3d[idx, 0], features_3d[idx, 1], features_3d[idx, 2], label=label, color=colour, alpha=alpha, s=8)

        ax.set_title(f"3-D UMAP — {title}")
        ax.legend(loc="upper left", fontsize=8, markerscale=2)
        filename = Path(output_dir) / f"{title.lower().replace(' ', '_')}_umap_3d.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved static 3D UMAP: {filename.name}")
        plt.close()


# ============================================================================
# 5. K-MEANS MAPPING & MAIN
# ============================================================================

def map_clusters_to_folders(cluster_labels, folder_labels):
    unique_clusters = list(set(cluster_labels))

    # We only want to map to actual folders, not the "Other Class" fallback
    target_folders = set(folder_labels)
    if "Other Class" in target_folders:
        target_folders.remove("Other Class")

    # Count occurrences of each folder in each cluster
    counts = {c: {f: 0 for f in set(folder_labels)} for c in unique_clusters}
    for c, f in zip(cluster_labels, folder_labels):
        counts[c][f] += 1

    # Flatten counts into (cluster, folder, count) and sort strongest to weakest
    flat_counts =[]
    for c in unique_clusters:
        for f in target_folders:
            flat_counts.append((c, f, counts[c][f]))

    flat_counts.sort(key=lambda x: x[2], reverse=True)

    mapping = {}
    used_folders = set()

    # Greedily assign the strongest available match
    for c, f, count in flat_counts:
        # If the cluster hasn't been assigned yet AND the folder is still available
        if c not in mapping and f not in used_folders:
            mapping[c] = f
            used_folders.add(f)

    # Any remaining clusters (if n_clusters > folders) get "Other Class"
    for c in unique_clusters:
        if c not in mapping:
            mapping[c] = "Other Class"

    return [mapping[c] for c in cluster_labels], mapping


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Gather images
    image_dir = Path(args.image_dir)
    all_paths = list(image_dir.rglob("*.jpg")) + list(image_dir.rglob("*.png"))
    print(f"Found {len(all_paths)} images in {image_dir}")

    n = min(args.num_images, len(all_paths))
    initial_image_paths = random.sample(all_paths, n)
    processor = load_processor(args.model_id)

    # Pre-select query images
    query_paths = random.sample(initial_image_paths, min(args.num_queries, len(initial_image_paths)))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_to_evaluate = {
        "Custom Model":     args.checkpoint_path,
        "Original DINOv3":  None,
    }

    results_summary = {}

    for model_name, model_path in models_to_evaluate.items():
        print("\n" + "=" * 60)
        print(f"  EVALUATING: {model_name}")
        print("=" * 60)

        model = load_model(args.model_id, model_path, device)

        feature_bank, valid_paths = extract_features(model, initial_image_paths, processor, device, args.batch_size)

        if len(feature_bank) == 0:
            print("⚠️ No images successfully loaded. Skipping model.")
            continue

        folder_labels =[]
        for path in valid_paths:
            folder = path.parent.name
            if args.target_folders and folder not in args.target_folders:
                folder_labels.append("Other Class")
            else:
                folder_labels.append(folder)

        unique_classes = sorted(set(l for l in folder_labels if l != "Other Class"))
        n_clusters = args.n_clusters if args.n_clusters > 0 else max(1, len(unique_classes))

        print("\n  ── Nearest Neighbours ──")
        valid_queries = [q for q in query_paths if q in valid_paths]
        for i, q_path in enumerate(valid_queries):
            neighbors = get_nearest_neighbors(
                q_path, feature_bank, valid_paths, model, processor, device, k=args.k_neighbors
            )
            plot_neighbors(q_path, neighbors, title=f"{model_name} Query {i+1}", output_dir=out_dir)

        print("\n  ── Clustering & Metrics ──")
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        cluster_labels = kmeans.fit_predict(feature_bank)

        ss = min(5000, len(feature_bank))
        score_kmeans  = silhouette_score(feature_bank, cluster_labels, sample_size=ss) if len(set(cluster_labels)) > 1 else 0
        score_folders = silhouette_score(feature_bank, folder_labels, sample_size=ss) if len(set(folder_labels)) > 1 else 0

        mapped_kmeans_labels, cluster_mapping = map_clusters_to_folders(cluster_labels, folder_labels)

        sep = "─" * 52
        print(f"\n{sep}")
        print(f"  METRICS: {model_name}")
        print(sep)
        print(f"  K-Means Silhouette Score  : {score_kmeans:.4f}")
        print(f"  Folder  Silhouette Score  : {score_folders:.4f}")
        print(sep)

        print("\n  K-Means → Folder mapping:")
        for k_idx, assigned in sorted(cluster_mapping.items()):
            print(f"    Cluster {k_idx:>2}  →  {assigned}")

        results_summary[model_name] = {"kmeans": score_kmeans, "folder": score_folders}

        if args.save_features:
            prefix = str(out_dir / model_name.lower().replace(" ", "_"))
            np.save(f"{prefix}_features.npy", feature_bank)
            np.save(f"{prefix}_folder_labels.npy", np.array(folder_labels))
            np.save(f"{prefix}_kmeans_labels.npy", np.array(mapped_kmeans_labels))

        print("\n  ── Generating Plots ──")
        visualize_tsne(feature_bank, folder_labels, f"{model_name} GT", out_dir)
        visualize_tsne(feature_bank, mapped_kmeans_labels, f"{model_name} KMeans", out_dir)

        if args.use_umap:
            visualize_umap(feature_bank, folder_labels, f"{model_name} GT", out_dir)
            visualize_umap(feature_bank, mapped_kmeans_labels, f"{model_name} KMeans", out_dir)

        if args.use_umap_3d:
            visualize_umap_3d(feature_bank, folder_labels, f"{model_name} GT", out_dir)
            visualize_umap_3d(feature_bank, mapped_kmeans_labels, f"{model_name} KMeans", out_dir)

        del model, feature_bank
        if device == "cuda": torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("  FINAL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<22} {'KMeans Sil':>12} {'Folder Sil':>12}")
    print("  " + "─" * 48)
    for name, scores in results_summary.items():
        print(f"  {name:<22} {scores['kmeans']:>12.4f} {scores['folder']:>12.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DINOv3 fine-tuned models")
    parser.add_argument("--image_dir",       type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--num_images",  type=int, default=5000)
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--num_queries", type=int, default=3)
    parser.add_argument("--k_neighbors", type=int, default=5)
    parser.add_argument("--n_clusters",  type=int, default=0)
    parser.add_argument("--target_folders", type=str, nargs="*", default=None)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--use_umap",    action="store_true")
    parser.add_argument("--use_umap_3d", action="store_true")
    parser.add_argument("--output_dir",  type=str, default="./eval_outputs")
    parser.add_argument("--save_features", action="store_true")

    args = parser.parse_args()
    main(args)
