"""CPU-only projections and metrics; labels never determine point coordinates."""
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from .common import normalize_features


def cluster(features, labels, requested, seed):
    x = normalize_features(features)
    unique = len(np.unique(x, axis=0))
    known = [v for v in labels if v is not None]
    default = len(set(known)) if known else 8
    k = min(requested or default, len(x), unique)
    if k < 1:
        raise ValueError("n_clusters must be positive")
    ids = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(x) if k > 1 else np.zeros(len(x), dtype=int)
    measured = dict(clusters_requested=requested or default, clusters_effective=int(len(set(ids))), silhouette_cosine=None, adjusted_rand_index=None, normalized_mutual_information=None)
    if 1 < len(set(ids)) < len(x):
        rng = np.random.default_rng(seed)
        subset = rng.choice(len(x), min(2000, len(x)), replace=False)
        if 1 < len(set(ids[subset])) < len(subset):
            measured["silhouette_cosine"] = float(silhouette_score(x[subset], ids[subset], metric="cosine"))
    mask = np.array([v is not None for v in labels])
    if mask.sum() >= 2 and len(set(known)) > 1:
        measured["adjusted_rand_index"] = float(adjusted_rand_score(np.array(labels, dtype=object)[mask].tolist(), ids[mask]))
        measured["normalized_mutual_information"] = float(normalized_mutual_info_score(np.array(labels, dtype=object)[mask].tolist(), ids[mask]))
    return ids, measured


def projection(features, method, dimensions, seed):
    x = normalize_features(features)
    n, width = x.shape
    if dimensions not in (2, 3):
        raise ValueError("Only 2D and 3D views are supported")
    available = min(dimensions, n, width)
    meta = dict(method=method, dimensions=dimensions, seed=seed, samples=n)
    if n < 4 or len(np.unique(x, axis=0)) < 2:
        coords = np.zeros((n, dimensions), dtype=np.float32)
        if len(np.unique(x, axis=0)) > 1:
            coords[:, :available] = PCA(n_components=available, random_state=seed).fit_transform(x)
        return coords, dict(**meta, fallback="PCA/zero coordinates: fewer than 4 samples or constant features")
    if method == "pca":
        reducer = PCA(n_components=available, random_state=seed)
        values = reducer.fit_transform(x)
        coords = np.pad(values, ((0, 0), (0, dimensions - available)))
        meta["explained_variance_ratio"] = reducer.explained_variance_ratio_.tolist()
    elif method == "umap":
        import umap
        neighbors = min(15, n - 1)
        # Random initialization also supports tiny datasets without spectral eigensolver failure.
        reducer = umap.UMAP(n_components=dimensions, n_neighbors=neighbors, min_dist=.1,
                            metric="cosine", random_state=seed, n_jobs=1, init="random")
        coords = reducer.fit_transform(x)
        meta.update(n_neighbors=neighbors, min_dist=.1, metric="cosine", init="random")
    elif method == "tsne":
        pre_dimensions = min(50, n - 1, width)
        reduced = PCA(n_components=pre_dimensions, random_state=seed).fit_transform(x)
        perplexity = min(30., max(1., (n - 1) / 3.))
        reducer = TSNE(n_components=dimensions, random_state=seed, perplexity=perplexity,
                       max_iter=1000, init="pca" if pre_dimensions >= dimensions else "random", learning_rate="auto")
        coords = reducer.fit_transform(reduced)
        meta.update(pre_pca_dimensions=pre_dimensions, perplexity=perplexity, max_iter=1000, kl_divergence=float(reducer.kl_divergence_))
    else:
        raise ValueError(f"Unknown projection: {method}")
    if not np.isfinite(coords).all():
        raise ValueError(f"Non-finite {method} coordinates")
    return np.asarray(coords, dtype=np.float32), meta


def fit_patch_pca(patches, components=(1, 2, 3), seed=42, maximum=50000):
    indices = np.array(components) - 1
    if len(indices) != 3 or np.any(indices < 0) or len(set(indices)) != 3:
        raise ValueError("Choose three distinct positive 1-based PCA components")
    combined = np.concatenate(patches).astype(np.float32)
    rank = min(combined.shape)
    if indices.max() >= rank:
        raise ValueError(f"Requested PC {indices.max()+1}, but at most {rank} components are available")
    rng = np.random.default_rng(seed)
    fit = combined if len(combined) <= maximum else combined[rng.choice(len(combined), maximum, replace=False)]
    if float(np.max(np.var(fit, axis=0))) < 1e-14:
        mean = fit.mean(0)
        basis = np.zeros((int(indices.max()) + 1, fit.shape[1]), dtype=np.float32)
        explained = np.zeros(len(basis))
    else:
        pca = PCA(n_components=int(indices.max()) + 1, random_state=seed)
        pca.fit(fit)
        mean, basis, explained = pca.mean_, pca.components_, pca.explained_variance_ratio_
        for row in basis:
            if row[np.argmax(np.abs(row))] < 0:
                row *= -1
    projected = [(p - mean) @ basis[indices].T for p in patches]
    all_scores = np.concatenate(projected)
    med = np.median(all_scores, axis=0)
    mad = np.median(np.abs(all_scores - med), axis=0)
    limits = []
    for channel in range(3):
        values = all_scores[:, channel]
        inside = values[np.abs(values - med[channel]) <= 2 * mad[channel] + 1e-8]
        if not len(inside):
            inside = values
        limits.append([float(inside.min()), float(inside.max())])
    return projected, dict(components=list(components), explained_variance_ratio=explained.tolist(), fitted_patches=len(fit),
                           mean=mean, basis=basis, color_limits=limits, scaling="shared per model; median absolute deviation threshold=2")


def scale_rgb(scores, limits):
    limits = np.asarray(limits)
    return np.clip((scores - limits[:, 0]) / np.maximum(limits[:, 1] - limits[:, 0], 1e-8), 0, 1)
