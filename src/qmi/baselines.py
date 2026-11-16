"""Classical multi-class segmentation baselines: Otsu heuristic, K-means, watershed."""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import binary_erosion
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from skimage.filters import threshold_otsu, sobel
from skimage.segmentation import watershed
from .data.preprocessing import brain_mask

TARGET_LABELS = np.array([0, 1, 2, 4], dtype=np.uint8)


def otsu_multiclass(modalities: np.ndarray) -> np.ndarray:
    """Deterministic threshold baseline. WT uses FLAIR; T1ce heuristics split TC/ET.

    This is deliberately a simple classical reference, not a clinical algorithm.
    """
    m = np.asarray(modalities, dtype=np.float32)
    if m.shape[0] != 4:
        raise ValueError("Expected modalities in FLAIR,T1,T1ce,T2 order")
    flair, _, t1ce, _ = m
    mask = brain_mask(m)
    result = np.zeros(mask.shape, dtype=np.uint8)
    if not np.any(mask):
        return result
    values = flair[mask]
    threshold = float(threshold_otsu(values)) if np.ptp(values) > 1e-8 else float(values.mean())
    wt = mask & (flair > threshold)
    # Quantile thresholds are estimated only from the current input image, not its labels.
    if np.any(wt):
        tce_vals = t1ce[wt]
        tc_thr = float(np.quantile(tce_vals, 0.45))
        et_thr = float(np.quantile(tce_vals, 0.85))
        tc = wt & (t1ce >= tc_thr)
        et = tc & (t1ce >= et_thr)
    else:
        tc = np.zeros_like(mask)
        et = np.zeros_like(mask)
    result[wt] = 2
    result[tc] = 1
    result[et] = 4
    return result


class KMeansMultiClassBaseline:
    """Train-split-calibrated K-means baseline with cluster-to-BraTS-label mapping."""
    def __init__(self, n_clusters: int = 4, random_state: int = 42, max_iter: int = 100):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state,
                                         batch_size=4096, n_init=5, max_iter=max_iter)
        self.cluster_to_label: dict[int, int] = {}
        self.is_fitted = False

    @staticmethod
    def _pixel_features(modalities: np.ndarray) -> np.ndarray:
        return np.moveaxis(np.asarray(modalities, dtype=np.float32), 0, -1).reshape(-1, 4)

    def fit(self, training_slices, max_pixels_per_slice: int = 5000):
        rng = np.random.default_rng(self.random_state)
        xs, ys = [], []
        for modalities, labels in training_slices:
            X = self._pixel_features(modalities)
            y = np.asarray(labels, dtype=np.uint8).ravel()
            take = min(max_pixels_per_slice, len(y))
            idx = rng.choice(len(y), size=take, replace=False)
            xs.append(X[idx]); ys.append(y[idx])
        if not xs:
            raise ValueError("No training slices supplied to K-means baseline")
        X = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        scaled = self.scaler.fit_transform(X)
        clusters = self.clusterer.fit_predict(scaled)
        contingency = np.zeros((self.n_clusters, len(TARGET_LABELS)), dtype=np.int64)
        label_to_col = {int(v): i for i, v in enumerate(TARGET_LABELS)}
        for c, t in zip(clusters, y):
            if int(t) in label_to_col:
                contingency[int(c), label_to_col[int(t)]] += 1
        rows, cols = linear_sum_assignment(contingency.max(initial=0) - contingency)
        self.cluster_to_label = {int(r): int(TARGET_LABELS[c]) for r, c in zip(rows, cols)}
        # Any unmatched cluster is assigned background conservatively.
        for c in range(self.n_clusters):
            self.cluster_to_label.setdefault(c, 0)
        self.is_fitted = True
        return self

    def predict(self, modalities: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("KMeansMultiClassBaseline.fit must be called first")
        X = self._pixel_features(modalities)
        clusters = self.clusterer.predict(self.scaler.transform(X))
        labels = np.array([self.cluster_to_label[int(c)] for c in clusters], dtype=np.uint8)
        return labels.reshape(np.asarray(modalities).shape[1:])


class WatershedMultiClassBaseline:
    """Marker-controlled watershed using calibrated K-means regions as seed labels."""
    def __init__(self, kmeans_model: KMeansMultiClassBaseline):
        self.kmeans_model = kmeans_model

    def predict(self, modalities: np.ndarray) -> np.ndarray:
        m = np.asarray(modalities, dtype=np.float32)
        mask = brain_mask(m)
        if not np.any(mask):
            return np.zeros(mask.shape, dtype=np.uint8)
        initial = self.kmeans_model.predict(m)
        gradient = (sobel(m[0]) + sobel(m[2])) / 2.0
        markers = np.zeros(mask.shape, dtype=np.int32)
        # Watershed label 0 means "unmarked" to scikit-image, so use marker 5
        # for the background class and map it back to output label 0 afterwards.
        background_region = (initial == 0) & mask
        background_seeds = binary_erosion(background_region, iterations=1, border_value=0)
        if not np.any(background_seeds) and np.any(background_region):
            coords = np.argwhere(background_region)
            background_seeds[tuple(coords[len(coords)//2])] = True
        markers[background_seeds] = 5
        for label in (1, 2, 4):
            region = (initial == label) & mask
            seeds = binary_erosion(region, iterations=1, border_value=0)
            if not np.any(seeds) and np.any(region):
                # Keep at least one seed for every represented class.
                coords = np.argwhere(region)
                seeds[tuple(coords[len(coords)//2])] = True
            markers[seeds] = label
        if not np.any(markers):
            return initial
        out = watershed(gradient, markers=markers, mask=mask).astype(np.uint8)
        out[out == 5] = 0
        out[~mask] = 0
        return out
