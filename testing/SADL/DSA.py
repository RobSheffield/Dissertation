import numpy as np


def find_closest_at(at, train_ats):
	dist = np.linalg.norm(train_ats - at, axis=1)
	idx = np.argmin(dist)
	return float(dist[idx]), train_ats[idx]


def _k_nearest_mean_distance(at, ref_ats, k=3):
	if len(ref_ats) == 0:
		return np.nan
	dists = np.linalg.norm(ref_ats - at, axis=1)
	k = max(1, min(int(k), len(dists)))
	nearest = np.partition(dists, k - 1)[:k]
	return float(np.mean(nearest))


def fetch_dsa(train_ats, train_labels, target_ats, target_labels, k=3, eps=1e-12, clip_quantile=0.99):
	train_ats = np.asarray(train_ats, dtype=np.float64)
	target_ats = np.asarray(target_ats, dtype=np.float64)
	train_labels = np.asarray(train_labels)
	target_labels = np.asarray(target_labels)

	if train_ats.ndim != 2 or target_ats.ndim != 2:
		raise ValueError("train_ats and target_ats must be 2D arrays.")
	if train_ats.shape[1] != target_ats.shape[1]:
		raise ValueError("train_ats and target_ats must have the same feature dimension.")
	if len(train_ats) != len(train_labels):
		raise ValueError("train_labels length must match train_ats rows.")
	if len(target_ats) != len(target_labels):
		raise ValueError("target_labels length must match target_ats rows.")
	if len(train_ats) == 0 or len(target_ats) == 0:
		return []

	class_matrix = {}
	all_idx = np.arange(len(train_labels))

	for idx, label in enumerate(train_labels):
		class_matrix.setdefault(label, []).append(idx)

	dsa = []
	for at, label in zip(target_ats, target_labels):
		same_idx = np.asarray(class_matrix.get(label, []), dtype=int)
		if len(same_idx) == 0:
			same_idx = all_idx

		other_idx = np.setdiff1d(all_idx, same_idx)
		if len(other_idx) == 0:
			# Single-class fallback: use all train points for denominator.
			other_idx = all_idx

		a_dist = _k_nearest_mean_distance(at, train_ats[same_idx], k=k)
		b_dist = _k_nearest_mean_distance(at, train_ats[other_idx], k=k)
		if np.isnan(a_dist) or np.isnan(b_dist):
			dsa.append(np.nan)
		else:
			dsa.append(a_dist / max(b_dist, eps))

	dsa = np.asarray(dsa, dtype=np.float64)
	finite_mask = np.isfinite(dsa)
	if not np.any(finite_mask):
		return [0.0] * len(target_ats)

	finite_values = dsa[finite_mask]
	clip_value = np.quantile(finite_values, clip_quantile)
	dsa = np.nan_to_num(dsa, nan=clip_value, posinf=clip_value, neginf=0.0)
	dsa = np.clip(dsa, 0.0, clip_value)

	return dsa.tolist()


__all__ = [
	"find_closest_at",
	"fetch_dsa",
]
