import numpy as np


def find_closest_at(at, train_ats):
	dist = np.linalg.norm(train_ats - at, axis=1)
	idx = np.argmin(dist)
	return float(dist[idx]), train_ats[idx]


def fetch_dsa(train_ats, train_labels, target_ats, target_labels):
	train_ats = np.asarray(train_ats, dtype=np.float64)
	target_ats = np.asarray(target_ats, dtype=np.float64)
	train_labels = np.asarray(train_labels)
	target_labels = np.asarray(target_labels)

	class_matrix = {}
	all_idx = np.arange(len(train_labels))

	for idx, label in enumerate(train_labels):
		class_matrix.setdefault(label, []).append(idx)

	dsa = []
	for at, label in zip(target_ats, target_labels):
		same_idx = class_matrix[label]
		other_idx = np.setdiff1d(all_idx, same_idx)

		a_dist, a_dot = find_closest_at(at, train_ats[same_idx])
		b_dist, _ = find_closest_at(a_dot, train_ats[other_idx])

		dsa.append(a_dist / b_dist if b_dist > 0 else np.nan)

	return dsa


__all__ = [
	"find_closest_at",
	"fetch_dsa",
]
