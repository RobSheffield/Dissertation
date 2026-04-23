import numpy as np
from scipy.stats import gaussian_kde


def remove_low_variance_cols(train_ats, var_threshold=1e-5):
	train_ats = np.asarray(train_ats, dtype=np.float64)
	variances = np.var(train_ats, axis=0)
	removed_cols = np.where(variances < var_threshold)[0]
	refined = np.delete(train_ats, removed_cols, axis=1)
	return refined, removed_cols


def _project_for_kde(train_ats):
	"""
	Project features to a rank-safe subspace for gaussian_kde.
	Returns projected_train, projection_metadata.
	"""
	train_ats = np.asarray(train_ats, dtype=np.float64)
	n_samples, n_features = train_ats.shape

	# gaussian_kde requires n_samples > n_features and non-singular covariance.
	max_dims = max(1, n_samples - 1)
	target_dims = min(n_features, max_dims)

	mean_vec = np.mean(train_ats, axis=0, keepdims=True)
	centered = train_ats - mean_vec

	# Full-rank subspace via SVD; if SVD fails, fallback to axis truncation.
	try:
		_, svals, vt = np.linalg.svd(centered, full_matrices=False)
		# Keep numerically stable directions only.
		tol = max(centered.shape) * np.finfo(np.float64).eps * (svals[0] if svals.size > 0 else 1.0)
		numerical_rank = int(np.sum(svals > tol)) if svals.size > 0 else 0
		keep_dims = max(1, min(target_dims, numerical_rank if numerical_rank > 0 else target_dims))
		basis = vt[:keep_dims].T
		projected = centered @ basis
		meta = {
			"mean": mean_vec.ravel(),
			"basis": basis,
		}
		return projected, meta
	except np.linalg.LinAlgError:
		projected = centered[:, :target_dims]
		meta = {
			"mean": mean_vec.ravel(),
			"basis": None,
			"truncate": target_dims,
		}
		return projected, meta


def _apply_projection(at, projection_meta):
	at = np.asarray(at, dtype=np.float64)
	mean = projection_meta["mean"]
	centered = at - mean
	basis = projection_meta.get("basis")
	if basis is not None:
		return centered @ basis
	truncate = projection_meta.get("truncate")
	if truncate is None:
		return centered
	return centered[:truncate]


def fit_lsa(train_ats, var_threshold=1e-5):
	refined_train_ats, removed_cols = remove_low_variance_cols(train_ats, var_threshold)
	projected_train_ats, projection_meta = _project_for_kde(refined_train_ats)

	# Last-resort jitter if covariance is near-singular despite projection.
	try:
		kde = gaussian_kde(projected_train_ats.T)
	except np.linalg.LinAlgError:
		rng = np.random.default_rng(42)
		jitter = rng.normal(loc=0.0, scale=1e-8, size=projected_train_ats.shape)
		kde = gaussian_kde((projected_train_ats + jitter).T)

	prep = {
		"removed_cols": removed_cols,
		"projection": projection_meta,
	}
	return kde, prep


def lsa_score(kde, at, removed_cols):
	# Backward compatible: `removed_cols` can be old ndarray or new prep dict.
	if isinstance(removed_cols, dict):
		cols = removed_cols.get("removed_cols", [])
		projection = removed_cols.get("projection")
	else:
		cols = removed_cols
		projection = None

	refined_at = np.delete(np.asarray(at, dtype=np.float64), cols, axis=0)
	if projection is not None:
		refined_at = _apply_projection(refined_at, projection)

	logp = kde.logpdf(refined_at.reshape(-1, 1))
	return float(-logp[0])


def fetch_lsa(train_ats, target_ats, var_threshold=1e-5):
	kde, removed_cols = fit_lsa(train_ats, var_threshold=var_threshold)
	return [lsa_score(kde, at, removed_cols) for at in target_ats]


__all__ = [
	"remove_low_variance_cols",
	"fit_lsa",
	"lsa_score",
	"fetch_lsa",
]
