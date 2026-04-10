import numpy as np
from scipy.stats import gaussian_kde


def remove_low_variance_cols(train_ats, var_threshold=1e-5):
	train_ats = np.asarray(train_ats, dtype=np.float64)
	variances = np.var(train_ats, axis=0)
	removed_cols = np.where(variances < var_threshold)[0]
	refined = np.delete(train_ats, removed_cols, axis=1)
	return refined, removed_cols


def fit_lsa(train_ats, var_threshold=1e-5):
	refined_train_ats, removed_cols = remove_low_variance_cols(train_ats, var_threshold)
	kde = gaussian_kde(refined_train_ats.T)
	return kde, removed_cols


def lsa_score(kde, at, removed_cols):
	refined_at = np.delete(np.asarray(at, dtype=np.float64), removed_cols, axis=0)
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
