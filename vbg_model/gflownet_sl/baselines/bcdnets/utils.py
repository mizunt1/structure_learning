from typing import Union, Callable, cast, Any
import jax.numpy as np
from jax import vmap, jit, vjp, ops, grad
from jax import random as rnd
import numpy as onp
import cdt
import time
from sklearn.metrics import roc_curve, auc
import optax

# cdt.SETTINGS.rpath = "/path/to/Rscript/binary""
from cdt.metrics import SHD_CPDAG
import networkx as nx


import pickle as pkl
from gflownet_sl.baselines.bcdnets.divergences import (
    kl_sample_loss,
    wasserstein_sample_loss,
    kl_loss,
    wasserstein_loss,
    precision_kl_sample_loss,
    precision_kl_loss,
    precision_wasserstein_sample_loss,
    precision_wasserstein_loss,
)
import haiku as hk
from jax.flatten_util import ravel_pytree
from jax import tree_util
import jax.numpy as jnp
from jax.tree_util import tree_map

#### from dag_utils.py
from typing import Dict
def count_accuracy(W_true, W_est, W_und=None) -> Dict["str", float]:
    """
    Compute FDR, TPR, and FPR for B, or optionally for CPDAG B + B_und.

    Args:
        W_true: ground truth graph
        W_est: predicted graph
        W_und: predicted undirected edges in CPDAG, asymmetric

    Returns in dict:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    B_true = W_true != 0
    B = W_est != 0
    B_und = None if W_und is None else W_und
    d = B.shape[0]

    # linear index of nonzeros
    pred = np.flatnonzero(B)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    if B_und is not None:
        # treat undirected edge favorably
        pred_und = np.flatnonzero(B_und)
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    if B_und is not None:
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)  # type: ignore
        false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    if B_und is not None:
        pred_size += len(pred_und)  # type: ignore
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    B_lower = np.tril(B + B.T)
    if B_und is not None:
        B_lower += np.tril(B_und + B_und.T)
    pred_lower = np.flatnonzero(B_lower)
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)

    return {"fdr": fdr, "tpr": tpr, "fpr": fpr, "shd": shd, "pred_size": pred_size}


Tensor = Union[onp.ndarray, np.ndarray]


def un_pmap(x):
    return tree_map(lambda x: x[0], x)


def get_double_tree_variance(w, z) -> jnp.ndarray:
    """Given two pytrees w, z, compute std[w, z]"""

    def tree_size(x):
        leaves, _ = tree_util.tree_flatten(x)
        return sum([jnp.size(leaf) for leaf in leaves])

    def tree_sum(x):
        leaves, _ = tree_util.tree_flatten(x)
        return sum([jnp.sum(leaf) for leaf in leaves])

    def sum_square_tree(x, mean):
        leaves, _ = tree_util.tree_flatten(x)
        return sum([jnp.sum((leaf - mean) ** 2) for leaf in leaves])

    # Average over num_repeats, then over all params

    total_size = tree_size(w) + tree_size(z)
    grad_mean = (tree_sum(w) + tree_sum(z)) / total_size
    tree_variance = (
        sum_square_tree(w, grad_mean) + sum_square_tree(z, grad_mean)
    ) / total_size
    return jnp.sqrt(tree_variance)


def num_params(params: hk.Params) -> int:
    return len(ravel_pytree(params)[0])


def make_to_W(dim: int,) -> Callable[[jnp.ndarray], jnp.ndarray]:
    out = np.zeros((dim, dim))
    w_param_dim = dim * (dim - 1)
    upper_idx = np.triu_indices(dim, 1)
    lower_idx = np.tril_indices(dim, -1)

    def to_W(w_params: jnp.ndarray) -> jnp.ndarray:
        """Turns a (d x (d-1)) vector into a d x d matrix with zero diagonal."""
        tmp = ops.index_update(out, upper_idx, w_params[: w_param_dim // 2])
        tmp = ops.index_update(tmp, lower_idx, w_params[w_param_dim // 2 :])
        return tmp

    return to_W


def from_W(W: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Turns a d x d matrix into a (d x (d-1)) vector with zero diagonal."""
    out_1 = W[np.triu_indices(dim, 1)]
    out_2 = W[np.tril_indices(dim, -1)]
    return np.concatenate([out_1, out_2])


def lower(theta: Tensor, dim: int) -> Tensor:
    """Given n(n-1)/2 parameters theta, form a
    strictly lower-triangular matrix"""
    out = np.zeros((dim, dim))
    out = ops.index_update(out, np.triu_indices(dim, 1), theta).T
    return out


def upper(theta: Tensor, dim: int) -> Tensor:
    """Given n(n-1)/2 parameters theta, form a
    strictly upper-triangular matrix"""
    out = np.zeros((dim, dim))
    out = ops.index_update(out, np.tril_indices(dim, -1), theta).T
    return out


def get_variances(W_params: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    """The maximum likelihood estimate of sigma is the sample variance"""
    dim = X.shape[1]
    to_W = make_to_W(dim)
    W = to_W(W_params)
    residuals = X.T - W.T @ X.T
    residuals = cast(jnp.ndarray, residuals)
    return np.mean(residuals ** 2, axis=1)


def get_variances_from_W(W, X):
    """The maximum likelihood estimate of sigma is the sample variance"""
    residuals = X.T - W.T @ X.T
    return np.mean(residuals ** 2, axis=1)


def get_variance(W_params, X):
    """The maximum likelihood estimate in the equal variance case"""
    n, dim = X.shape
    to_W = make_to_W(dim)
    W = to_W(W_params)
    residuals = X.T - W.T @ X.T
    return np.sum(residuals ** 2) / (dim * n)


def samples_near(mode: Tensor, samples: Tensor, tol: float):
    """Returns the number of samples in an l_0 ball around the mode"""
    is_close = np.linalg.norm(samples - mode[None, :], ord=np.inf, axis=-1) < tol
    return np.mean(is_close)


def get_labels(dim):
    w_param_dim = dim * (dim - 1)
    x1s, y1s = np.triu_indices(dim, 1)
    x2s, y2s = np.tril_indices(dim, -1)
    xs = np.concatenate((x1s, x2s))
    ys = np.concatenate((y1s, y2s))
    return [f"{xs[i]}->{ys[i]}" for i in range(w_param_dim)]


def get_permutation(key: jnp.ndarray, d: int) -> Tensor:
    return rnd.permutation(key, np.eye(d))


def our_jacrev(fun):
    def jacfun(x):
        y, pullback = vjp(fun, x)
        jac = vmap(pullback, in_axes=0)(np.eye(len(y)))
        return jac, y

    return jacfun


def save_params(P_params, L_params, L_states, P_opt_params, L_opt_state, filename,
                train_dag_samples=None, dag_samples=None, ground_truth_W=None):
    import os
    newpath = "./bcdnets_tmp"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    filenames = []
    filenames.append("./bcdnets_tmp/P_params" + filename)
    filenames.append("./bcdnets_tmp/L_params" + filename)
    filenames.append("./bcdnets_tmp/L_states" + filename)
    filenames.append("./bcdnets_tmp/P_opt" + filename)
    filenames.append("./bcdnets_tmp/L_opt" + filename)

    inputs = [P_params, L_params, L_states, P_opt_params, L_opt_state]

    if train_dag_samples is not None:
        filenames.append("./bcdnets_tmp/train_dag_samples" + filename)
        inputs = inputs + [train_dag_samples]
    if dag_samples is not None:
        filenames.append("./bcdnets_tmp/test_dag_samples" + filename)
        inputs = inputs + [dag_samples]
    if ground_truth_W is not None:
        filenames.append("./bcdnets_tmp/ground_truth_W" + filename)
        inputs = inputs + [ground_truth_W]

    for name, obj in zip(filenames, inputs):
        pkl.dump(obj, open(name, "wb"))


def load_params(filename):
    filenames = []
    filenames.append("./bcdnets_tmp/P_params" + filename)
    filenames.append("./bcdnets_tmp/L_params" + filename)
    filenames.append("./bcdnets_tmp/L_states" + filename)
    filenames.append("./bcdnets_tmp/P_opt" + filename)
    filenames.append("./bcdnets_tmp/L_opt" + filename)
    outs = []
    for name in filenames:
        outs.append(pkl.load(open(name, "rb")))
    return outs


def eval_W_ev(
    est_W,
    true_W,
    true_noise,
    threshold,
    Xs,
    est_noise=None,
    provided_x_prec=None,
    do_shd_c=True,
    get_wasserstein=True,
    do_sid=True,
):
    dim = np.shape(est_W)[0]
    if provided_x_prec is None:
        x_prec = onp.linalg.inv(np.cov(Xs.T))
    else:
        x_prec = provided_x_prec
    x_prec = onp.linalg.inv(np.cov(Xs.T))
    est_W_clipped = np.where(np.abs(est_W) > threshold, est_W, 0)
    # Can provide noise or use the maximum-likelihood estimate
    if est_noise is None:
        est_noise = np.ones(dim) * get_variance(from_W(est_W_clipped, dim), Xs)
    else:
        est_noise = np.ones(dim) * est_noise
    stats = count_accuracy(true_W, est_W_clipped)

    if get_wasserstein:
        true_wasserstein_distance = precision_wasserstein_loss(
            true_noise, true_W, est_noise, est_W_clipped,
        )
        sample_wasserstein_loss = precision_wasserstein_sample_loss(
            x_prec, est_noise, est_W_clipped
        )
    else:
        true_wasserstein_distance, sample_wasserstein_loss = 0.0, 0.0
    true_KL_divergence = precision_kl_loss(true_noise, true_W, est_noise, est_W_clipped)
    sample_kl_divergence = precision_kl_sample_loss(x_prec, est_noise, est_W_clipped)
    if do_shd_c:
        shd_c = SHD_CPDAG(
            nx.DiGraph(onp.array(est_W_clipped)), nx.DiGraph(onp.array(true_W))
        )
        stats["shd_c"] = shd_c
    else:
        stats["shd_c"] = np.nan
    if do_sid:
        sid = SHD_CPDAG(onp.array(est_W_clipped != 0), onp.array(true_W != 0))
    else:
        sid = onp.nan
    stats["true_kl"] = true_KL_divergence
    stats["sample_kl"] = sample_kl_divergence
    stats["true_wasserstein"] = true_wasserstein_distance
    stats["sample_wasserstein"] = sample_wasserstein_loss
    stats["MSE"] = np.mean((Xs.T - est_W_clipped.T @ Xs.T) ** 2)
    stats["sid"] = sid
    return stats


def auroc(Ws, W_true, threshold):
    """Given a sample of adjacency graphs of shape n x d x d, 
    compute the AUROC for detecting edges. For each edge, we compute
    a probability that there is an edge there which is the frequency with 
    which the sample has edges over threshold."""
    _, dim, dim = Ws.shape
    edge_present = jnp.abs(Ws) > threshold
    prob_edge_present = jnp.mean(edge_present, axis=0)
    true_edges = from_W(jnp.abs(W_true) > threshold, dim).astype(int)
    predicted_probs = from_W(prob_edge_present, dim)
    fprs, tprs, _ = roc_curve(y_true=true_edges, y_score=predicted_probs, pos_label=1)
    auroc = auc(fprs, tprs)
    return auroc


def eval_W_non_ev(
    est_W,
    true_W,
    true_noise,
    threshold,
    Xs,
    est_noise=None,
    provided_x_prec=None,
    do_shd_c=True,
    get_wasserstein=True,
    do_sid=True,
):
    dim = np.shape(est_W)[0]
    if provided_x_prec is None:
        x_prec = onp.linalg.inv(np.cov(Xs.T))
    else:
        x_prec = provided_x_prec
    est_W_clipped = np.where(np.abs(est_W) > threshold, est_W, 0)
    # Can provide noise or use the maximum-likelihood estimate
    if est_noise is None:
        est_noise = np.ones(dim) * jit(get_variances)(from_W(est_W_clipped, dim), Xs)
    # Else est_noise is already given as a vector
    stats = count_accuracy(true_W, est_W_clipped)
    true_KL_divergence = jit(precision_kl_loss)(
        true_noise, true_W, est_noise, est_W_clipped
    )
    sample_kl_divergence = jit(precision_kl_sample_loss)(
        x_prec, est_noise, est_W_clipped
    )

    if get_wasserstein:
        true_wasserstein_distance = jit(precision_wasserstein_loss)(
            true_noise, true_W, est_noise, est_W_clipped,
        )
        sample_wasserstein_loss = jit(precision_wasserstein_sample_loss)(
            x_prec, est_noise, est_W_clipped
        )
    else:
        true_wasserstein_distance, sample_wasserstein_loss = 0.0, 0.0

    if do_shd_c:
        shd_c = SHD_CPDAG(
            nx.DiGraph(onp.array(est_W_clipped)), nx.DiGraph(onp.array(true_W))
        )
        # print("SHD_CPDAG didn't work: do you have R installed?")
    else:
        shd_c = np.nan
    if do_sid:
        sid = SHD_CPDAG(onp.array(est_W_clipped != 0), onp.array(true_W != 0))
    else:
        sid = onp.nan

    stats["true_kl"] = float(true_KL_divergence)
    stats["sample_kl"] = float(sample_kl_divergence)
    stats["true_wasserstein"] = float(true_wasserstein_distance)
    stats["sample_wasserstein"] = float(sample_wasserstein_loss)
    stats["MSE"] = float(np.mean((Xs.T - est_W_clipped.T @ Xs.T) ** 2))
    stats["shd_c"] = shd_c
    stats["sid"] = sid
    return stats


def eval_W(est_W, true_W, true_noise, threshold, Xs, get_wasserstein=True):
    dim = np.shape(est_W)[0]
    x_cov = np.cov(Xs.T)
    est_W_clipped = np.where(np.abs(est_W) > threshold, est_W, 0)
    est_noise = jit(get_variances)(from_W(est_W_clipped, dim), Xs)
    stats = count_accuracy(true_W, est_W_clipped)
    true_KL_divergence = jit(kl_loss)(true_noise, true_W, est_noise, est_W_clipped,)
    sample_kl_divergence = jit(kl_sample_loss)(x_cov, est_noise, est_W)
    if get_wasserstein:
        true_wasserstein_distance = jit(wasserstein_loss)(
            true_noise, true_W, est_noise, est_W_clipped,
        )
        sample_wasserstein_loss = jit(wasserstein_sample_loss)(x_cov, est_noise, est_W)
    else:
        true_wasserstein_distance, sample_wasserstein_loss = 0.0, 0.0

    shd_c = np.nan
    try:
        shd_c = SHD_CPDAG(
            nx.DiGraph(onp.array(est_W_clipped)), nx.DiGraph(onp.array(true_W))
        )
    except:
        # print("SHD_CPDAG didn't work: do you have R installed?")
        stats["shd_c"] = np.nan
    stats["true_kl"] = true_KL_divergence
    stats["sample_kl"] = sample_kl_divergence
    stats["true_wasserstein"] = true_wasserstein_distance
    stats["sample_wasserstein"] = sample_wasserstein_loss
    stats["MSE"] = np.mean((est_W_clipped - true_W) ** 2)
    stats["shd_c"] = shd_c
    return stats


def random_str():
    out = onp.random.randint(1_000_000) + time.time()
    return str(out)


def ff2(x):
    if type(x) is str:
        return x
    if onp.abs(x) > 1000 or onp.abs(x) < 0.1:
        return onp.format_float_scientific(x, 3)
    else:
        return f"{x:.2f}"


def rk(x):
    return rnd.PRNGKey(x)


def fit_known_edges(
    W_binary: Tensor,
    Xs: Tensor,
    tol: float = 1e-3,
    max_iters: int = 3_000,
    lr: float = 1e-2,
    verbose: bool = True,
    lambda_1: float = 0.0,
) -> jnp.ndarray:
    """Given a binary adjacency matrix W_binary, fit linear SEM coefficients from data Xs"""
    # Make sure W_binary is a 1-0 adjacency matrix
    mask = np.where(W_binary == 0, np.zeros_like(W_binary), np.ones_like(W_binary))
    dim = len(W_binary)
    # Add a bit of regularization to keep things nicely-conditioned
    lambda_2 = 1e-6

    def make_optimizer():
        """SGD with nesterov momentum and a custom lr schedule.
        We should be able to use Nesterov momentum since the problem is convex"""
        # (Maybe we will run into issues with the masking etc interacting with the nesterov?)
        return optax.sgd(lr, nesterov=True)

    def inner_loss(p):
        W = p * mask
        return (
            jnp.linalg.norm(Xs.T - W.T @ Xs.T)
            - jnp.linalg.slogdet(jnp.eye(dim) - W)[1]
            + lambda_1 * jnp.sum(np.abs(W))
            + lambda_2 * jnp.sum(W ** 2)
        )

    @jit
    def step(p, opt_state):
        g = grad(inner_loss)(p)
        updates, opt_state = make_optimizer().update(g, opt_state, p)
        p = optax.apply_updates(p, updates)
        return p, opt_state, g

    p = rnd.normal(rnd.PRNGKey(0), shape=W_binary.shape)
    g = jnp.ones_like(W_binary) * jnp.inf
    opt_state = make_optimizer().init(p)

    for i in range(max_iters):
        if jnp.linalg.norm(g) < tol:
            if verbose:
                print(f"Converged to gradient norm <{tol} after {i} iterations")
            return p * mask
        p, opt_state, g = step(p, opt_state)

    if verbose:
        print(
            f"Failed to converge to tol {tol}, actual gradient norm: {jnp.linalg.norm(g)}"
        )
    return p * mask


def npperm(M):
    # From user lesshaste on github: https://github.com/scipy/scipy/issues/7151
    n = M.shape[0]
    d = onp.ones(n)
    j = 0
    s = 1
    f = onp.arange(n)
    v = M.sum(axis=0)
    p = onp.prod(v)
    while j < n - 1:
        v -= 2 * d[j] * M[j]
        d[j] = -d[j]
        s = -s
        prod = onp.prod(v)
        p += s * prod
        f[0] = 0
        f[j] = f[j + 1]
        f[j + 1] = j + 1
        j = f[0]
    return p / 2 ** (n - 1)
