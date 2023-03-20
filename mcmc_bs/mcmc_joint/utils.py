import os
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision

import jax.numpy as jnp

from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from jax import random, jit

def graph_to_mat(G):
    """Returns adjacency matrix of ig.Graph object """
    return jnp.array(G.get_adjacency().data)

def tree_zip_leading(pytree_list):
    """
    Converts n pytrees without leading dimension into one pytree with leading dim [n, ...]
    """
    return tree_map(lambda *args: jnp.stack([*args]) if len(args) > 1 else tree_expand_leading_by(*args, 1), *pytree_list)

def tree_expand_leading_by(pytree, n):
    """
    Converts pytree with leading pytrees with additional `n` leading dimensions
    """
    return tree_map(lambda leaf: jnp.expand_dims(leaf, axis=tuple(range(n))), pytree)

def tree_key_split(key, pytree):
    """
    Generates one subkey from `key` for each leaf of `pytree` and returns it in tree of shape `pytree`
    """

    tree_flat, treedef = tree_flatten(pytree)
    subkeys_flat = random.split(key, len(tree_flat))
    subkeys_tree = tree_unflatten(treedef, subkeys_flat)
    return subkeys_tree

def tree_shapes(pytree):
    """
    Returns pytree with same tree but leaves replaced by original shapes
    """
    return tree_map(lambda leaf: jnp.array(leaf.shape), pytree)

@jit
def sel(mat, mask):
    '''
        jit/vmap helper function

        mat:   [N, d]
        mask:  [d, ]   boolean 

        returns [N, d] with columns of `mat` with `mask` == 1 non-zero a
        and the columns with `mask` == 0 are zero

        e.g. 
        mat 
        1 2 3
        4 5 6
        7 8 9

        mask
        1 0 1

        out
        1 0 3
        4 0 6
        7 0 9
    '''
    return jnp.where(mask, mat, 0)
