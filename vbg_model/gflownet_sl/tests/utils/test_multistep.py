import pytest
import numpy as np

from numpy.random import default_rng
from gflownet_sl.utils.multistep import get_nstep_indices, get_nstep_mask

def test_get_nstep_indices_rows_cols():
    rng = default_rng(0)
    n_step = 3
    indices = get_nstep_indices(n_step=n_step)

    # Compute the sum for all subsequences
    inputs = rng.normal(size=(n_step, 2, 3, 5))
    outputs = np.zeros((n_step * (n_step + 1) // 2, 2, 3, 5))
    np.add.at(outputs, indices.rows, inputs[indices.cols])

    expected_outputs = np.zeros_like(outputs)
    row = 0
    for block in range(1, n_step + 1):
        for start in range(block):
            for i in range(start, block):
                expected_outputs[row] += inputs[i]
            row += 1

    np.testing.assert_allclose(outputs, expected_outputs)


def test_get_nstep_indices_start():
    rng = default_rng(0)
    n_step = 3
    indices = get_nstep_indices(n_step=n_step)

    # Get start element for all subsequences
    inputs = rng.normal(size=(n_step, 2, 3, 5))
    outputs = inputs[indices.start]

    expected_outputs = np.zeros_like(outputs)
    row = 0
    for block in range(n_step):
        for _ in range(block + 1):
            expected_outputs[row] = inputs[block]
            row += 1

    assert outputs.shape == (n_step * (n_step + 1) // 2, 2, 3, 5)
    np.testing.assert_allclose(outputs, expected_outputs)


def test_get_nstep_indices_end():
    rng = default_rng(0)
    n_step = 3
    indices = get_nstep_indices(n_step=n_step)

    # Get end element for all subsequences
    inputs = rng.normal(size=(n_step, 2, 3, 5))
    outputs = inputs[indices.end]

    expected_outputs = np.zeros_like(outputs)
    row = 0
    for block in range(n_step):
        for i in range(block + 1):
            expected_outputs[row] = inputs[i]
            row += 1

    assert outputs.shape == (n_step * (n_step + 1) // 2, 2, 3, 5)
    np.testing.assert_allclose(outputs, expected_outputs)


def test_get_nstep_mask():
    lengths = np.array([2, 3, 3, 1])
    mask = get_nstep_mask(lengths, n_step=3)

    expected_mask = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0]
    ], dtype=np.float32)

    assert mask.shape == (3 * (3 + 1) // 2, 4)
    np.testing.assert_equal(mask, expected_mask)
