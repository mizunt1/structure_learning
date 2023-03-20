import numpy as np

from collections import namedtuple
from itertools import chain


NStepMask = namedtuple('NStepMask', ['rows', 'cols', 'start', 'end'])

def get_nstep_indices(n_step):
    """Get all the indices for all the multistep combinations, for efficient
    computations. As an example, suppose that we have a path of length 3
    (here the length of a path is the number of edges in the path) in our batch:

        s_1 <- s_2 <- s_3 <- s_4

    Note that the replay buffer returns a path in reverse order, as shown above.
    We want to compute the loss for all possible subsequence of this path. This
    means we want to compute the loss for these subsequences:

        s_1 <- s_2
        s_1 <- s_2 <- s_3
               s_2 <- s_3
        s_1 <- s_2 <- s_3 <- s_4
               s_2 <- s_3 <- s_4
                      s_3 <- s_4

    This function returns a number of useful elements to compute the loss for all
    the subsequences:

        - To compute the loss, we have to compute multiple sums over the subsquences
        (e.g. summing the difference in log-rewards to get the difference in log-rewards
        between the start and the end of the subsequence). We want to compute:

            x_1
            x_1 + x_2
                  x_2
            x_1 + x_2 + x_3
                  x_2 + x_3
                        x_3

        Here x_* correspond to a property of *an edge*, e.g. x_1 is the difference in
        log-rewards between s_2 and s_1 (s_1 <- s_2). This can be done using a sparse
        matrix multiplication:
           _         _
          | 1         |    _   _
          | 1   1     |   | x_1 |
          |     1     | * | x_2 |
          | 1   1   1 |   | x_3 |
          |     1   1 |   |_   _|
          |_        1_|
          

        The objects `rows` and `cols` correspond to the sparse representation of this
        sparse matrix (COO representation). If we have x the vector of x_*, we can
        compute this matrix multiplication in Jax using

            results = jnp.zeros((6,), dtype=x.dtype)
            results = results.at[rows].add(x[cols])

        - To compute the loss, we also need to have access to informations about the
        start and the end of a subsequence (e.g. to get P(s_f | s) for the end of
        the subsequence). We want to get, for example

            x_1
            x_1
                  x_2
            x_1
                  x_2
                        x_3

        Here x_* corresponds to a property of *an edge*, e.g. x_1 = P(s_f | s_1).
        Again, this can be done using a sparse matrix multiplication:
           _         _
          | 1         |    _   _
          | 1         |   | x_1 |
          |     1     | * | x_2 |
          | 1         |   | x_3 |
          |     1     |   |_   _|
          |_        1_|

        The object `end` corresponds to the sparse representation of this sparse
        matrix ([!] recall that the subsequences are stored in reverse order).
        Since we are guaranteed to have only 1 non-zero element in each row,
        we just store the column indices of these non-zero entries. If we have x
        the vector of x_*, we can compute the matrix multiplication in Jax as

            results = x[end]

        Similarly for `start` to get properties about the start of the subsequences.
    """
    block_lengths = np.arange(1, n_step + 1)
    start = np.fromiter(chain(*map(range, block_lengths)), dtype=np.int_)
    end = np.repeat(block_lengths, block_lengths)

    rows = np.repeat(np.arange(n_step * (n_step + 1) // 2), end - start)
    cols = np.fromiter(chain.from_iterable(
        range(*p) for p in zip(start, end)), dtype=np.int_)

    return NStepMask(
        rows=rows,
        cols=cols,
        # [!] Sequences are stored in reverse order, this is not a typo
        start=end - 1,
        end=start
    )


def get_nstep_mask(lengths, n_step):
    limits = lengths * (lengths + 1) // 2
    arange = np.arange(n_step * (n_step + 1) // 2)
    return (arange[:, None] < limits).astype(np.float32)


if __name__ == '__main__':
    masks = get_nstep_indices(n_step=3)
    print(masks)
