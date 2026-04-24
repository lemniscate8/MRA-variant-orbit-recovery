import numpy as np
import itertools as it
import scipy.sparse as sparse

# import scipy.sparse.linalg as splg
from scipy.special import comb, factorial


def symmetric_lift(basis, k):
    if k == 1:
        return basis
    nrow, ncol = basis.shape
    lrow = comb(nrow + k - 1, k, exact=True)
    lcol = comb(ncol + k - 1, k, exact=True)
    if sparse.issparse(basis):
        lifted_basis = sparse.csr_array((lrow, lcol), dtype=basis.dtype)
    else:
        lifted_basis = np.zeros(shape=(lrow, lcol), dtype=basis.dtype)
    row_iter = np.fromiter(
        it.combinations_with_replacement(range(nrow), k),
        dtype=np.dtype((np.int64, k)),
        count=lrow,
    )
    col_iter = np.fromiter(
        it.combinations_with_replacement(range(ncol), k),
        dtype=np.dtype((np.int64, k)),
        count=lcol,
    )
    for perm in it.permutations(range(k)):
        prod = 1
        for index in range(k):
            row_col_selection = basis[
                row_iter[:, index, None], col_iter[:, perm[index]]
            ]
            prod = row_col_selection * prod
        lifted_basis += prod
    return lifted_basis / factorial(k)


def multiset_weights(dim, k):
    lift_dim = comb(dim + k - 1, k, exact=True)
    return np.fromiter(
        map(
            num_unique_permutations_of, it.combinations_with_replacement(range(dim), k)
        ),
        dtype=float,
        count=lift_dim,
    )


def num_unique_permutations_of(tup):
    # Short cut if we're just doing a symmetric power.
    if len(tup) == 2:
        if tup[0] == tup[1]:
            return 1.0
        else:
            return 2.0
    mults = np.unique_counts(tup).counts
    sums = np.cumsum(mults)
    binoms = comb(sums[1:], mults[1:])
    return np.prod(binoms)


def minor_poly_lut(dim):
    sym_sym = it.combinations_with_replacement(
        it.combinations_with_replacement(range(dim), 2), 2
    )
    return {tup: ind for ind, tup in enumerate(sym_sym)}


def minor_poly_inv_lut(dim):
    sym_sym = it.combinations_with_replacement(
        it.combinations_with_replacement(range(dim), 2), 2
    )
    return {ind: tup for ind, tup in enumerate(sym_sym)}


def alt_corners(pair_pair):
    p0, p1 = pair_pair
    p0_new = (p0[0], p1[1]) if p0[0] < p1[1] else (p1[1], p0[0])
    p1_new = (p1[0], p0[1]) if p1[0] < p0[1] else (p0[1], p1[0])
    return (p0_new, p1_new) if p0_new < p1_new else (p1_new, p0_new)


def is_minor(pair_pair):
    return pair_pair < alt_corners(pair_pair)


def minor_corner_iter(dim):
    sym_sym = it.combinations_with_replacement(
        it.combinations_with_replacement(range(dim), 2), 2
    )
    return filter(is_minor, sym_sym)


def sym_matrix_minors_subspace(dim):
    minor_corners = minor_corner_iter(dim)
    lut = minor_poly_lut(dim)
    rows = []
    cols = []
    vals = []
    row_index = 0
    for mc in minor_corners:
        vals.extend([1, -1])
        rows.extend([row_index, row_index])
        cols.append(lut[mc])
        cols.append(lut[alt_corners(mc)])
        row_index += 1
    return sparse.csr_array(
        (vals, (rows, cols)), shape=(row_index, len(lut)), dtype=int
    )


def sym_matrix_antiminors_subspace(dim):
    col_lut = {
        key: value
        for value, key in enumerate(it.combinations_with_replacement(range(dim), 4))
    }
    rows = []
    vals = []
    for col_key in it.combinations_with_replacement(
        it.combinations_with_replacement(range(dim), 2), 2
    ):
        col_tuple = tuple(
            sorted((col_key[0][0], col_key[0][1], col_key[1][0], col_key[1][1]))
        )
        rows.append(col_lut[col_tuple])
    cols = np.arange(len(rows), dtype=int)
    vals = np.ones_like(rows, dtype=int)
    arr = sparse.csr_array((vals, (rows, cols)), shape=(len(col_lut), len(cols)))
    return arr
