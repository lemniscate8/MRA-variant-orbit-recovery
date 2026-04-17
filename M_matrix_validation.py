import numpy as np
import itertools as it
import scipy.linalg as sclg
import scipy.sparse as sparse
import scipy.sparse.linalg as splg
from MRA_helpers import *
from polynomials import *


def construct_M_matrix(signal, col_map_method):
    dim = len(signal)
    half_dim = dim // 2
    col_map = col_map_method(half_dim)
    num_columns = max(col_map.values()) + 1
    col_pair_map = {
        tup: (index - 1)
        for index, tup in enumerate(
            it.combinations_with_replacement(range(num_columns + 1), 2)
        )
    }
    minor_corners = minor_corner_iter(half_dim)
    rows = []
    cols = []
    vals = []
    row_index = 0
    for mc in minor_corners:
        oc = alt_corners(mc)
        for corner, sgn1 in zip([mc, oc], [1, -1]):
            # First four values
            # w1 = 1 if corner[0][0] == corner[0][1] else 2
            # w2 = 1 if corner[1][0] == corner[1][1] else 2
            for parity in range(2):
                sgn2 = sign(corner[1 - parity], half_dim)
                if sgn2 != 0:
                    rows.append(row_index)
                    col_ind = col_pair_map[(0, col_map[corner[1 - parity]] + 1)]
                    cols.append(col_ind)
                    val = (
                        signal[2 * corner[parity][0] + 1]
                        * signal[2 * corner[parity][1] + 1]
                        * signal[(2 * (np.sum(corner[1 - parity]) + 1)) % dim]
                        * sgn1
                        * sgn2
                    )
                    vals.append(val)
            # Last two values
            sgn2 = sign(corner[0], half_dim) * sign(corner[1], half_dim)
            if sgn2 != 0:
                rows.append(row_index)
                b0 = col_map[corner[0]] + 1
                b1 = col_map[corner[1]] + 1
                bpair = (b0, b1) if b0 < b1 else (b1, b0)
                weight = 2 if b0 == b1 else 1
                cols.append(col_pair_map[bpair])
                val = (
                    signal[(2 * (np.sum(corner[0]) + 1)) % dim]
                    * signal[(2 * (np.sum(corner[1]) + 1)) % dim]
                    * sgn1
                    * sgn2
                    * weight
                )
                vals.append(val)
        row_index += 1
    vals = np.array(vals) / 2
    arr = sparse.csr_array(
        (vals, (rows, cols)),
        shape=(row_index, len(col_pair_map) - 1),
        dtype=signal.dtype,
    )
    return arr


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dMRA_dims = [8, 16, 32]
    print("---------- dMRA cases ----------")
    for dim in dMRA_dims:
        print("Dim:", dim)
        real_signal = rng.normal(size=(dim))
        signal = np.fft.fft(real_signal)
        M_matrix = construct_M_matrix(signal, dMRA_nullspace_col_lut)
        print("\tM-shape:", M_matrix.shape)
        print("\tM-rank:", np.linalg.matrix_rank(M_matrix.toarray()))

    pMRA_dims = [8, 16, 32]
    print("\n---------- pMRA cases ----------")
    for dim in pMRA_dims:
        print("Dim:", dim)
        real_signal = rng.normal(size=(dim))
        signal = np.fft.fft(real_signal)
        signal[dim // 2] = 1
        M_matrix = construct_M_matrix(signal, pMRA_nullspace_col_lut)
        print("\tM-shape:", M_matrix.shape)
        print("\tM-rank:", np.linalg.matrix_rank(M_matrix.toarray()))
