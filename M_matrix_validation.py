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
    num_samples = 50**2
    z_score = 2.576  # For a 99% confidence interval
    dMRA_dims = [4, 8, 16]
    print("---------- dMRA cases ----------")
    for dim in dMRA_dims:
        samples = np.zeros(num_samples)
        print("Dim:", dim)
        count = 0
        while count < num_samples:
            real_signal = rng.normal(size=(2 * dim))
            signal = np.fft.fft(real_signal)
            M_matrix = construct_M_matrix(signal, dMRA_nullspace_col_lut)
            # Finding eigenvalues with sigma=0 puts method in shift-invert mode
            # so we find the smallest eigenvalues
            gram = M_matrix.T.conjugate() @ M_matrix
            val, vec = splg.eigsh(
                gram,
                k=1,
                which="LM",
                sigma=0,
                return_eigenvectors=True,
                maxiter=10000,
                tol=0,
                ncv=min(5, gram.shape[0]),
            )
            mvec = gram @ vec
            converged = np.allclose(mvec / vec, val)
            if converged:
                samples[count] = np.sqrt(val[0])
                count += 1
        mean = np.mean(samples)
        pm_val = z_score * np.std(samples) / np.sqrt(num_samples)
        print("\tM-shape:", M_matrix.shape)
        print("\tSmallest singular value: ", mean, "±", pm_val)

    pMRA_dims = [4, 8, 16]
    print("\n---------- pMRA cases ----------")
    for dim in pMRA_dims:
        samples = np.zeros(num_samples)
        print("Dim:", dim)
        count = 0
        while count < num_samples:
            real_signal = rng.normal(size=(2 * dim))
            signal = np.fft.fft(real_signal)
            signal[dim // 2] = 1
            M_matrix = construct_M_matrix(signal, pMRA_nullspace_col_lut)
            # Finding eigenvalues with sigma=0 puts method in shift-invert mode
            # so we find the smallest eigenvalues
            gram = M_matrix.T.conjugate() @ M_matrix
            val, vec = splg.eigsh(
                gram,
                k=1,
                which="LM",
                sigma=0,
                return_eigenvectors=True,
                maxiter=10000,
                tol=0,
                ncv=min(5, gram.shape[0]),
            )
            mvec = gram @ vec
            converged = np.allclose(mvec / vec, val)
            if converged:
                samples[count] = np.sqrt(val[0])
                count += 1
        mean = np.mean(samples)
        pm_val = z_score * np.std(samples) / np.sqrt(num_samples)
        print("\tM-shape:", M_matrix.shape)
        print("\tSmallest singular value: ", mean, "±", pm_val)
