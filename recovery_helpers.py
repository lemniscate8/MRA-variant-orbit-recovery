import numpy as np
import polynomials as poly
import scipy.sparse.linalg as splg
import warnings


def get_base_dimension(lift_dim):
    dim = int(np.round(np.sqrt(2 * lift_dim + 0.25) - 0.5))
    if lift_dim != dim * (dim + 1) // 2:
        warnings.warn("Dimension is not that of a 2nd order lift.", RuntimeWarning)
    return dim


def orthonormal_lift(ortho_basis, k):
    rows, cols = ortho_basis.shape
    raw_lift = poly.symmetric_lift(ortho_basis, k)
    w_left = np.sqrt(poly.multiset_weights(rows, k))
    w_right = np.sqrt(poly.multiset_weights(cols, k))
    ortho_lift = w_left[:, None] * raw_lift * w_right
    return ortho_lift, w_left, w_right


def kernel_recovery_matrix_for(basis):
    dim = get_base_dimension(basis.shape[0])
    return poly.sym_matrix_minors_subspace(dim) @ poly.symmetric_lift(basis, 2)


def top_vector_recovery_matrix_for(ortho_basis):
    dim = get_base_dimension(ortho_basis.shape[0])
    ortho_lift, w_left, w_right = orthonormal_lift(ortho_basis, 2)
    antiminors_subspace = poly.sym_matrix_antiminors_subspace(dim)
    antiminors_subspace *= w_left
    antiminors_subspace /= splg.norm(antiminors_subspace, axis=1)[:, None]
    rec_mat = antiminors_subspace @ ortho_lift
    return rec_mat, w_left, w_right


def round_sym_to_rank1(ut_vals, matrix_dim):
    mat = np.zeros((matrix_dim, matrix_dim), dtype=ut_vals.dtype)
    rows, cols = np.triu_indices_from(mat)
    mat[rows, cols] = ut_vals
    mat[cols, rows] = ut_vals
    vals, vecs = np.linalg.eig(mat)
    max_loc = np.argmax(np.abs(vals))
    # weight = np.sqrt(np.linalg.norm(mat, ord="fro"))
    max_vec = vecs[:, max_loc]
    mat_norm = np.linalg.norm(mat, ord="fro")
    phase = 1
    if ut_vals.dtype == complex:
        phase_sq = max_vec.conj() @ (mat / mat_norm @ max_vec.conj())
        phase = np.sqrt(phase_sq)
    return phase * np.sqrt(mat_norm) * max_vec, vals
