import numpy as np
import itertools as it
import scipy.sparse as sparse
import scipy.sparse.linalg as splg
from MRA_helpers import *
import polynomials as poly
import recovery_helpers as rh
import pandas as pd


def construct_nullspace(even_coefs, lut_method):
    dim = len(even_coefs)
    column_lut = lut_method(dim)
    num_cols = max(column_lut.values()) + 1
    num_rows = (dim + 1) * dim // 2
    arr = sparse.dok_array((num_rows, num_cols), dtype=even_coefs.dtype)
    for row_ind, row_tup in enumerate(it.combinations_with_replacement(range(dim), 2)):
        col_index = column_lut.get(row_tup, -1)
        if col_index < 0:
            continue
        phase_index = (row_tup[0] + row_tup[1] + 1) % dim
        arr[row_ind, col_index] = even_coefs[phase_index] * sign(row_tup, dim)
    return sparse.csr_array(arr)


def construct_psuedoinverse(even_coefs, row_lut):
    dim = even_coefs.size
    num_rows = (dim + 1) * dim // 2
    num_cols = max(row_lut.values()) + 1
    # print("Max cols:", num_cols)
    arr = sparse.dok_array((num_rows, num_cols), dtype=even_coefs.dtype)
    for row_index, row_tuple in enumerate(
        it.combinations_with_replacement(range(dim), 2)
    ):
        j, k = row_tuple
        ell1 = (-(j + k + 1)) % dim
        ell2 = (j + k + 1) % dim
        t1 = tuple(sorted([2 * j + 1, 2 * k + 1, 2 * ell1]))
        t2 = tuple(
            sorted([(-2 * j - 1) % (2 * dim), (-2 * k - 1) % (2 * dim), 2 * ell2])
        )
        col_ind1 = row_lut.get(t1, -1)
        col_ind2 = row_lut.get(t1, -1)
        if (col_ind1 == -1) or (col_ind2 == -1):
            continue
        if col_ind1 == col_ind2:
            val = even_coefs[ell1].conjugate() / (np.abs(even_coefs[ell1]) ** 2)
        else:
            val = even_coefs[ell1].conjugate() / (
                np.abs(even_coefs[ell1]) ** 2 + np.abs(even_coefs[ell2]) ** 2
            )
        # Correct normalization in our strange weighting
        # if j != k:
        #     val *= np.sqrt(2)
        # print(j, k, t1, t2, ell1, ell2)
        arr[row_index, col_ind1] = val
        arr[row_index, col_ind2] = val
    csr_array = sparse.csr_array(arr)

    return csr_array


# Alternative method of constructing the M-matrix (plus one additional column)
# using our symmetric lift utility (see unit_tests where we validate this is
# the same)
def construct_M_matrix_alt(coef, col_map_method):
    null = construct_nullspace(coef[::2], col_map_method)
    odd_rank1 = poly.symmetric_lift(coef[1::2, None], 2)
    null_extend = sparse.hstack([odd_rank1, null], format="dok", dtype=null.dtype)
    lifted = poly.symmetric_lift(null_extend, 2)
    return poly.sym_matrix_minors_subspace(len(coef) // 2) @ lifted


# def kernel_recovery_matrix(coef, col_map_method):
#     null = construct_nullspace(coef[::2], col_map_method)
#     null_csr = sparse.csr_array(null)
#     null_csr /= splg.norm(null_csr, axis=0)
#     odd_rank1 = poly.symmetric_lift(coef[1::2, None])
#     part_sol = odd_rank1 - null_csr @ (null_csr.conjugate().T @ odd_rank1)
#     part_sol /= np.linalg.norm(part_sol)
#     # sol_lift = symmetric_lift(part_sol[:, None])
#     # print(odd_lift.shape)
#     # odd_lift /= odd_lift / np.linalg.norm(odd_lift, axis=0)
#     null_extend = sparse.hstack([part_sol, null_csr], format="dok", dtype=null.dtype)
#     lifted = poly.symmetric_lift(null_extend)
#     lifted_csr = sparse.csr_array(lifted)
#     minors = poly.sym_matrix_minors_subspace(len(coef) // 2)
#     return minors @ lifted_csr


# Analog of the M-matrix except solution is the top singular vector
def singular_vector_recovery_matrix(coefs, col_map_method):
    null = construct_nullspace(coefs[::2], col_map_method)
    null /= splg.norm(null, axis=0)
    odd_lift = poly.symmetric_lift(coefs[1::2, None])
    part_sol = odd_lift - null @ (null.conjugate().T @ odd_lift)
    part_sol_norm = np.linalg.norm(part_sol)
    part_sol /= part_sol_norm
    # sol_lift = symmetric_lift(part_sol[:, None])
    # print(odd_lift.shape)
    # odd_lift /= odd_lift / np.linalg.norm(odd_lift, axis=0)
    null_extend = sparse.hstack([part_sol, null], format="csr", dtype=null.dtype)

    lifted = poly.symmetric_lift(null_extend, normalize=True)
    lifted_csr = sparse.csr_array(lifted)
    # lifted_csr /= splg.norm(lifted_csr, axis=0)
    # minors = construct_minors_matrix(len(coef) // 2)
    # return minors @ lifted_csr
    anti_minors = poly.sym_matrix_antiminors_subspace(coefs.size // 2)
    return (anti_minors.transpose() @ lifted_csr), null_extend, part_sol_norm


def get_particular_dMRA_solution(even_coefs, invars, invar_inds):
    row_lut = {}
    for ind, row in enumerate(invar_inds):
        row_lut[tuple(row)] = ind
    psuedo = construct_psuedoinverse(even_coefs, row_lut)
    proj = psuedo @ invars
    return proj


# def is_close_orthonormal(arr):
#     gram = arr.conj().T @ arr
#     gram_vec = gram.toarray().flatten()
#     Ivec = np.identity(gram.shape[0]).flatten()
#     print("Is orthonormal? ", np.allclose(gram_vec, Ivec))


def dMRA_recover_odd_coefs(even_coefs, invars, invar_inds, verbose=False):
    dim = even_coefs.size
    proj = get_particular_dMRA_solution(even_coefs, invars, invar_inds)
    proj_mag = np.linalg.norm(proj)
    proj /= proj_mag
    null = construct_nullspace(even_coefs, dMRA_nullspace_col_lut)
    null /= splg.norm(null, axis=0)
    ortho_basis = sparse.hstack((proj[:, None], null), format="csr")
    rec_mat, _, w_right = rh.top_vector_recovery_matrix_for(ortho_basis)
    _, s, vh = splg.svds(rec_mat, k=2, return_singular_vectors="vh", tol=0)
    if verbose:
        print("Top two singular values: ", s)
    weight_vec = vh[1, :].conj() / w_right
    weights, eigvals1 = rh.round_sym_to_rank1(weight_vec, ortho_basis.shape[1])
    print(np.abs(eigvals1))
    normalized_weights = weights * proj_mag / weights[0]
    odd_lift = ortho_basis @ normalized_weights
    odd_coefs, eigvals2 = rh.round_sym_to_rank1(odd_lift, dim)
    print(np.abs(eigvals2))
    return odd_coefs


# def test_normalizations(dim, rng):
#     vec = rng.normal(size=dim)
#     vec /= np.linalg.norm(vec)
#     vlift = symmetric_lift(vec[:, None], normalize=True)
#     print("1st lift norm:", np.linalg.norm(vlift))
#     vlift2 = symmetric_lift(vlift, normalize=True)
#     print("2nd lift  norm:", np.linalg.norm(vlift))
#     anti_minors = construct_anti_minors_matrix(dim, normalize=True)
#     extent = anti_minors.transpose() @ vlift2
#     print("extent norm:", np.linalg.norm(extent))


def computed_dMRA_moment(fourier_signal, k, condition=None):
    pass


def theoretic_dMRA_invariants(fourier_signal, k, condition=None):
    d = fourier_signal.size
    indices = dMRA_tensor_index_array(d, k, condition)
    return np.real(np.prod(fourier_signal[indices], axis=1)), indices


def computed_pMRA_moment(fourier_signal, k, condition=None):
    pass


def extract_dMRA_invars_from_pMRA(pMRA_invars, indices):
    pass


# def practice_noiseless_dMRA_invariants(fourier_signal, k):
#     for


def noiseless_pMRA_invariants(signal, k):
    pass


# def colspan_A_dMRA(even_coefs, invariant_lut):
#     d = even_coefs.size

#     for j,k in it.combinations_with_replacement(range(d)):


# def colspan_A_pMRA(dim):
#     pass


def compare_dMRA_projections(coef, invars, invar_inds):
    null = construct_nullspace(coef[::2], dMRA_nullspace_col_lut)
    null /= splg.norm(null, axis=0)
    odd_lift = poly.symmetric_lift(coef[1::2, None], 2)
    part_sol1 = odd_lift - null @ (null.conjugate().T @ odd_lift)
    part_sol2 = get_particular_dMRA_solution(coef[::2], invars, invar_inds)
    return np.squeeze(part_sol1), np.squeeze(part_sol2)


if __name__ == "__main__":
    rng = np.random.default_rng()
    dim = 8
    real_signal = rng.normal(size=(dim))
    signal = np.fft.fft(real_signal)
    invars, invar_inds = theoretic_dMRA_invariants(signal, 3)
    odd_coefs = dMRA_recover_odd_coefs(signal[::2], invars, invar_inds, verbose=True)
    true_odd = signal[1::2]
    ratio = odd_coefs / true_odd
    print(np.abs(ratio))

    # print(signal[1::2])
    # print(odd_coefs)
    # print(signal[1::2] / odd_coefs)
    # cheesed_recovery(signal, dMRA_nullspace_col_lut)

    # p1, p2 = compare_dMRA_projections(signal, invars, invar_inds)
    # print(p1 / p2)
    # print(np.allclose(p1 / p2, 1))

    # arr = dMRA_tensor_index_array(
    #     dim,
    #     3,
    #     condition=(lambda inds: np.sum((ind % 2 for ind in inds)) == 2),
    # )
    # print(arr.shape)
    # print(arr)

    # ----- Kernel method works -----
    # mat = kernel_recovery_matrix(signal, dMRA_nullspace_col_lut)
    # print(mat.shape)
    # print(np.linalg.matrix_rank(mat.toarray()))

    # ----- Singular value method -----
    # sp_arr, lift, null_extend = singular_vector_recovery_matrix(
    #     signal, dMRA_nullspace_col_lut
    # )

    # signal[dim // 2] = 1
    # sp_arr, lift, null_extend = singular_vector_recovery_matrix(
    #     signal, pMRA_nullspace_col_lut
    # )
    # print("Shape:", sp_arr.shape)
    # print("Footprint:", np.prod(sp_arr.shape))
    # print("Non-zero entries:", sp_arr.nnz)

    # num_vals = 3
    # _, s, vh = splg.svds(sp_arr, k=num_vals, return_singular_vectors="vh", tol=0)
    # # s = np.linalg.svdvals(sp_arr.toarray())
    # order = np.argsort(-np.abs(s))
    # print("Vals:", s[order][0:num_vals])
    # print("Angles:", 180 * np.acos(np.clip(s[order][0:num_vals], 0, 1)) / np.pi)

    # arr = sp_arr.toarray()
    # svd_vals = np.linalg.svdvals(arr)
    # print("Vals:", svd_vals[:10])
    # print("Angles:", 180 * np.arccos(np.clip(svd_vals[:10], -1, 1)) / np.pi)

    # test_normalizations(dim, rng)

    # eye = sparse.dok_array(lift.conjugate().T @ lift)
    # eye.eliminate_zeros()
    # print(eye)
    # arr = (lift.conj().T @ lift).toarray()
    # np.savetxt("test.csv", np.abs(arr), fmt="%.3f")

    # minors = construct_minors_matrix(dim)
    # anti_minors = construct_anti_minors_matrix(dim, normalize=True)
    # prod = minors @ anti_minors
    # norm = splg.norm(prod, ord="fro")
    # print(norm)
    # print(arr)

    # mat = construct_anti_minors_matrix(4, normalize=True)
    # np.savetxt("test.csv", mat.toarray(), fmt="%.1f")
