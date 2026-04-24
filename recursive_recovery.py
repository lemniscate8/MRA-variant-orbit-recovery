import numpy as np
import itertools as it
import scipy.sparse as sparse
import scipy.sparse.linalg as splg
import MRA_helpers as mh
import polynomials as poly
import recovery_helpers as rh
import pandas as pd
from scipy.special import comb


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
        arr[row_ind, col_index] = even_coefs[phase_index] * mh.sign(row_tup, dim)
    return sparse.csr_array(arr)


def construct_psuedoinverse(even_coefs, row_lut):
    dim = even_coefs.size
    num_rows = (dim + 1) * dim // 2
    num_cols = max(row_lut.values()) + 1
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
        col_ind2 = row_lut.get(t2, -1)
        if (col_ind1 == -1) or (col_ind2 == -1):
            continue
        if col_ind1 == col_ind2:
            val = even_coefs[ell1].conj() / (np.abs(even_coefs[ell1]) ** 2)
        else:
            val = even_coefs[ell1].conj() / (
                np.abs(even_coefs[ell1]) ** 2 + np.abs(even_coefs[ell2]) ** 2
            )
        arr[row_index, col_ind1] = val
        arr[row_index, col_ind2] = val
    csr_array = sparse.csr_array(arr)

    return csr_array


# Alternative method of constructing the M-matrix (plus one additional column)
# using our symmetric lift utility (see unit_tests where we validate this
# produces the same matrix)
def construct_M_matrix_alt(coef, col_map_method):
    null = construct_nullspace(coef[::2], col_map_method)
    odd_rank1 = poly.symmetric_lift(coef[1::2, None], 2)
    null_extend = sparse.hstack([odd_rank1, null], format="dok", dtype=null.dtype)
    lifted = poly.symmetric_lift(null_extend, 2)
    return poly.sym_matrix_minors_subspace(len(coef) // 2) @ lifted


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


# Subroutine to extract a particular solution in either the dMRA or pMRA case
def get_particular_solution(even_coefs, invars, invar_inds):
    row_lut = {}
    for ind, row in enumerate(invar_inds):
        row_lut[tuple(row)] = ind
    psuedo = construct_psuedoinverse(even_coefs, row_lut)
    proj = psuedo @ invars
    return proj


# Subroutine that recovers odd Fourier coefficients from the even ones and
# 3rd order dMRA invariants or pMRA invariants
def recover_odd_coefs(even_coefs, invars, invar_inds, null_lut, verbose=False):
    dim = even_coefs.size
    if verbose:
        print("Recovering odd coefficients at dimension ", 2 * dim)
    x_p = get_particular_solution(even_coefs, invars, invar_inds)
    x_p_norm = np.linalg.norm(x_p)
    x_p /= x_p_norm
    null = construct_nullspace(even_coefs, null_lut)
    null /= splg.norm(null, axis=0)
    ortho_basis = sparse.hstack((x_p[:, None], null), format="csr")
    # gram = (ortho_basis.T.conj() @ ortho_basis).toarray()
    # ident = np.eye(gram.shape[1])
    # print(ortho_basis.shape)
    # print("Is orthonormal? ", np.allclose(gram.flatten(), ident.flatten()))
    rec_mat, _, w_right = rh.top_vector_recovery_matrix_for(ortho_basis)
    _, s, vh = splg.svds(rec_mat, k=2, return_singular_vectors="vh", tol=0)
    if verbose:
        print("Top two singular values: ", s)
    weight_vec = vh[1, :].conj() / w_right
    weights, eigvals1 = rh.round_sym_to_rank1(weight_vec, ortho_basis.shape[1])
    if verbose:
        num_vecs = min(eigvals1.size, 5)
        abs_vals1 = np.abs(eigvals1)
        sel = np.argpartition(abs_vals1, -num_vecs)[(-num_vecs):]
        print(
            "Singular vector-as-matrix eigenvalues (largest 5):\n",
            np.abs(eigvals1[sel]),
        )
    normalized_weights = (weights / weights[0]) * x_p_norm
    odd_lift = ortho_basis @ normalized_weights
    odd_coefs, eigvals2 = rh.round_sym_to_rank1(odd_lift, dim)
    if verbose:
        num_vecs = min(eigvals2.size - 1, 5)
        abs_vals2 = np.abs(eigvals2)
        sel = np.argpartition(abs_vals2, -num_vecs)[(-num_vecs):]
        print("Planted matrix eigenvalues (largest 5):\n", eigvals2[sel])
    return odd_coefs


# A method that computes the dMRA invariants in the fourier basis
def theoretic_dMRA_invariants(fourier_signal, k):
    d = fourier_signal.size
    _, indices = mh.non_zero_dMRA_invariants(d, k)
    return np.real(np.prod(fourier_signal[indices], axis=1)), indices


# Computes the rescaled invariants one should theoretically have access to
# (this is a subset of the dMRA invariants)
def theoretic_pMRA_invariants(fourier_signal, k):
    d = fourier_signal.size
    _, indices = mh.non_zero_dMRA_invariants(d, k)
    sel = ~np.any((indices == (d // 2)), axis=1)
    print(sel)
    print(indices)
    indices = indices[sel, :]
    print(indices)
    return np.real(np.prod(fourier_signal[indices], axis=1)), indices


# Computes the pMRA moment tensor in a compressed format and in the Fourier
# basis
def computed_pMRA_moment(real_signal, k):
    dim = real_signal.size
    moment = np.zeros(comb(dim + k - 1, k, exact=True), dtype=complex)
    for ell in range(dim):
        rolled = np.roll(real_signal, ell)
        sym_proj = rolled + np.flip(rolled)
        fproj = np.fft.fft(sym_proj)
        moment += np.squeeze(poly.symmetric_lift(fproj[:, None], k))
    moment /= dim
    return moment


def extract_dMRA_invars_from_pMRA(dim, pMRA_moment):
    pass


def recursive_recover(
    dim, invars, invar_inds, null_lut, base_case_method, verbose=False
):
    even_coefs = np.zeros(dim // 2, dtype=complex)
    if dim > 8:
        sel = np.all((invar_inds % 2) == 0, axis=1)
        even_invars = invars[sel]
        even_invar_inds = invar_inds[sel, :] // 2
        even_coefs[:] = recursive_recover(
            dim // 2,
            even_invars,
            even_invar_inds,
            null_lut,
            base_case_method,
            verbose=verbose,
        )
    else:
        if verbose:
            print(
                "At dimension {} using base case method '{}'".format(
                    dim, base_case_method.__name__
                )
            )
        even_coefs[:] = base_case_method(invars)

    if verbose:
        print("Solving level ", dim)
    odd_coefs = recover_odd_coefs(
        even_coefs, invars, invar_inds, null_lut, verbose=verbose
    )
    all_coefs = np.zeros(dim, dtype=complex)
    all_coefs[::2] = even_coefs
    all_coefs[1::2] = odd_coefs
    return all_coefs


def cyclic_fdistance(f_signal1, f_signal2):
    dim = f_signal1.size
    steps = np.arange(dim)
    outer = np.outer(steps, steps)
    phases = np.exp(-2j * np.pi * outer / dim)
    shifted = phases * f_signal1[None, :]
    diffs = shifted - f_signal2[None, :]
    return np.min(np.linalg.norm(diffs, axis=1))


def dihedral_fdistance(f_signal1, f_signal2):
    cyc_dist = cyclic_fdistance(f_signal1, f_signal2)
    f_signal2_rev = np.flip(np.roll(f_signal2, -1))
    cyc_dist_rev = cyclic_fdistance(f_signal1, f_signal2_rev)
    return min(cyc_dist, cyc_dist_rev)


if __name__ == "__main__":

    inital_coefs = np.loadtxt("input/initialization.csv", dtype=complex, delimiter=",")

    # Return the first four Fourier coefficients as the base case
    def oracle_dMRA_base_case(invars):
        return inital_coefs

    # Assume that the dimension is 64
    signal_dim = 64

    # Load the dMRA moment stored in a .csv file
    dMRA_moment = np.loadtxt("input/dMRA_3rd_moment.csv", dtype=complex, delimiter=",")

    # Compress the invariants by omitting the values assumed to be zero
    dMRA_select, dMRA_invar_inds = mh.non_zero_dMRA_invariants(signal_dim, 3)
    print("Number of non-zero dMRA invariants: ", np.sum(dMRA_select))
    nonzero_dMRA_invars = dMRA_moment[dMRA_select]

    # Run the recovery algorithm for the dMRA invariants
    print("----- Starting recovery for dMRA model -----")
    recovered_signal_dMRA = recursive_recover(
        signal_dim,
        nonzero_dMRA_invars,
        dMRA_invar_inds,
        mh.dMRA_nullspace_col_lut,
        oracle_dMRA_base_case,
        verbose=True,
    )

    # Load the dMRA moment stored in a .csv file
    pMRA_moment = np.loadtxt("input/pMRA_3rd_moment.csv", dtype=complex, delimiter=",")

    # Extract the relevant invariants
    pMRA_select, pMRA_invar_inds, signs = mh.non_zero_pMRA_invariants(
        signal_dim // 2, 3
    )
    nonzero_pMRA_invars = pMRA_moment[pMRA_select]
    print("Number of non-zero dMRA invariants: ", np.sum(pMRA_select))

    # Reweight pMRA invariants to get equivalent dMRA ones
    reordering, weights, phase_shifts, pMRA_invar_inds, sel = (
        mh.pMRA_invariants_to_dMRA(signal_dim)
    )
    phases = np.exp(2j * np.pi * phase_shifts / signal_dim)
    normalized_pMRA_invars = (phases * nonzero_pMRA_invars / weights)[reordering]
    # adjusting_phases = phases[reordering]
    # selected_dMRA = nonzero_dMRA_invars[sel]
    # ratio = selected_dMRA / normalized_pMRA_invars
    # print(pMRA_invar_inds[off_values, :])
    # angles = np.astype(np.angle(ratio, deg=True) * signal_dim / 360, int) % signal_dim
    # diff = angles - adjusting_phases
    # data = np.hstack(
    #     (pMRA_invar_inds, angles[:, None], adjusting_phases[:, None], diff[:, None])
    # )
    # np.savetxt("phase_problems.csv", data, fmt="%i")

    # print("All close? ", np.allclose(selected_dMRA, normalized_pMRA_invars))

    # nonzero_pMRA_invars = nonzero_dMRA_invars[sel]

    # def oracle_pMRA_base_case(invars):
    #     inital_coefs[2] == 1
    #     return inital_coefs

    print("----- Starting recovery for pMRA model -----")
    recovered_signal_pMRA = recursive_recover(
        signal_dim,
        normalized_pMRA_invars,
        pMRA_invar_inds,
        mh.pMRA_nullspace_col_lut,
        oracle_dMRA_base_case,
        verbose=True,
    )

    # Load the true signal
    true_fourier = np.loadtxt(
        "solution/true_fourier_coefficients.csv", dtype=complex, delimiter=","
    )

    # Check how close dMRA recovery method got
    dist_dMRA = dihedral_fdistance(true_fourier, recovered_signal_dMRA)
    print("Recovery dihedral-invariant distance from 3rd dMRA moment:", dist_dMRA)

    # Check how close pMRA recovery method got
    dist_pMRA = dihedral_fdistance(true_fourier, recovered_signal_pMRA)
    print("Recovery dihedral-invariant distance from 3rd pMRA moment:", dist_pMRA)
