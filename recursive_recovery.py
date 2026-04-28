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
# def singular_vector_recovery_matrix(coefs, col_map_method):
#     null = construct_nullspace(coefs[::2], col_map_method)
#     null /= splg.norm(null, axis=0)
#     odd_lift = poly.symmetric_lift(coefs[1::2, None])
#     part_sol = odd_lift - null @ (null.conjugate().T @ odd_lift)
#     part_sol_norm = np.linalg.norm(part_sol)
#     part_sol /= part_sol_norm
#     null_extend = sparse.hstack([part_sol, null], format="csr", dtype=null.dtype)
#     lifted = poly.symmetric_lift(null_extend, normalize=True)
#     lifted_csr = sparse.csr_array(lifted)
#     anti_minors = poly.sym_matrix_antiminors_subspace(coefs.size // 2)
#     return (anti_minors.transpose() @ lifted_csr), null_extend, part_sol_norm


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


def recursive_recover(
    dim, invars, invar_inds, null_lut, base_case_dim, base_case_method, verbose=False
):
    even_coefs = np.zeros(dim // 2, dtype=complex)
    sel = np.all((invar_inds % 2) == 0, axis=1)
    even_invars = invars[sel]
    even_invar_inds = invar_inds[sel, :] // 2
    if dim // 2 > base_case_dim:
        even_coefs[:] = recursive_recover(
            dim // 2,
            even_invars,
            even_invar_inds,
            null_lut,
            base_case_dim,
            base_case_method,
            verbose=verbose,
        )
    else:
        if verbose:
            print(
                "At dimension {} using base case method '{}'".format(
                    dim // 2, base_case_method.__name__
                )
            )
        even_coefs[:] = base_case_method(even_invars, even_invar_inds)
        print(even_coefs)

    if verbose:
        print("Solving level ", dim)
    odd_coefs = recover_odd_coefs(
        even_coefs, invars, invar_inds, null_lut, verbose=verbose
    )
    all_coefs = np.zeros(dim, dtype=complex)
    all_coefs[::2] = even_coefs
    all_coefs[1::2] = odd_coefs
    return all_coefs


# Compute the cyclic invariant distance from signals in the Fourier basis
def cyclic_invariant_distance(f_signal1, f_signal2):
    dim = f_signal1.size
    steps = np.arange(dim)
    outer = np.outer(steps, steps)
    phases = np.exp(-2j * np.pi * outer / dim)
    shifted = phases * f_signal1[None, :]
    diffs = shifted - f_signal2[None, :]
    norms = np.linalg.norm(diffs, axis=1)
    min_loc = np.argmin(norms)
    return norms[min_loc], shifted[min_loc, :]


# Compute the dihedral invariant distance from signals in the Fourier basis
def dihedral_invariant_distance(f_signal1, f_signal2):
    cyc_dist, orbit_element = cyclic_invariant_distance(f_signal1, f_signal2)
    f_signal2_rev = np.flip(np.roll(f_signal2, -1))
    cyc_dist_rev, orbit_element_rev = cyclic_invariant_distance(
        f_signal1, f_signal2_rev
    )
    if cyc_dist < cyc_dist_rev:
        return cyc_dist, orbit_element
    else:
        return cyc_dist_rev, orbit_element_rev


# A method using the computations from Lemma 52 to get the same invariants
# as appear in the 3rd order dMRA moment from the entries of the pMRA moment
def normalize_3rd_order_pMRA_invariants(dim, pMRA_3rd_moment):
    # Extract the relevant invariants which should be non-zero
    pMRA_select, pMRA_invar_inds, _ = mh.non_zero_pMRA_invariants(dim // 2, 3)
    nonzero_pMRA_invars = pMRA_3rd_moment[pMRA_select]

    # Reweight pMRA invariants to get equivalent dMRA ones
    reordering, weights, phase_shifts, pMRA_invar_inds, _ = mh.pMRA_invariants_to_dMRA(
        dim
    )

    phases = np.exp(2j * np.pi * phase_shifts / signal_dim)
    normalized_pMRA_invars = (phases * nonzero_pMRA_invars / weights)[reordering]
    return normalized_pMRA_invars, pMRA_invar_inds


# Frequency marching for dihedral MRA in down at 4 dimensions
# The 5 invariants should correspond to indices
# [[0 0 0]
#  [0 1 3]
#  [0 2 2]
#  [1 1 2]
#  [2 3 3]]
def dMRA_frequency_march_base_case(invars, invar_inds):
    x0 = np.cbrt(np.abs(invars[0]))
    print("x0: ", x0)
    x1_mag_sq = np.abs(invars[1] / x0)
    x2 = np.sqrt(np.abs(invars[2] / x0))
    x1_phase = 0.5 * np.acos(invars[3] / (x2 * x1_mag_sq))
    x1 = np.sqrt(x1_mag_sq) * np.exp(1.0j * x1_phase)
    x3 = np.sqrt(x1_mag_sq) * np.exp(-1.0j * x1_phase)
    return np.array(
        [x0, x1, x2, x3],
        dtype=complex,
    )


if __name__ == "__main__":

    inital_coefs = np.loadtxt("input/initialization.csv", dtype=complex, delimiter=",")

    # Return the first four Fourier coefficients as the base case
    def oracle_base_case(invars, invar_inds):
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
        4,
        dMRA_frequency_march_base_case,
        verbose=True,
    )

    # Load the dMRA moment stored in a .csv file
    pMRA_moment = np.loadtxt("input/pMRA_3rd_moment.csv", dtype=complex, delimiter=",")

    # Use Lemma 52 to remove pesky phase factors and weights from projection step
    normalized_pMRA_invars, pMRA_invar_inds = normalize_3rd_order_pMRA_invariants(
        signal_dim, pMRA_moment
    )

    print("----- Starting recovery for pMRA model -----")
    recovered_signal_pMRA = recursive_recover(
        signal_dim,
        normalized_pMRA_invars,
        pMRA_invar_inds,
        mh.pMRA_nullspace_col_lut,
        4,
        oracle_base_case,
        verbose=True,
    )

    # Load the true signal
    true_fourier = np.loadtxt(
        "solution/true_fourier_coefficients.csv", dtype=complex, delimiter=","
    )

    # Check how close dMRA recovery method got
    dist_dMRA, _ = dihedral_invariant_distance(true_fourier, recovered_signal_dMRA)
    print("Recovery dihedral-invariant distance from dMRA 3rd moment:", dist_dMRA)

    # Check how close pMRA recovery method got
    true_fourier_no_nyq = true_fourier.copy()
    true_fourier_no_nyq[signal_dim // 2] = 0
    recovered_signal_pMRA[signal_dim // 2] = 0

    dist_pMRA, _ = dihedral_invariant_distance(
        true_fourier_no_nyq, recovered_signal_pMRA
    )
    print("Recovery dihedral-invariant distance from pMRA 3rd moment:", dist_pMRA)
