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
def get_particular_solution(even_coefs, invariants):
    row_lut = {}
    invars = []
    for ind, (row, val) in enumerate(invariants.items()):
        row_lut[row] = ind
        invars.append(val)
    invar_vec = np.array(invars, dtype=even_coefs.dtype)
    psuedo = construct_psuedoinverse(even_coefs, row_lut)
    proj = psuedo @ invar_vec
    return proj


# Subroutine that recovers odd Fourier coefficients from the even ones and
# 3rd order dMRA invariants or pMRA invariants
def recover_odd_coefs(even_coefs, invariants, null_lut, verbose=False):
    dim = even_coefs.size
    if verbose:
        print("Recovering odd coefficients at dimension ", 2 * dim)
    x_p = get_particular_solution(even_coefs, invariants)
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
    return odd_coefs, np.max(s)


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


def recursive_recover(
    dim,
    invariants,
    null_lut,
    base_case_dim,
    base_case_method,
    forward_tol=1e-8,
    verbose=False,
):
    # even_coefs = np.zeros(dim // 2, dtype=complex)
    # sel = ((invar_inds % 2) == 0).all(axis=1)
    # even_invars = invars[sel]
    # even_invar_inds = invar_inds[sel, :] // 2
    even_invariants = {
        tuple(t // 2 for t in key): value
        for key, value in invariants.items()
        if (np.array(key) % 2 == 0).all()
    }
    if dim // 2 > base_case_dim:
        even_coefs = recursive_recover(
            dim // 2,
            even_invariants,
            null_lut,
            base_case_dim,
            base_case_method,
            forward_tol=forward_tol,
            verbose=verbose,
        )
    else:
        if verbose:
            print(
                "At dimension {} using base case method '{}'".format(
                    dim // 2, base_case_method.__name__
                )
            )

        even_coefs = base_case_method(even_invariants)
    if even_coefs.ndim == 2:
        num_sol = even_coefs.shape[1]
        corrs = np.zeros(num_sol)
        all_coefs = np.zeros((dim, num_sol), dtype=complex)
        all_coefs[::2, :] = even_coefs
        if verbose:
            print("Found ", num_sol, " solutions in dimension ", dim // 2)
            print(even_coefs.shape)
        for s in range(num_sol):
            if verbose:
                print("Extending solution ", s)
            odd_coefs, corr = recover_odd_coefs(
                even_coefs[:, s], invariants, null_lut, verbose=verbose
            )
            all_coefs[1::2, s] = odd_coefs
            corrs[s] = corr
        sel = (1 - corrs) < forward_tol
        if not sel.any():
            if verbose:
                print("No solutions can be extended. Picking best.")
            sel = np.argmax(corrs)
        return np.squeeze(all_coefs[:, sel])
    else:
        odd_coefs, s = recover_odd_coefs(
            even_coefs, invariants, null_lut, verbose=verbose
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
    # Reweight pMRA invariants to get equivalent dMRA ones
    reordering, weights, phase_shifts, pMRA_invar_inds, _ = mh.pMRA_invariants_to_dMRA(
        dim
    )
    phases = np.exp(2j * np.pi * phase_shifts / dim)
    normalized_pMRA_invars = (phases * pMRA_3rd_moment / weights)[reordering]
    return normalized_pMRA_invars, pMRA_invar_inds


# Frequency marching for dihedral MRA in 4 dimensions
# Invariants must at least contain the third moment
def dMRA_frequency_march_base_case(invars):
    x0_fallback = np.cbrt(np.abs(invars[(0, 0, 0)]))
    # See if invariants contain lower orders like the means and power spectrum
    x0 = invars.get((0,), x0_fallback)
    x1_mag_sq_fallback = invars[(0, 1, 3)] / x0
    x1_mag_sq = np.abs(invars.get((1, 3), x1_mag_sq_fallback))
    x2_fallback = invars[(0, 2, 2)] / x0
    x2 = np.sqrt(np.abs(invars.get((2, 2), x2_fallback)))
    x1_phase = 0.5 * np.acos(np.real(invars[(1, 1, 2)]) / (x1_mag_sq * x2))
    x1 = np.sqrt(x1_mag_sq) * np.exp(1.0j * x1_phase)
    x3 = np.sqrt(x1_mag_sq) * np.exp(-1.0j * x1_phase)
    return np.array(
        [x0, x1, x2, x3],
        dtype=complex,
    )

# Frequency marching for projected MRA in 8 dimensions
def pMRA_frequency_march_base_case(invars):
    # See if invariants contain lower orders like the means and power spectrum
    x0_fallback = np.cbrt(np.abs(invars[(0, 0, 0)]))
    x0 = invars.get((0,), x0_fallback)
    mags_sq = np.array(
        [
            np.abs(invars.get((i, 8 - i), invars[(0, i, 8 - i)] / x0))
            for i in range(1, 4)
        ]
    )
    print(mags_sq)
    tensor_inds = [(1, 1, 6), (1, 2, 5), (2, 3, 3)]
    ind_arr = 4 - np.abs(4 - np.array(tensor_inds)) - 1
    # print("Index array")
    # print(ind_arr)
    tensor_vals = np.array([invars[key] for key in tensor_inds])
    # print("Mag array")
    # print(np.sqrt(mags_sq[ind_arr]))

    tensor_cosines = tensor_vals / np.sqrt(np.prod(mags_sq[ind_arr], axis=1))
    phases = np.acos(tensor_cosines)
    signs = 1 - 2 * ((np.arange(4) // (2 ** np.arange(3))[:, None]) % 2)
    print(signs)
    possible_combos = phases[:, None] * signs
    print("Phase relations")
    print(phases)
    phase_combo_coefs = np.array([[2, -1, 0], [1, 1, -1], [0, 1, 2]])
    invert_combos = np.linalg.inv(phase_combo_coefs)
    print(phase_combo_coefs)
    print(invert_combos)
    phase_vals = invert_combos @ possible_combos
    print(phase_vals)
    solutions = np.zeros((8, 4), dtype=complex)
    solutions[0, :] = x0
    solutions[1:4, :] = np.exp(1.0j * phase_vals) * np.sqrt(mags_sq)[:, None]
    solutions[7:4:-1, :] = solutions[1:4, :].conj()
    solutions[4, :] = 1
    print(solutions.shape)
    print(solutions)
    print(np.abs(solutions))
    return solutions


if __name__ == "__main__":

    inital_coefs = np.loadtxt("input/initialization.csv", dtype=complex, delimiter=",")

    # Return the first four Fourier coefficients as the base case
    def oracle_base_case(invariants):
        return inital_coefs

    # Assume that the dimension is 64
    signal_dim = 64

    # Load the dMRA moment stored in a .csv file
    dMRA_moment = np.loadtxt("input/dMRA_3rd_moment.csv", dtype=complex, delimiter=",")

    # Compress the invariants by omitting the values assumed to be zero
    _, dMRA_invar_inds = mh.non_zero_dMRA_invariants(signal_dim, 3)
    dMRA_invariants = {
        tuple(row.tolist()): value for row, value in zip(dMRA_invar_inds, dMRA_moment)
    }

    # Run the recovery algorithm for the dMRA invariants
    print("----- Starting recovery for dMRA model -----")
    recovered_signal_dMRA = recursive_recover(
        signal_dim,
        dMRA_invariants,
        mh.dMRA_nullspace_col_lut,
        4,
        dMRA_frequency_march_base_case,
        verbose=True,
    )

    # Load the dMRA moment stored in a .csv file
    pMRA_moment = np.loadtxt("input/pMRA_3rd_moment.csv", dtype=complex, delimiter=",")

    # Use Lemma 52 to remove pesky phase factors and weights from projection step
    normalized_pMRA_moment, pMRA_invar_inds = normalize_3rd_order_pMRA_invariants(
        signal_dim, pMRA_moment
    )
    pMRA_invariants = {
        tuple(row.tolist()): value
        for row, value in zip(pMRA_invar_inds, normalized_pMRA_moment)
    }

    # Load the true signal
    true_fourier = np.loadtxt(
        "solution/true_fourier_coefficients.csv", dtype=complex, delimiter=","
    )

    pesky = np.angle(true_fourier[::8])[1:4]
    print("Angles should be")
    print(pesky)
    phase_combo_coefs = np.array([[2, -1, 0], [1, 1, -1], [0, 1, 2]])
    combos = phase_combo_coefs @ pesky
    print("True angle linear combos")
    print(combos)
    print("True angle combos mod 2pi")
    print(np.fmod(combos + 6 * np.pi, 2 * np.pi))
    print("Recover one of these")
    signs = 1 - 2 * ((np.arange(4) // (2 ** np.arange(3))[:, None]) % 2)
    rec = np.arccos(np.cos(combos)[:, None] * signs)
    print(rec)

    print("----- Starting recovery for pMRA model -----")
    recovered_signal_pMRA = recursive_recover(
        signal_dim,
        pMRA_invariants,
        mh.pMRA_nullspace_col_lut,
        8,
        pMRA_frequency_march_base_case,
        verbose=True,
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
