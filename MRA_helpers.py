import itertools as it
import numpy as np
from scipy.special import comb


def reflect(pair, dim):
    return tuple(sorted([(dim - val - 1) % dim for val in pair]))


def sign(pair, dim):
    ref_pair = reflect(pair, dim)
    if pair == ref_pair:
        return 0
    elif pair < ref_pair:
        return 1
    else:
        return -1


def reflect(pair, dim):
    return tuple(sorted([(dim - val - 1) % dim for val in pair]))


def sign(pair, dim):
    ref_pair = reflect(pair, dim)
    if pair == ref_pair:
        return 0
    elif pair < ref_pair:
        return 1
    else:
        return -1


def dMRA_nullspace_col_lut(dim):
    # First, beta function
    beta = {}
    col_index = 0
    for pair in it.combinations_with_replacement(range(dim), 2):
        if pair not in beta:
            ref = reflect(pair, dim)
            if pair != ref:
                beta[pair] = col_index
                beta[ref] = col_index
                col_index += 1
    return beta


def pMRA_nullspace_col_lut(dim):
    # Second, gamma function
    gamma = {}
    col_index = 0
    for pair in it.combinations_with_replacement(range(dim), 2):
        if pair not in gamma:
            ref = reflect(pair, dim)
            if pair != ref:
                gamma[pair] = col_index
                # Give these pairs a different column
                if ((pair[0] + pair[1] + 1) % (dim // 2)) == 0:
                    col_index += 1
                gamma[ref] = col_index
                col_index += 1
    return gamma


def non_zero_dMRA_invariants(dim, k, condition=None):
    tensor_inds = np.fromiter(
        it.combinations_with_replacement(range(dim), 3),
        dtype=(int, 3),
        count=comb(dim + 3 - 1, 3, exact=True),
    )
    sel = (np.sum(tensor_inds, axis=1) % dim) == 0
    return sel, tensor_inds[sel, :]


def non_zero_pMRA_invariants(tensor_dim, k):
    tensor_inds = np.fromiter(
        it.combinations_with_replacement(range(tensor_dim), 3),
        dtype=(int, 3),
        count=comb(tensor_dim + 3 - 1, 3, exact=True),
    )
    sel = np.zeros(comb(tensor_dim + 3 - 1, 3, exact=True), dtype=bool)
    signs = np.zeros_like(tensor_inds)
    for num in range(2 ** (k - 1)):
        sign_vec = 1 - 2 * ((num // 2 ** np.arange(k)) % 2)
        mod_zero = np.sum(sign_vec * tensor_inds, axis=1) % (2 * tensor_dim) == 0
        signs[mod_zero, :] = sign_vec[None, :]
        sel |= mod_zero
    return sel, tensor_inds[sel, :], signs[sel, :]


def pMRA_invariants_to_dMRA(dMRA_dim):
    pMRA_dim = dMRA_dim // 2
    lut = {}
    _, dMRA_inds = non_zero_dMRA_invariants(dMRA_dim, 3)
    sel = ~np.any(dMRA_inds == pMRA_dim, axis=1)
    no_nyq_dMRA_inds = dMRA_inds[sel, :]
    _, pMRA_inds, signs = non_zero_pMRA_invariants(pMRA_dim, 3)
    for i, row in enumerate(pMRA_inds):
        lut[tuple(row)] = i
    reorder = []
    for row in no_nyq_dMRA_inds:
        t = tuple(sorted(pMRA_dim - np.abs(pMRA_dim - row)))
        # t = tuple(sorted([r if (r < pMRA_dim) else dMRA_dim - r for r in row]))
        reorder.append(lut.get(t, -1))
    num_zeros = np.sum(pMRA_inds == 0, axis=1)
    weights = np.ones_like(num_zeros) * 2
    weights[num_zeros == 3] = 8
    weights[num_zeros == 1] = 4
    phase_adjust = np.sum(pMRA_inds * ((signs - 1) // 2), axis=1) % dMRA_dim

    return (
        np.array(reorder),
        np.array(weights),
        np.array(phase_adjust),
        no_nyq_dMRA_inds,
        sel,
    )


# def dMRA_tensor_index_translator(dim, k, condition=None):
#     table = {}
#     index = 0
#     for tup in it.combinations_with_replacement(range(dim), k):
#         if np.sum(tup) % dim == 0 and (condition is not None and condition(tup)):
#             table[tup] = index
#             table[index] = tup
#             index += 1
#     return table


# def dMRA_index_to_order(dim, k, condition=None):
#     table = {}
#     order = 0
#     for tup in it.combinations_with_replacement(range(dim), k):
#         if np.sum(tup) % dim == 0 and (condition is not None and condition(tup)):
#             table[tup] = order
#             order += 1
#     return table


# def dMRA_order_to_index(dim, k, condition=None):
#     return {value: key for key, value in dMRA_index_to_order(dim, k, condition=None)}


# def dMRA_tensor_index_array(dim, k, condition=None):
#     return np.fromiter(
#         filter(
#             lambda inds: ((np.sum(inds) % dim) == 0)
#             and ((condition is None) or condition(inds)),
#             it.combinations_with_replacement(range(dim), k),
#         ),
#         dtype=(np.uint, k),
#     )

# def pMRA_conversion(dim, k):
if __name__ == "__main__":
    np.zeros(4)
    print(np.zeros(4).dtype)
