import itertools as it
import numpy as np


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


def dMRA_tensor_index_translator(dim, k, condition=None):
    table = {}
    index = 0
    for tup in it.combinations_with_replacement(range(dim), k):
        if np.sum(tup) % dim == 0 and (condition is not None and condition(tup)):
            table[tup] = index
            table[index] = tup
            index += 1
    return table


def dMRA_index_to_order(dim, k, condition=None):
    table = {}
    order = 0
    for tup in it.combinations_with_replacement(range(dim), k):
        if np.sum(tup) % dim == 0 and (condition is not None and condition(tup)):
            table[tup] = order
            order += 1
    return table


def dMRA_order_to_index(dim, k, condition=None):
    return {value: key for key, value in dMRA_index_to_order(dim, k, condition=None)}


def dMRA_tensor_index_array(dim, k, condition=None):
    return np.fromiter(
        filter(
            lambda inds: ((np.sum(inds) % dim) == 0)
            and ((condition is None) or condition(inds)),
            it.combinations_with_replacement(range(dim), k),
        ),
        dtype=(np.uint, k),
    )
