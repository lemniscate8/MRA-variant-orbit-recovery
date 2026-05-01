import numpy as np
import pandas as pd
import plotly.express as ex
import problem_generation as pg
import MRA_helpers as mh
import recursive_recovery as rr


def pMRA_invariant_weight(index):
    num_zeros = np.sum(np.array(index) == 0)
    return 2 ** (num_zeros + (num_zeros != len(index)))


def normalize_pMRA_invariants(pMRA_invariant_table):
    return {
        key: np.abs(value / pMRA_invariant_weight(key))
        for key, value in pMRA_invariant_table.items()
    }


if __name__ == "__main__":
    # First, get a baseline error with no noise
    rng = np.random.default_rng(0)
    samples = 1000
    signal_dim = 64
    dMRA_errors = np.zeros(shape=(samples, signal_dim))
    pMRA_errors = np.zeros(shape=(samples, signal_dim))
    for i in range(samples):
        signal = rng.normal(size=signal_dim)

        dMRA_moment = pg.computed_dMRA_moment(signal, 3)
        dMRA_select, dMRA_invar_inds = mh.non_zero_dMRA_invariants(signal_dim, 3)
        nonzero_dMRA_invars = dMRA_moment[dMRA_select]
        recovered_signal_dMRA = rr.recursive_recover(
            signal_dim,
            nonzero_dMRA_invars,
            dMRA_invar_inds,
            mh.dMRA_nullspace_col_lut,
            rr.oracle_base_case,
            verbose=True,
        )

        pMRA_moment = pg.computed_pMRA_moment(signal, 3)

        fourier_signal = np.fft.fft(signal)
