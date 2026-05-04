import numpy as np
import pandas as pd
import plotly.express as ex
import problem_generation as pg
import MRA_helpers as mh
import recursive_recovery as rr


def pMRA_invariant_weight(index):
    num_zeros = np.sum(np.array(index) == 0)
    return 2 ** (num_zeros + (num_zeros != len(index)))


if __name__ == "__main__":
    # First, get a baseline error with no noise
    rng = np.random.default_rng(0)
    samples = 1000
    signal_dim = 64
    dMRA_coef_errors = np.zeros(shape=(samples, signal_dim), dtype=complex)
    pMRA_coef_errors = np.zeros(shape=(samples, signal_dim), dtype=complex)
    batch_size = 10
    for i in range(samples):
        if i % batch_size == 0:
            batch = i // batch_size
            print("Working on batch", 1 + 10 * batch, "-", 10 * (batch + 1))
        signal = rng.normal(size=signal_dim)
        fourier_signal = np.fft.fft(signal)
        dMRA_dict = mh.packaged_moments(pg.computed_dMRA_moment(signal, 3))
        dMRA_rec_signal = rr.recursive_recover(
            signal_dim, dMRA_dict, mh.dMRA_nullspace_col_lut, 4, rr.dMRA_fm_4d_base_case
        )
        d_dist, shift_dMRA_rec = rr.dihedral_invariant_distance(
            dMRA_rec_signal, fourier_signal
        )
        dMRA_coef_errors[i, :] = fourier_signal - shift_dMRA_rec
        # print(d_dist)

        pMRA_moment, _ = pg.computed_pMRA_moment(signal, 3)
        pMRA_dict = mh.packaged_moments(
            rr.reweight_3rd_order_pMRA_invariants(signal_dim, pMRA_moment)
        )
        pMRA_rec_signal = rr.recursive_recover(
            signal_dim, pMRA_dict, mh.pMRA_nullspace_col_lut, 8, rr.pMRA_fm_8d_base_case
        )
        # Don't forget to set Nyquist components to zero before computing distances
        fourier_signal[signal_dim // 2] = 0
        pMRA_rec_signal[signal_dim // 2] = 0
        p_dist, shift_pMRA_rec = rr.dihedral_invariant_distance(
            pMRA_rec_signal, fourier_signal
        )

        pMRA_coef_errors[i, :] = fourier_signal - shift_pMRA_rec
        # print(p_dist, np.linalg.norm(pMRA_coef_errors[i, :]))

    np.savetxt("numerical_experiments/dMRA_coeficient_errors.csv", dMRA_coef_errors)
    np.savetxt("numerical_experiments/pMRA_coeficient_errors.csv", pMRA_coef_errors)
