import numpy as np
from scipy.special import comb
import MRA_helpers


# Computes the symmetric dMRA invariants in the Fourier basis
# Note: we only save the values expected to be non-zero
def computed_dMRA_moment(real_signal, k):
    dim = real_signal.size
    _, dMRA_inds = MRA_helpers.non_zero_dMRA_invariants(dim, k)
    moment = np.zeros(shape=dMRA_inds.shape[0], dtype=complex)
    for ell in range(dim):
        rolled = np.roll(real_signal, ell)
        fproj = np.fft.fft(rolled)
        moment += np.prod(fproj[dMRA_inds], axis=1)
        fproj_rev = np.fft.fft(np.flip(rolled))
        moment += np.prod(fproj_rev[dMRA_inds], axis=1)
    moment /= 2 * dim
    return moment


# Computes the symmetric pMRA moment tensor in the Fourier basis
# Note: we only save the values expected to be non-zero
def computed_pMRA_moment(real_signal, k):
    dim = real_signal.size
    _, pMRA_inds, _ = MRA_helpers.non_zero_pMRA_invariants(dim // 2, k)
    moment = np.zeros(pMRA_inds.shape[0], dtype=complex)
    for ell in range(dim):
        rolled = np.roll(real_signal, ell)
        sym_proj = rolled + np.flip(rolled)
        fproj = np.fft.fft(sym_proj)
        moment += np.prod(fproj[pMRA_inds], axis=1)
    moment /= dim
    return moment


if __name__ == "__main__":
    yesno = input(
        "(Y/N) Would you like to generate a new random test set (and replace the old set)? "
    )
    if (yesno == "Y") or (yesno == "y"):
        rng = np.random.default_rng()
        dim = 64
        real_signal = rng.normal(size=dim)
        fourier_signal = np.fft.fft(real_signal)

        # Save the true signal to a file
        true_signal_filename = "solution/true_fourier_coefficients.csv"
        np.savetxt(true_signal_filename, fourier_signal, delimiter=",")
        print("Saved true signal to '", true_signal_filename, "'")

        # Save the initial 4 fourier coefficients to a file
        initial_coefs_filename = "input/initialization.csv"
        np.savetxt(initial_coefs_filename, fourier_signal[:: (dim // 4)], delimiter=",")
        print("Saved initial 4 coefficients to '", initial_coefs_filename, "'")

        # Save the moments to correct locations
        dMRA_moment = computed_dMRA_moment(real_signal, 3)
        dMRA_moment_filename = "input/dMRA_3rd_moment.csv"
        np.savetxt(dMRA_moment_filename, dMRA_moment, delimiter=",")
        print("Saved dMRA 3rd moment to '", dMRA_moment_filename, "'")

        pMRA_moment = computed_pMRA_moment(real_signal, 3)
        pMRA_moment_filename = "input/pMRA_3rd_moment.csv"
        np.savetxt(pMRA_moment_filename, pMRA_moment, delimiter=",")
        print("Saved pMRA 3rd moment to '", pMRA_moment_filename, "'")
