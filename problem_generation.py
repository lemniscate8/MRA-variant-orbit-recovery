import json
import numpy as np
from scipy.special import comb
import polynomials as poly


# Computes the symmetric dMRA moment tensor in the Fourier basis
# Note: vector is indexed by weakly increasing subsequences
def computed_dMRA_moment(real_signal, k):
    dim = real_signal.size
    moment = np.zeros(comb(dim + k - 1, k, exact=True), dtype=complex)
    for ell in range(dim):
        rolled = np.roll(real_signal, ell)
        fproj = np.fft.fft(rolled)
        moment += np.squeeze(poly.symmetric_lift(fproj[:, None], k))
        fproj_rev = np.fft.fft(np.flip(rolled))
        moment += np.squeeze(poly.symmetric_lift(fproj_rev[:, None], k))
    moment /= 2 * dim
    return moment


# Computes the symmetric pMRA moment tensor in the Fourier basis
# Note: vector is indexed by weakly increasing subsequences
def computed_pMRA_moment(real_signal, k):
    dim = real_signal.size
    moment = np.zeros(comb((dim // 2) + k - 1, k, exact=True), dtype=complex)
    for ell in range(dim):
        rolled = np.roll(real_signal, ell)
        sym_proj = rolled + np.flip(rolled)
        fproj = np.fft.fft(sym_proj)
        moment += np.squeeze(poly.symmetric_lift(fproj[: (dim // 2), None], k))
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
        true_signal_loc = "solution/true_fourier_coefficients.csv"
        np.savetxt(true_signal_loc, fourier_signal, delimiter=",")
        print("Saved true signal to '", true_signal_loc, "'")

        # Save the initial 4 fourier coefficients to a file
        initial_loc = "input/initialization.csv"
        np.savetxt(initial_loc, fourier_signal[:: (dim // 4)], delimiter=",")
        print("Saved initial 4 coefficients to '", initial_loc, "'")

        # Save the moments to correct locations
        dMRA_moment = computed_dMRA_moment(real_signal, 3)
        np.savetxt("input/dMRA_3rd_moment.csv", dMRA_moment, delimiter=",")
        print("Third order dMRA moment to 'input/dMRA_3rd_moment.csv'")
        pMRA_moment = computed_pMRA_moment(real_signal, 3)
        np.savetxt("input/pMRA_3rd_moment.csv", pMRA_moment, delimiter=",")
        print("Third order pMRA moment to 'input/pMRA_3rd_moment.csv'")
