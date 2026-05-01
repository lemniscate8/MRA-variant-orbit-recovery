import unittest
import numpy as np
import scipy.sparse.linalg as splg
import MRA_helpers as mh
import polynomials as poly
import recursive_recovery as rr
import recovery_helpers as rec_help
import M_matrix_validation as mmv
import problem_generation as pg

# import warnings


class TestIntersect(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.unitary_test_set = [[3, 2, 2], [10, 6, 2], [4, 4, 2], [3, 2, 3]]
        self.plant_test_dims = [3, 6, 10]
        self.minors_test_dims = np.arange(2, 10, dtype=int)
        self.M_matrix_test_dims = [8, 16, 32]

    def generate_orthonormal_matrix(self, m, n):
        A = self.rng.normal(size=(m, n))
        Q, _ = np.linalg.qr(A)
        return Q

    def generate_unitary_matrix(self, m, n):
        A = self.rng.normal(size=(m, n)) + 1.0j * self.rng.normal(size=(m, n))
        Q, _ = np.linalg.qr(A)
        return Q

    def test_orthonormal_lift(self):
        for m, n, k in self.unitary_test_set:
            Q = self.generate_orthonormal_matrix(m, n)
            Q_lift = poly.symmetric_lift(Q, k)
            w_left = np.sqrt(poly.multiset_weights(m, k))
            w_right = np.sqrt(poly.multiset_weights(n, k))
            weighted_Q_lift = w_left[:, None] * Q_lift * w_right
            prod = weighted_Q_lift.T @ weighted_Q_lift
            # print(prod)
            I = np.identity(prod.shape[0])
            pvec = prod.flatten()
            Ivec = I.flatten()
            self.assertTrue(
                np.allclose(pvec, Ivec),
                "Symmetric lift of unitary matrix is not orthonormal for m={}, n={} and k={}".format(
                    m, n, k
                ),
            )

    def test_unitary_lift(self):
        for m, n, k in self.unitary_test_set:
            Q = self.generate_unitary_matrix(m, n)
            Q_lift = poly.symmetric_lift(Q, k)
            w_left = np.sqrt(poly.multiset_weights(m, k))
            w_right = np.sqrt(poly.multiset_weights(n, k))
            weighted_Q_lift = w_left[:, None] * Q_lift * w_right
            prod = weighted_Q_lift.conj().T @ weighted_Q_lift
            # print(prod)
            I = np.identity(prod.shape[0])
            pvec = prod.flatten()
            Ivec = I.flatten()
            self.assertTrue(
                np.allclose(pvec, Ivec),
                "Symmetric lift of unitary matrix is not unitary for m={}, n={} and k={}".format(
                    m, n, k
                ),
            )

    def get_max_allowed_subspace_dim(self, dim, s):
        return (int)(np.sqrt(dim**2 * (dim - 1) * (dim + 1) / 6 + 2 * s + 0.25) - 0.5)

    def assert_close_orthogonal(self, arr, msg):
        gram = arr.conj().T @ arr
        dim = gram.shape[0]
        rows, cols = np.triu_indices(dim, k=1)
        zz = np.zeros_like(rows)
        self.assertTrue(np.allclose(gram[rows, cols], zz), msg)

    def assert_close_orthonormal(self, arr, msg):
        gram = arr.conj().T @ arr
        gram_vec = gram.flatten()
        Ivec = np.identity(gram.shape[0]).flatten()
        self.assertTrue(np.allclose(gram_vec, Ivec), msg)

    def test_minor_antiminor_partition(self):
        for dim in self.minors_test_dims:
            lift_dim = dim * (dim + 1) // 2
            double_lift_dim = lift_dim * (lift_dim + 1) // 2
            minors = poly.sym_matrix_minors_subspace(dim)
            antiminors = poly.sym_matrix_antiminors_subspace(dim)
            # Verify the subspaces dimension partition the double lift
            # Verify that each subspace is an orthogonal basis at least
            self.assertEqual(
                minors.shape[0] + antiminors.shape[0],
                double_lift_dim,
                "Minors and antiminors do not partition space.",
            )
            self.assert_close_orthogonal(
                antiminors.T, "Antiminors matrix is not orthogonal"
            )

    # Test recovery in real case
    def test_symmetric_plant_real_recovery(self):
        for dim in self.plant_test_dims:
            signal = self.rng.normal(size=dim)
            rank_1 = poly.symmetric_lift(signal[:, None], 2)
            # Do I weight the lift or not?
            lift_dim = rank_1.shape[0]
            U_dim = self.get_max_allowed_subspace_dim(dim, 1) - 1
            nullspace = self.generate_orthonormal_matrix(lift_dim, U_dim)
            oproj = rank_1 - nullspace @ (nullspace.T @ rank_1)
            oproj /= np.linalg.norm(oproj)
            U = np.hstack((oproj, nullspace))
            Ulift = poly.symmetric_lift(U, 2)
            w_left = np.sqrt(poly.multiset_weights(lift_dim, 2))
            w_right = np.sqrt(poly.multiset_weights(U_dim + 1, 2))
            ortho_Ulift = w_left[:, None] * Ulift * w_right
            self.assert_close_orthonormal(ortho_Ulift, "Lift is not orthonormal.")
            antiminors = poly.sym_matrix_antiminors_subspace(dim)
            antiminors *= w_left
            antiminors /= splg.norm(antiminors, axis=1)[:, None]
            self.assert_close_orthonormal(ortho_Ulift, "Antiminors not orthonormal.")
            rec_matrix = antiminors @ ortho_Ulift
            _, s, vh = splg.svds(rec_matrix, k=2, return_singular_vectors="vh", tol=0)
            max_ind = np.argmax(s)
            self.assertAlmostEqual(
                s[max_ind], 1, msg="Top singular value is not close to 1."
            )
            gap = np.max(s) - np.min(s)
            self.assertGreaterEqual(gap, 0, "Trivial gap.")
            weights_vec = vh[max_ind, :] / w_right
            weights, eigvals = rec_help.round_sym_to_rank1(weights_vec, U_dim + 1)
            eig_mags1 = np.sort(np.abs(eigvals))
            self.assertTrue(
                np.allclose(eig_mags1[:-1], 0),
                msg="Weights vector is not close to rank-1.",
            )
            recovered = U @ weights
            signal_est, eigvals2 = rec_help.round_sym_to_rank1(recovered, dim)
            eig_mags2 = np.sort(np.abs(eigvals2))
            self.assertTrue(
                np.allclose(eig_mags2[:-1], 0),
                msg="Recovered solution is not close to rank-1.",
            )
            nsig = signal / np.linalg.norm(signal)
            nsig_est = signal_est / np.linalg.norm(signal_est)
            corr = np.abs(np.dot(nsig, nsig_est))
            self.assertAlmostEqual(
                corr,
                1,
                msg="Estimated signal not sufficiently correlated with planted.",
            )

    # Test recovery in complex case
    def test_symmetric_plant_complex_recovery(self):
        for dim in self.plant_test_dims:
            signal = self.rng.normal(size=dim) + 1.0j * self.rng.normal(size=dim)
            rank_1 = poly.symmetric_lift(signal[:, None], 2)
            # Do I weight the lift or not?
            lift_dim = rank_1.shape[0]
            U_dim = self.get_max_allowed_subspace_dim(dim, 1) - 1
            nullspace = self.generate_unitary_matrix(lift_dim, U_dim)
            oproj = rank_1 - nullspace @ (nullspace.conj().T @ rank_1)
            oproj /= np.linalg.norm(oproj)
            U = np.hstack((oproj, nullspace))
            Ulift = poly.symmetric_lift(U, 2)
            w_left = np.sqrt(poly.multiset_weights(lift_dim, 2))
            w_right = np.sqrt(poly.multiset_weights(U_dim + 1, 2))
            ortho_Ulift = w_left[:, None] * Ulift * w_right
            self.assert_close_orthonormal(ortho_Ulift, "Lift is not orthonormal.")
            antiminors = poly.sym_matrix_antiminors_subspace(dim)
            antiminors *= w_left
            antiminors /= splg.norm(antiminors, axis=1)[:, None]
            self.assert_close_orthonormal(ortho_Ulift, "Antiminors not orthonormal.")
            rec_matrix = antiminors @ ortho_Ulift
            _, s, vh = splg.svds(rec_matrix, k=2, return_singular_vectors="vh", tol=0)
            max_ind = np.argmax(s)
            self.assertAlmostEqual(
                s[max_ind], 1, msg="Top singular value is not close to 1."
            )
            gap = np.max(s) - np.min(s)
            self.assertGreaterEqual(gap, 0, "Trivial gap.")
            weights_vec = vh[max_ind, :].conj() / w_right
            weights, eigvals = rec_help.round_sym_to_rank1(weights_vec, U_dim + 1)
            eig_mags1 = np.sort(np.abs(eigvals))
            self.assertTrue(
                np.allclose(eig_mags1[:-1], 0),
                msg="Weights vector is not close to rank-1.",
            )
            recovered = U @ weights
            signal_est, eigvals2 = rec_help.round_sym_to_rank1(recovered, dim)
            eig_mags2 = np.sort(np.abs(eigvals2))
            self.assertTrue(
                np.allclose(eig_mags2[:-1], 0),
                msg="Recovered solution is not close to rank-1.",
            )
            nsig = signal / np.linalg.norm(signal)
            nsig_est = signal_est / np.linalg.norm(signal_est)
            # print(nsig / nsig_est)
            corr = np.abs(np.dot(nsig, nsig_est.conj()))
            self.assertAlmostEqual(
                corr,
                1,
                msg="Estimated signal not sufficiently correlated with planted.",
            )

    # Verify that two different methods for constructing the M-matrix are correct
    def test_equivalent_dMRA_M_matrix_constructions(self):
        for dim in self.M_matrix_test_dims:
            real_signal = self.rng.normal(size=dim)
            fourier_signal = np.fft.fft(real_signal)
            dMRA_lut = mh.dMRA_nullspace_col_lut
            M1 = mmv.construct_M_matrix(fourier_signal, dMRA_lut)
            M2 = rr.construct_M_matrix_alt(fourier_signal, dMRA_lut)[:, 1:]
            diff = M1 - M2
            distance = splg.norm(diff, ord="fro")
            self.assertAlmostEqual(
                distance, 0, msg="dMRA M-matrix constructions differ."
            )

    def test_equivalent_pMRA_M_matrix_constructions(self):
        for dim in self.M_matrix_test_dims:
            real_signal = self.rng.normal(size=dim)
            fourier_signal = np.fft.fft(real_signal)
            fourier_signal[dim // 2] = 1
            pMRA_lut = mh.pMRA_nullspace_col_lut
            M1 = mmv.construct_M_matrix(fourier_signal, pMRA_lut)
            M2 = rr.construct_M_matrix_alt(fourier_signal, pMRA_lut)[:, 1:]
            diff = M1 - M2
            distance = splg.norm(diff, ord="fro")
            self.assertAlmostEqual(
                distance, 0, msg="pMRA M-matrix constructions differ."
            )

    def test_dMRA_psuedoinverse(self):
        for dim in self.M_matrix_test_dims:
            real_signal = self.rng.normal(size=dim)
            coef = np.fft.fft(real_signal)
            invars, invar_inds = rr.theoretic_dMRA_invariants(coef, 3)
            invariants = {
                tuple(row.tolist()): value for row, value in zip(invar_inds, invars)
            }
            null = rr.construct_nullspace(coef[::2], mh.dMRA_nullspace_col_lut)
            null /= splg.norm(null, axis=0)
            odd_lift = np.squeeze(poly.symmetric_lift(coef[1::2, None], 2))
            part_sol1 = odd_lift - null @ (null.conjugate().T @ odd_lift)
            part_sol2 = rr.get_particular_solution(coef[::2], invariants)
            self.assertTrue(
                np.allclose(part_sol1, part_sol2),
                msg="dMRA psuedoinverse construction is incorrect.",
            )

    def test_pMRA_psuedoinverse(self):
        for dim in self.M_matrix_test_dims:
            real_signal = self.rng.normal(size=dim)
            coef = np.fft.fft(real_signal)
            invars, invar_inds = rr.theoretic_pMRA_invariants(coef, 3)
            invariants = {
                tuple(row.tolist()): value for row, value in zip(invar_inds, invars)
            }
            null = rr.construct_nullspace(coef[::2], mh.pMRA_nullspace_col_lut)
            null /= splg.norm(null, axis=0)
            odd_lift = np.squeeze(poly.symmetric_lift(coef[1::2, None], 2))
            part_sol1 = odd_lift - null @ (null.T.conj() @ odd_lift)
            part_sol2 = rr.get_particular_solution(coef[::2], invariants)
            ratio = part_sol1 / part_sol2
            print(part_sol1.shape)
            print(part_sol2.shape)
            print(ratio)
            self.assertTrue(
                np.allclose(part_sol1, part_sol2),
                msg="pMRA psuedoinverse construction is incorrect.",
            )

    def test_pMRA_converts_to_dMRA(self):
        for dim in self.M_matrix_test_dims:
            real_signal = self.rng.normal(size=dim)
            dMRA_moment = pg.computed_dMRA_moment(real_signal, 3)
            pMRA_moment = pg.computed_pMRA_moment(real_signal, 3)
            _, invar_inds = mh.non_zero_dMRA_invariants(dim, 3)
            sel = (invar_inds == dim // 2).any(axis=1)
            dMRA_moment_no_nyq = dMRA_moment[~sel]
            normalize_pMRA_moment, _ = rr.normalize_3rd_order_pMRA_invariants(
                dim, pMRA_moment
            )
            self.assertTrue(
                np.allclose(dMRA_moment_no_nyq, normalize_pMRA_moment),
                msg="Normalization failed in dimension " + str(dim),
            )

    def test_pMRA_frequency_march(self):
        real_signal = self.rng.normal(size=8)
        pMRA_moment = pg.computed_pMRA_moment(real_signal, 3)
        normalize_pMRA_moment, _ = rr.normalize_3rd_order_pMRA_invariants(
            8, pMRA_moment
        )


if __name__ == "__main__":
    unittest.main()
