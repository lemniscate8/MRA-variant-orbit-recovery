import numpy as np


def error_cohorts(errors, base, exclude_middle=False):
    cohort = errors
    dim = errors.shape[1]
    while dim > base:
        print("\tCohort", dim)
        odd = cohort[:, 1::2]
        odd_l2_norm = np.linalg.norm(odd, axis=1)
        print("\tMedian of odds:", np.median(odd_l2_norm) / odd.shape[1])
        print("\tTotal median of odds:", np.median(np.abs(odd)))
        cohort = cohort[:, ::2]
        dim //= 2
    print("\tCohort", base)
    if exclude_middle:
        cohort = np.delete(cohort, base // 2, axis=1)
    final_l2_norms = np.linalg.norm(cohort, axis=1)
    print("\tMedian of odds:", np.median(final_l2_norms))
    print("\tTotal median of odds:", np.median(np.abs(cohort)))


if __name__ == "__main__":

    dMRA_coef_error = np.loadtxt(
        "numerical_experiments/dMRA_coeficient_errors.csv", dtype=complex
    )
    pMRA_coef_error = np.loadtxt(
        "numerical_experiments/pMRA_coeficient_errors.csv", dtype=complex
    )
    dMRA_l2_error = np.linalg.norm(dMRA_coef_error, axis=1)
    print("Median L2 recovery error from dMRA:", np.median(dMRA_l2_error))

    pMRA_coef_error_no_nyq = np.delete(pMRA_coef_error, 32, axis=1)
    pMRA_l2_error = np.linalg.norm(pMRA_coef_error_no_nyq, axis=1)
    print("Median L2 recovery error from pMRA:", np.median(pMRA_l2_error))

    print("Cohort errors for dMRA:")
    error_cohorts(dMRA_coef_error, 1)
    print("Cohort errors for pMRA:")
    error_cohorts(pMRA_coef_error, 2, exclude_middle=True)
