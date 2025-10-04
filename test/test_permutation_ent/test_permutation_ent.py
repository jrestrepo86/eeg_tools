"""
Permutation Entropy test battery
Run:
    python test_permutation_entropy.py

Assumes:
    from eeg_tools.entropy.permutation_entropy import permutation_entropy
"""

import math

import numpy as np

from eeg_tools.entropy.permutation_entropy import permutation_entropy


def check(name, value, target, tol_abs=None, tol_rel=None):
    """
    Basic assertion helper.
    - Use tol_abs for absolute bounds (|value - target| <= tol_abs)
    - Use tol_rel for relative bounds (|value - target| <= tol_rel * max(1, |target|))
    """
    if tol_abs is None and tol_rel is None:
        raise ValueError("Provide tol_abs or tol_rel")

    if tol_abs is None:
        tol_abs = (tol_rel or 0) * max(1.0, abs(target))

    ok = abs(value - target) <= tol_abs
    status = "OK " if ok else "FAIL"
    print(
        f"[{status}] {name}: value={value:.6f}, target={target:.6f}, tol_abs={tol_abs:.6f}"
    )
    if not ok:
        raise AssertionError(f"{name}: {value} not within {tol_abs} of {target}")


def check_bound(name, value, lower=None, upper=None):
    ok = True
    if lower is not None and not (value >= lower):
        ok = False
    if upper is not None and not (value <= upper):
        ok = False
    status = "OK " if ok else "FAIL"
    print(f"[{status}] {name}: value={value:.6f}, bounds=({lower}, {upper})")
    if not ok:
        raise AssertionError(f"{name}: {value} not in bounds ({lower},{upper})")


def main():
    rng = np.random.default_rng(12345)

    # 1) Constant / strictly monotone sequences → PE = 0 exactly (theoretical)
    n = 5000
    x_const = np.ones(n)
    x_inc = np.arange(n, dtype=float)
    x_dec = -np.arange(n, dtype=float)

    for emb_dim in (3, 4, 5):
        H_const = permutation_entropy(
            x_const, emb_dim=emb_dim, emb_lag=1, normalize=True
        )
        H_inc = permutation_entropy(x_inc, emb_dim=emb_dim, emb_lag=1, normalize=True)
        H_dec = permutation_entropy(x_dec, emb_dim=emb_dim, emb_lag=1, normalize=True)

        # Exact 0 is expected; allow tiny numerical noise
        check(f"Const PE (m={emb_dim})", H_const, 0.0, tol_abs=1e-12)
        check(f"Increasing PE (m={emb_dim})", H_inc, 0.0, tol_abs=1e-12)
        check(f"Decreasing PE (m={emb_dim})", H_dec, 0.0, tol_abs=1e-12)

    # 2) i.i.d. continuous white noise → PE ≈ 1 (normalized), i.e. uniform over m! patterns (asymptotic)
    #    We use a fairly long series so sampling error is small.
    n = 200_000
    wn = rng.standard_normal(n)

    for emb_dim in (3, 4, 5, 6):
        H = permutation_entropy(wn, emb_dim=emb_dim, emb_lag=1, normalize=True)
        # With this sample size, normalized PE should be extremely close to 1.
        # We accept small deviation due to sampling; 0.99 is a tight but reasonable threshold.
        check_bound(f"White-noise PE ~ 1 (m={emb_dim})", H, lower=0.99, upper=1.00001)

    # 3) White noise with larger lag → still ≈ 1 (independence across lagged samples)
    for tau in (2, 5, 10):
        H = permutation_entropy(wn, emb_dim=5, emb_lag=tau, normalize=True)
        check_bound(f"White-noise PE ~ 1 with lag {tau}", H, lower=0.99, upper=1.00001)

    # 4) Strongly correlated AR(1) process → PE < 1 (interpretation: more ordinal predictability)
    #    x_t = phi x_{t-1} + eps_t, |phi| < 1. For large phi, patterns concentrate, entropy drops.
    def ar1(phi=0.95, n=200_000, burn=1000):
        e = rng.standard_normal(n + burn)
        x = np.zeros(n + burn)
        for t in range(1, n + burn):
            x[t] = phi * x[t - 1] + e[t]
        return x[burn:]

    x_ar = ar1(phi=0.95)
    H_ar = permutation_entropy(x_ar, emb_dim=5, emb_lag=1, normalize=True)
    # Expect clearly below white-noise level; empirical bound ~0.7–0.9 depending on m
    check_bound("AR(1) phi=0.95 PE", H_ar, lower=0.6, upper=0.95)

    # 5) Periodic sequences
    # 5a) Periodic sawtooth with lag equal to period (τ = P) → PE ≈ 0
    #     Rationale: sampling at the same phase makes embedded vectors equal
    #     (ties broken by time), so they map to a single ordinal pattern.
    P = 7  # any small integer ≥ 2
    reps = 5000
    saw = np.tile(np.arange(P, dtype=float), reps)
    t = np.arange(saw.size, dtype=float)
    saw = saw + 1e-12 * t  # tiny monotone offset to avoid exact ties
    H_saw_tauP = permutation_entropy(saw, emb_dim=5, emb_lag=P, normalize=True)
    check_bound(
        "Periodic sawtooth PE with tau={P} (very low)",
        H_saw_tauP,
        lower=0.0,
        upper=0.02,
    )

    # 5b) Periodic sawtooth with τ = 1 but LARGE period → low (not zero) PE
    #     Making P ≫ m reduces how often windows straddle the discontinuity.
    P = 101  # large period compared to m=5
    reps = 2000
    saw_longP = np.tile(np.arange(P, dtype=float), reps)
    t = np.arange(saw_longP.size, dtype=float)
    saw_longP = saw_longP + 1e-12 * t
    H_saw_tau1 = permutation_entropy(saw_longP, emb_dim=5, emb_lag=1, normalize=True)
    # Expect a clearly small value (< ~0.1). If your implementation drops tie windows
    # or handles edges differently, you might see even smaller values.
    check_bound(
        "Periodic sawtooth PE with tau=1, large P (low)",
        H_saw_tau1,
        lower=0.0,
        upper=0.10,
    )

    # 6) Invariance under strictly monotone transforms
    base_sig = rng.normal(size=100_000)
    mono1 = 3.0 * base_sig + 2.0  # affine (monotone)
    mono2 = np.exp(base_sig)  # strictly increasing
    H0 = permutation_entropy(base_sig, emb_dim=5, emb_lag=1, normalize=True)
    H1 = permutation_entropy(mono1, emb_dim=5, emb_lag=1, normalize=True)
    H2 = permutation_entropy(mono2, emb_dim=5, emb_lag=1, normalize=True)
    # Values should match up to small Monte Carlo noise
    check("Monotone invariance (affine)", H1, H0, tol_abs=5e-3)
    check("Monotone invariance (exp)", H2, H0, tol_abs=5e-3)

    # 7) NaN handling: a patch of NaNs shouldn't crash; PE should be finite and reasonable
    x_nan = rng.normal(size=50_000)
    x_nan[1234:5678] = np.nan
    H_nan = permutation_entropy(x_nan, emb_dim=5, emb_lag=1, normalize=True)
    check_bound("NaN-robustness (finite)", H_nan, lower=0.0, upper=1.00001)

    # 8) Short-signal edge case: if the series is too short, some implementations return NaN.
    short = np.array([1.0, 2.0, 3.0, 4.0])
    try:
        H_short = permutation_entropy(short, emb_dim=5, emb_lag=1, normalize=True)
        # If reached, accept NaN; some variants may raise instead.
        if np.isnan(H_short):
            print("[OK ] Short-signal returns NaN")
        else:
            print(
                "[WARN] Short-signal returned finite value; check implementation choice."
            )
    except Exception as e:
        print(f"[OK ] Short-signal raised as expected: {e!r}")

    print("\nAll tests completed.")


if __name__ == "__main__":
    main()
