import math
from scipy.stats import distributions
from collections import namedtuple
import numpy as np
    
def ks_2samp(cdf1, cdf2, n1, n2, alternative='two-sided', mode='auto'):
    
    KstestResult = namedtuple('KstestResult', ('statistic', 'pvalue'))
    
    if mode not in ['auto', 'exact', 'asymp']:
        raise ValueError(f'Invalid value for mode: {mode}')
    alternative = {'t': 'two-sided', 'g': 'greater', 'l': 'less'}.get(
       alternative.lower()[0], alternative)
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError(f'Invalid value for alternative: {alternative}')
    MAX_AUTO_N = 10000  # 'auto' will attempt to be exact if n1,n2 <= MAX_AUTO_N
#     if np.ma.is_masked(data1):
#         data1 = data1.compressed()
#     if np.ma.is_masked(data2):
#         data2 = data2.compressed()
#     data1 = np.sort(data1)
#     data2 = np.sort(data2)
#     n1 = data1.shape[0]
#     n2 = data2.shape[0]
#     if min(n1, n2) == 0:
#         raise ValueError('Data passed to ks_2samp must not be empty')

#     data_all = np.concatenate([data1, data2])
#     # using searchsorted solves equal data problem
#     cdf1 = np.searchsorted(data1, data_all, side='right') / n1
#     cdf2 = np.searchsorted(data2, data_all, side='right') / n2
    cddiffs = cdf1 - cdf2
    # Ensure sign of minS is not negative.
    minS = np.clip(-np.min(cddiffs), 0, 1)
    maxS = np.max(cddiffs)
    alt2Dvalue = {'less': minS, 'greater': maxS, 'two-sided': max(minS, maxS)}
    d = alt2Dvalue[alternative]
    g = math.gcd(n1, n2)
    n1g = n1 // g
    n2g = n2 // g
    prob = -np.inf
    original_mode = mode
    if mode == 'auto':
        mode = 'exact' if max(n1, n2) <= MAX_AUTO_N else 'asymp'
    elif mode == 'exact':
        # If lcm(n1, n2) is too big, switch from exact to asymp
        if n1g >= np.iinfo(np.int32).max / n2g:
            mode = 'asymp'
            warnings.warn(
                f"Exact ks_2samp calculation not possible with samples sizes "
                f"{n1} and {n2}. Switching to 'asymp'.", RuntimeWarning)

    if mode == 'exact':
        success, d, prob = _attempt_exact_2kssamp(n1, n2, g, d, alternative)
        if not success:
            mode = 'asymp'
            if original_mode == 'exact':
                warnings.warn(f"ks_2samp: Exact calculation unsuccessful. "
                              f"Switching to mode={mode}.", RuntimeWarning)

    if mode == 'asymp':
        # The product n1*n2 is large.  Use Smirnov's asymptoptic formula.
        # Ensure float to avoid overflow in multiplication
        # sorted because the one-sided formula is not symmetric in n1, n2
        m, n = sorted([float(n1), float(n2)], reverse=True)
        en = m * n / (m + n)
        if alternative == 'two-sided':
            prob = distributions.kstwo.sf(d, np.round(en))
        else:
            z = np.sqrt(en) * d
            # Use Hodges' suggested approximation Eqn 5.3
            # Requires m to be the larger of (n1, n2)
            expt = -2 * z**2 - 2 * z * (m + 2*n)/np.sqrt(m*n*(m+n))/3.0
            prob = np.exp(expt)

    prob = np.clip(prob, 0, 1)
    return KstestResult(d, prob)

def _attempt_exact_2kssamp(n1, n2, g, d, alternative):
    """Attempts to compute the exact 2sample probability.
    n1, n2 are the sample sizes
    g is the gcd(n1, n2)
    d is the computed max difference in ECDFs
    Returns (success, d, probability)
    """
    lcm = (n1 // g) * n2
    h = int(np.round(d * lcm))
    d = h * 1.0 / lcm
    if h == 0:
        return True, d, 1.0
    saw_fp_error, prob = False, np.nan
    try:
        if alternative == 'two-sided':
            if n1 == n2:
                prob = _compute_prob_outside_square(n1, h)
            else:
                prob = _compute_outer_prob_inside_method(n1, n2, g, h)
        else:
            if n1 == n2:
                # prob = binom(2n, n-h) / binom(2n, n)
                # Evaluating in that form incurs roundoff errors
                # from special.binom. Instead calculate directly
                jrange = np.arange(h)
                prob = np.prod((n1 - jrange) / (n1 + jrange + 1.0))
            else:
                num_paths = _count_paths_outside_method(n1, n2, g, h)
                bin = special.binom(n1 + n2, n1)
                if not np.isfinite(bin) or not np.isfinite(num_paths)\
                        or num_paths > bin:
                    saw_fp_error = True
                else:
                    prob = num_paths / bin

    except FloatingPointError:
        saw_fp_error = True

    if saw_fp_error:
        return False, d, np.nan
    if not (0 <= prob <= 1):
        return False, d, prob
    return True, d, prob

def _compute_prob_outside_square(n, h):
    """
    Compute the proportion of paths that pass outside the two diagonal lines.
    Parameters
    ----------
    n : integer
        n > 0
    h : integer
        0 <= h <= n
    Returns
    -------
    p : float
        The proportion of paths that pass outside the lines x-y = +/-h.
    """
    # Compute Pr(D_{n,n} >= h/n)
    # Prob = 2 * ( binom(2n, n-h) - binom(2n, n-2a) + binom(2n, n-3a) - ... )
    # / binom(2n, n)
    # This formulation exhibits subtractive cancellation.
    # Instead divide each term by binom(2n, n), then factor common terms
    # and use a Horner-like algorithm
    # P = 2 * A0 * (1 - A1*(1 - A2*(1 - A3*(1 - A4*(...)))))

    P = 0.0
    k = int(np.floor(n / h))
    while k >= 0:
        p1 = 1.0
        # Each of the Ai terms has numerator and denominator with
        # h simple terms.
        for j in range(h):
            p1 = (n - k * h - j) * p1 / (n + k * h + j + 1)
        P = p1 * (1.0 - P)
        k -= 1
    return 2 * P

