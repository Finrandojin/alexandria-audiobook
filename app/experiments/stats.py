"""The two tests every harness in this investigation needs, written once.

Both were previously re-derived in throwaway analysis scripts, which is how the
crossover result got a decision rule that degenerated at zero variance. Rule 15
applies to statistics as much as to dispatch: two independently-maintained
copies of "is this significant?" will drift, and the drift is invisible because
both produce a plausible number.

No scipy dependency: the exact binomial tail and the Clopper-Pearson interval
are both computable from the regularised incomplete beta function, and the beta
quantile here is a bisection on that. It is slower than scipy and exact to
1e-10, which is far past what any of these sample sizes justify.
"""
import math


def _log_choose(n, k):
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def exact_mcnemar(b, c):
    """Two-sided exact McNemar on the discordant pairs only.

    `b` = rows the first arm got right and the second got wrong, `c` = the
    reverse. Concordant rows carry no information about which arm is better and
    are correctly ignored - which is also why a big n does not rescue a test
    with four discordant pairs.

    Returns (p, b, c). p is 1.0 when there are no discordant pairs, which is the
    honest answer: nothing changed anywhere, so nothing is demonstrated.
    """
    n = b + c
    if n == 0:
        return 1.0, b, c
    # Under H0 each discordant pair is a fair coin; sum the probability of every
    # outcome at least as extreme as the observed split.
    obs = min(b, c)
    tail = sum(math.exp(_log_choose(n, k) - n * math.log(2))
               for k in range(0, obs + 1))
    return min(1.0, 2.0 * tail), b, c


def _betainc(a, b, x, terms=400):
    """Regularised incomplete beta I_x(a, b) by its continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _betainc(b, a, 1 - x, terms)
    lead = math.exp(a * math.log(x) + b * math.log(1 - x)
                    + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))
    # Lentz's algorithm.
    tiny, f, c, d = 1e-30, 1.0, 1.0, 0.0
    for i in range(terms):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = tiny if abs(d) < tiny else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = tiny if abs(c) < tiny else c
        delta = c * d
        f *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return lead * (f - 1.0) / a


def _beta_quantile(p, a, b):
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial confidence interval, as a (lo, hi) percentage pair.

    Exact rather than normal-approximation because several strata in these
    analyses have single-digit n, where the normal interval runs past 0 or 100
    and quietly implies precision that is not there.
    """
    if n == 0:
        return 0.0, 100.0
    lo = 0.0 if k == 0 else _beta_quantile(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta_quantile(1 - alpha / 2, k + 1, n - k)
    return lo * 100, hi * 100


def paired(rows_a, rows_b):
    """McNemar over two {id: correct} maps, restricted to shared ids."""
    shared = set(rows_a) & set(rows_b)
    b = sum(1 for i in shared if rows_a[i] and not rows_b[i])
    c = sum(1 for i in shared if rows_b[i] and not rows_a[i])
    return exact_mcnemar(b, c) + (len(shared),)
