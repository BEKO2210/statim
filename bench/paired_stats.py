"""Small, dependency-free exact tests shared by evaluation and promotion tools."""
import math


def binomial_upper_half(n: int, k: int) -> float:
    """P(X >= k) for X ~ Binomial(n, 1/2), evaluated with integer arithmetic."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if k > n // 2:
        return sum(math.comb(n, i) for i in range(n - k + 1)) / 2 ** n
    return 1.0 - sum(math.comb(n, i) for i in range(k)) / 2 ** n


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p: twice the smaller Binomial(b+c, 1/2) tail."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)

