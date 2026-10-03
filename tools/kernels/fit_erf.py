#!/usr/bin/env python3
"""Reproduce the erf coefficients used by kernels.h and kernels.cpp.

The public scalar approximation is two polynomial regions. The AVX2 GeGLU
specialization uses a global (6, 5) rational for erf(x)/x; it is less suitable
as a standalone erff but is faster and preserves the kernel's 4-ulp contract.
All fits use deterministic binary64 samples and emit binary32 coefficients.
"""

import argparse
import math

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from numpy.polynomial import chebyshev as cheb


def values(function, points):
    return np.fromiter((function(float(x)) for x in points), dtype=np.float64,
                       count=len(points))


def polynomial_regions(samples, split=1.0, saturation=3.92):
    z = np.linspace(0.0, split * split, samples)
    x = np.sqrt(z)
    target = np.empty_like(z)
    target[0] = 2.0 / math.sqrt(math.pi)
    target[1:] = values(math.erf, x[1:]) / x[1:]
    small_t = 2.0 * z / (split * split) - 1.0
    small = Chebyshev.fit(small_t, target, 7,
                          domain=[-1.0, 1.0]).convert(kind=Polynomial).coef

    x = np.linspace(split, saturation, samples)
    tail_t = (2.0 * x - split - saturation) / (saturation - split)
    tail = Chebyshev.fit(tail_t, values(math.erf, x), 14,
                         domain=[-1.0, 1.0]).convert(kind=Polynomial).coef
    return small, tail


def global_rational(samples, saturation=3.92):
    z = np.linspace(0.0, saturation * saturation, samples)
    x = np.sqrt(z)
    target = np.empty_like(z)
    target[0] = 2.0 / math.sqrt(math.pi)
    target[1:] = values(math.erf, x[1:]) / x[1:]
    t = 2.0 * z / (saturation * saturation) - 1.0
    fit_t, fit_y = t[::20], target[::20]
    basis = cheb.chebvander(fit_t, 6)
    matrix = np.concatenate((basis[:, :7], -fit_y[:, None] * basis[:, 1:6]), axis=1)
    solution = np.linalg.lstsq(matrix, fit_y, rcond=None)[0]
    transform = Polynomial([-1.0, 2.0])
    numerator = Chebyshev(solution[:7]).convert(kind=Polynomial)(transform).coef
    denominator = Chebyshev(np.r_[1.0, solution[7:]]).convert(kind=Polynomial)(transform).coef
    return numerator, denominator


def emit(name, coefficients):
    rounded = np.asarray(coefficients, dtype=np.float32)
    print(f"{name} ({len(rounded) - 1}):")
    print("    " + ", ".join(f"{value:.9e}f" for value in rounded))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1_000_001)
    args = parser.parse_args()
    small, tail = polynomial_regions(args.samples)
    numerator, denominator = global_rational(args.samples)
    emit("scalar small numerator in t=2*x*x-1", small)
    emit("scalar tail in t=(2*x-4.92)/2.92", tail)
    emit("AVX2 numerator in u=x*x/3.92^2", numerator)
    emit("AVX2 denominator in u=x*x/3.92^2", denominator)


if __name__ == "__main__":
    main()
