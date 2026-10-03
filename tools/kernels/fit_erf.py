#!/usr/bin/env python3
"""Reproduce the erf coefficients of statim::erf_approx (src/kernels.h).

Two polynomial regions, split at |x| = 1 and saturating at 3.92, at most 2 ulp
from glibc's erff over every finite float (tests/test_kernels.cpp). GeGLU
vectorizes the same function; a faster (6, 5) rational was tried and rejected:
up to 7 ulp, values above 1, and GeGLU sign flips near the saturation knee.
All fits use deterministic binary64 samples and emit binary32 coefficients.
"""

import argparse
import math

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial


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


def emit(name, coefficients):
    rounded = np.asarray(coefficients, dtype=np.float32)
    print(f"{name} ({len(rounded) - 1}):")
    print("    " + ", ".join(f"{value:.9e}f" for value in rounded))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1_000_001)
    args = parser.parse_args()
    small, tail = polynomial_regions(args.samples)
    emit("scalar small numerator in t=2*x*x-1", small)
    emit("scalar tail in t=(2*x-4.92)/2.92", tail)


if __name__ == "__main__":
    main()
