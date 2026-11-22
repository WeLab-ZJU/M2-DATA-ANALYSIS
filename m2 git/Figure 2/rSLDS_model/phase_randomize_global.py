"""
Phase-randomization 
"""

import os
import numpy as np
import scipy.io as sio

# ════════════════════════════════════════════
# 3.  Phase randomization (global FFT, independent per dimension)
# ════════════════════════════════════════════

def phase_randomize_global(xhat, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    T, D = xhat.shape
    mean = xhat.mean(axis=0, keepdims=True)
    x = xhat - mean
    F = np.fft.rfft(x, axis=0)
    n_freq = F.shape[0]
    phases = rng.uniform(0, 2 * np.pi, size=(n_freq, D))
    phases[0] = 0.0
    if T % 2 == 0:
        phases[-1] = 0.0
    F_surr = F * np.exp(1j * phases)
    x_surr = np.fft.irfft(F_surr, n=T, axis=0) + mean
    return x_surr