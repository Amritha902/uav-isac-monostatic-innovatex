import numpy as np
from numpy.fft import fft, fftshift
from scipy.signal import windows

def range_doppler_map(cube, window_fast=True, window_slow=True):
    M, Nc, Ns = cube.shape
    X = cube.copy()
    if window_fast:
        wf = windows.hann(Ns, sym=False)[None, None, :]
        X *= wf
    if window_slow:
        ws = windows.hann(Nc, sym=False)[None, :, None]
        X *= ws
    R = fft(X, n=Ns, axis=2)
    RD = fftshift(fft(R, n=Nc, axis=1), axes=1)
    RD_mag = np.abs(RD).sum(axis=0)
    return RD_mag
