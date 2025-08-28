import numpy as np

def bins_to_physical(range_bin, doppler_bin, Ns, Nc, Fs, B, T_chirp, fc, C=299792458.0):
    freqs_fast = np.fft.fftfreq(Ns, d=1.0/Fs)
    f_b = freqs_fast[range_bin]
    slope = B / T_chirp
    R = (f_b * C) / (2 * slope)
    fd_grid = np.fft.fftfreq(Nc, d=T_chirp)
    fd_grid = np.fft.fftshift(fd_grid)
    f_D = fd_grid[doppler_bin]
    v = (f_D * C) / (2 * fc)
    return float(R), float(v)

def estimate_snr_confidence(RD_mag, r, c, noise_mean, guard=2):
    peak_pow = RD_mag[r,c]**2
    noise = max(noise_mean[r,c], 1e-12)
    snr_lin = peak_pow / noise
    snr_db = 10*np.log10(snr_lin + 1e-12)
    H,W = RD_mag.shape
    r0, r1 = max(0,r-guard-2), min(H-1, r+guard+2)
    c0, c1 = max(0,c-guard-2), min(W-1, c+guard+2)
    patch = RD_mag[r0:r1+1, c0:c1+1]
    ring = patch.copy()
    rr, cc = np.indices(ring.shape)
    rr_abs = rr + r0; cc_abs = cc + c0
    mask_guard = (np.abs(rr_abs - r) <= guard) & (np.abs(cc_abs - c) <= guard)
    ring[mask_guard] = np.nan
    ring_mean = np.nanmean(ring)
    sharp = float(RD_mag[r,c] / (ring_mean + 1e-6))
    conf = float(1.0 / (1.0 + np.exp(-(snr_db/3.0))) * min(sharp/3.0, 1.0))
    return float(snr_db), sharp, conf
