import numpy as np
from numpy.random import default_rng
from dataclasses import dataclass, asdict
import argparse, json

C = 299792458.0

@dataclass
class RadarConfig:
    fc: float = 77e9
    B: float = 200e6
    T_chirp: float = 5e-4
    Fs: float = 2e6
    N_chirps: int = 64
    M_ant: int = 8
    d_over_lambda: float = 0.5
    snr_db: float = 15.0
    max_targets: int = 2
    multipath: bool = False
    mp_paths_per_target: int = 2

def simulate_frame(cfg: RadarConfig, rng=None):
    rng = default_rng() if rng is None else rng
    Cc = C
    lam = Cc / cfg.fc
    d = cfg.d_over_lambda * lam
    Ns = int(cfg.Fs * cfg.T_chirp)
    t = np.arange(Ns) / cfg.Fs
    n = np.arange(cfg.N_chirps)
    slope = cfg.B / cfg.T_chirp

    K = rng.integers(1, cfg.max_targets+1)
    ranges = rng.uniform(20.0, 200.0, size=K)
    vels = rng.uniform(-15.0, 15.0, size=K)
    thetas = rng.uniform(-50.0, 50.0, size=K)
    alphas = rng.uniform(0.5, 1.0, size=K) * np.exp(1j * rng.uniform(0, 2*np.pi, size=K))

    cube = np.zeros((cfg.M_ant, cfg.N_chirps, Ns), dtype=complex)
    m = np.arange(cfg.M_ant)
    for k in range(K):
        tau = 2 * ranges[k] / Cc
        f_r = slope * tau
        f_d = 2 * vels[k] * cfg.fc / Cc
        theta = np.deg2rad(thetas[k])
        phase_ant = np.exp(1j * 2*np.pi * (m * d * np.sin(theta) / lam))[:, None, None]
        s_fast = np.exp(1j * 2*np.pi * (f_r * t))
        s_slow = np.exp(1j * 2*np.pi * (f_d * (n * cfg.T_chirp)))
        s = alphas[k] * (s_slow[:, None] * s_fast[None, :])
        cube += phase_ant * s[None, :, :]

        if cfg.multipath:
            P = cfg.mp_paths_per_target
            for _ in range(P):
                amp = alphas[k] * (0.25 + 0.25*rng.random()) * np.exp(1j*rng.uniform(0,2*np.pi))
                tau_mp = 2 * (ranges[k] + rng.uniform(-3.0, 6.0)) / Cc
                f_r_mp = slope * tau_mp
                theta_mp = np.deg2rad(thetas[k] + rng.uniform(-8.0, 8.0))
                phase_ant_mp = np.exp(1j * 2*np.pi * (m * d * np.sin(theta_mp) / lam))[:, None, None]
                s_fast_mp = np.exp(1j * 2*np.pi * (f_r_mp * t))
                s_mp = amp * (s_slow[:, None] * s_fast_mp[None, :])
                cube += phase_ant_mp * s_mp[None, :, :]

    sig_pow = np.mean(np.abs(cube)**2)
    snr_lin = 10**(cfg.snr_db/10)
    noise_pow = sig_pow / max(snr_lin, 1e-9)
    noise = (np.sqrt(noise_pow/2) * (np.random.randn(*cube.shape) + 1j*np.random.randn(*cube.shape)))
    cube_noisy = cube + noise

    k0 = np.argmax(np.abs(alphas))
    labels = {
        "ranges_m": ranges.tolist(),
        "vels_mps": vels.tolist(),
        "thetas_deg": thetas.tolist(),
        "primary_index": int(k0)
    }
    return cube_noisy, labels, asdict(cfg)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--out", type=str, default="datasets/synth_v3.npz")
    ap.add_argument("--snr-db", type=float, default=15.0)
    ap.add_argument("--max-targets", type=int, default=2)
    ap.add_argument("--multipath", action="store_true")
    args = ap.parse_args()

    cfg = RadarConfig(snr_db=args.snr_db, max_targets=args.max_targets, multipath=args.multipath)
    cubes, labels, cfgs = [], [], []
    for _ in range(args.num_samples):
        cube, lab, cfgdict = simulate_frame(cfg)
        cubes.append(cube.astype(np.complex64))
        labels.append(lab)
        cfgs.append(cfgdict)

    np.savez_compressed(args.out, cubes=np.array(cubes, dtype=object), labels=np.array(labels, dtype=object), cfgs=np.array(cfgs, dtype=object))
    print("Saved", args.out)
