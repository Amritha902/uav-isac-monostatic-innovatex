import numpy as np
from metrics.crlb import crlb_range_velocity_theta

def comm_rate(bw_hz, snr_db):
    # Shannon-ish rate for illustration
    snr_lin = 10**(snr_db/10)
    return bw_hz * np.log2(1 + snr_lin)

def cost_for(cfg, R, v, theta_deg, snr_db, w1, w2):
    sR, sv, st = crlb_range_velocity_theta(cfg, R, v, theta_deg, snr_db)
    crlb_sum = sR + sv + st
    rate = comm_rate(cfg["B"], snr_db)
    return w1*crlb_sum - w2*rate, (sR, sv, st), rate

def optimize(cfgd, guess_R=80.0, guess_v=2.0, guess_theta=10.0, snr_db=15.0, w1=1.0, w2=1.0):
    cfg0 = dict(cfgd)  # copy
    B_vals = np.array([0.5, 1.0, 1.5]) * cfg0["B"]
    T_vals = np.array([0.5, 1.0, 1.5]) * cfg0["T_chirp"]
    best = None
    best_rec = None
    for B in B_vals:
        for T in T_vals:
            cfg0["B"] = float(B)
            cfg0["T_chirp"] = float(T)
            J, crlbs, rate = cost_for(cfg0, guess_R, guess_v, guess_theta, snr_db, w1, w2)
            rec = {"B": float(B), "T_chirp": float(T), "cost": float(J), "crlb": tuple(float(x) for x in crlbs), "rate": float(rate)}
            if (best is None) or (J < best):
                best = J
                best_rec = rec
    return best_rec
