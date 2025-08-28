import numpy as np

def crlb_range_velocity_theta(cfg, R, v, theta_deg, snr_db):
    """
    Approximate CRLBs using a white-noise model and finite-difference gradients
    of the noiseless signal w.r.t parameters (R,v,theta).
    Returns (sigma_R, sigma_v, sigma_theta_deg).
    """
    C = 299792458.0
    lam = C / cfg["fc"]
    d = cfg["d_over_lambda"] * lam
    Ns = int(cfg["Fs"] * cfg["T_chirp"])
    t = np.arange(Ns) / cfg["Fs"]
    n = np.arange(cfg["N_chirps"])
    slope = cfg["B"] / cfg["T_chirp"]
    theta = np.deg2rad(theta_deg)
    M = int(cfg["M_ant"])

    tau = 2*R/C
    f_r = slope * tau
    f_d = 2 * v * cfg["fc"] / C
    m = np.arange(M)

    s_fast = np.exp(1j*2*np.pi*(f_r*t))
    s_slow = np.exp(1j*2*np.pi*(f_d*(n*cfg["T_chirp"])))
    phase_ant = np.exp(1j*2*np.pi*(m*d*np.sin(theta)/lam))[:,None,None]
    S = phase_ant * (s_slow[:,None]*s_fast[None,:])[None,:,:]  # [M,Nc,Ns]

    # vectorize
    x = S.ravel()
    # finite diffs
    epsR = 0.1
    epsv = 0.05
    epst = np.deg2rad(0.1)

    def S_of(Rp, vp, thetap):
        tau_p = 2*Rp/C
        fr_p = slope * tau_p
        fd_p = 2 * vp * cfg["fc"] / C
        s_fast_p = np.exp(1j*2*np.pi*(fr_p*t))
        s_slow_p = np.exp(1j*2*np.pi*(fd_p*(n*cfg["T_chirp"])))
        phase = np.exp(1j*2*np.pi*(m*d*np.sin(thetap)/lam))[:,None,None]
        return (phase * (s_slow_p[:,None]*s_fast_p[None,:])[None,:,:]).ravel()

    dR = (S_of(R+epsR, v, theta) - S_of(R-epsR, v, theta)) / (2*epsR)
    dv = (S_of(R, v+epsv, theta) - S_of(R, v-epsv, theta)) / (2*epsv)
    dt = (S_of(R, v, theta+epst) - S_of(R, v, theta-epst)) / (2*epst)

    # FIM under AWGN: (2/σ^2) Re{ J^H J }, J = [dR dv dt]
    snr_lin = 10**(snr_db/10)
    # approximate per-sample signal power:
    sig_pow = np.mean(np.abs(x)**2)
    noise_pow = sig_pow / max(snr_lin, 1e-9)
    sigma2 = noise_pow

    JHJ = np.array([
        [np.vdot(dR, dR).real, np.vdot(dR, dv).real, np.vdot(dR, dt).real],
        [np.vdot(dv, dR).real, np.vdot(dv, dv).real, np.vdot(dv, dt).real],
        [np.vdot(dt, dR).real, np.vdot(dt, dv).real, np.vdot(dt, dt).real],
    ], dtype=float)
    FIM = (2.0/sigma2) * JHJ
    try:
        CRB = np.linalg.inv(FIM)
    except np.linalg.LinAlgError:
        return np.inf, np.inf, np.inf
    var_R, var_v, var_t = CRB[0,0], CRB[1,1], CRB[2,2]
    # convert theta variance from rad^2 to deg^2
    var_t_deg = var_t * (180/np.pi)**2
    return np.sqrt(max(var_R,0)), np.sqrt(max(var_v,0)), np.sqrt(max(var_t_deg,0))
