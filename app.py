import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from estimators.dsp_baseline import range_doppler_map
from estimators.cfar import ca_cfar_2d, nms_peaks
from estimators.peak_utils import bins_to_physical, estimate_snr_confidence
from estimators.music_doa import estimate_doa_music
from simulate_isac import RadarConfig, simulate_frame, C
from metrics.crlb import crlb_range_velocity_theta
from optim.adaptive_isac import optimize
from security.provenance import sha256_file, merkle_root

# ====== ML baseline additions (NEW) ======
import os, joblib
RF = joblib.load("ml/baseline.joblib") if os.path.exists("ml/baseline.joblib") else None

# Set these to the feature size you trained with (e.g., 24x192 or 32x256)
D_BINS = 32
R_BINS = 256

def downsample_rd(RD, d_bins=D_BINS, r_bins=R_BINS):
    """Average-pool RD to (d_bins x r_bins) and normalize."""
    H, W = RD.shape
    dh = max(H // d_bins, 1)
    dw = max(W // r_bins, 1)
    RD_small = RD[:d_bins*dh, :r_bins*dw].reshape(d_bins, dh, r_bins, dw).mean(axis=(1,3))
    RD_small = RD_small / (RD_small.max() + 1e-9)
    return RD_small.astype(np.float32)
# ========================================

st.set_page_config(page_title="UAV-ISAC v3: Adaptive + CRLB + Security", layout="wide")
st.title("UAV-ISAC Monostatic (v3)")

def gaussian_denoise(arr, sigma=1.0):
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(arr, sigma=sigma)
    except Exception:
        return arr

def draw_rd_with_peaks(RD, peaks):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.imshow(20*np.log10(RD + 1e-6), aspect='auto', origin='lower')
    ys = [p[0] for p in peaks]
    xs = [p[1] for p in peaks]
    if len(xs)>0:
        ax.scatter(xs, ys, s=30, marker='x')
    ax.set_xlabel("Range bin")
    ax.set_ylabel("Doppler bin")
    return fig

with st.sidebar:
    st.header("Simulation")
    fc = st.number_input("Carrier (Hz)", value=77e9, format="%.0f")
    B = st.number_input("Chirp Bandwidth (Hz)", value=200e6, format="%.0f")
    T_chirp = st.number_input("Chirp Duration (s)", value=5e-4, format="%.6f")
    Fs = st.number_input("ADC Fs (Hz)", value=2e6, format="%.0f")
    N_chirps = st.number_input("Chirps per frame", value=64, step=1)
    M_ant = st.number_input("ULA elements", value=8, step=1)
    d_over_lambda = st.number_input("d / lambda", value=0.5, format="%.2f")
    snr_db = st.number_input("SNR (dB)", value=15.0, format="%.1f")
    max_targets = st.number_input("Max targets", value=2, step=1)
    multipath = st.checkbox("Enable multipath", value=False)

    st.header("Detection (CFAR)")
    use_cfar = st.checkbox("Use 2D CA-CFAR", value=True)
    train_r = st.number_input("Train rows", value=8, step=1)
    train_c = st.number_input("Train cols", value=8, step=1)
    guard_r = st.number_input("Guard rows", value=2, step=1)
    guard_c = st.number_input("Guard cols", value=2, step=1)
    pfa = st.number_input("Pfa", value=1e-3, format="%.1e")
    max_peaks = st.number_input("Max peaks", value=3, step=1)
    nms_rad = st.number_input("NMS radius", value=3, step=1)

    st.header("Enhancements")
    use_denoiser = st.checkbox("Use denoiser (Gaussian)", value=False)
    show_crlb = st.checkbox("Show CRLB vs MAE plot", value=False)
    adaptive_mode = st.checkbox("Adaptive waveform optimization", value=False)
    w1 = st.number_input("w1 (CRLB weight)", value=1.0, format="%.2f")
    w2 = st.number_input("w2 (Rate weight)", value=1.0, format="%.2f")

    # ====== ML baseline toggle (NEW) ======
    st.header("ML Baseline")
    use_ml = st.checkbox("Use ML baseline (RandomForest)", value=(RF is not None))
    if use_ml and RF is None:
        st.info("Place a trained model at ml/baseline.joblib")
    # =====================================

    if st.button("Simulate Frame / Optimize"):
        cfg = RadarConfig(fc=fc, B=B, T_chirp=T_chirp, Fs=Fs, N_chirps=int(N_chirps),
                          M_ant=int(M_ant), d_over_lambda=float(d_over_lambda),
                          snr_db=float(snr_db), max_targets=int(max_targets), multipath=bool(multipath))
        cfgd = cfg.__dict__.copy()
        if adaptive_mode:
            rec = optimize(cfgd, snr_db=float(snr_db), w1=float(w1), w2=float(w2))
            st.success(f"Adaptive choice: B={rec['B']:.2e} Hz, T_chirp={rec['T_chirp']:.2e} s, cost={rec['cost']:.3g}")
            cfg = RadarConfig(**{**cfgd, "B": rec["B"], "T_chirp": rec["T_chirp"]})
        cube, labels, cfgd = simulate_frame(cfg)
        st.session_state["cube"] = cube
        st.session_state["labels"] = labels
        st.session_state["cfg"] = cfgd
        st.session_state["cfar_params"] = dict(train_r=int(train_r), train_c=int(train_c),
                                               guard_r=int(guard_r), guard_c=int(guard_c),
                                               pfa=float(pfa), max_peaks=int(max_peaks),
                                               nms_rad=int(nms_rad), use_cfar=bool(use_cfar),
                                               use_denoiser=bool(use_denoiser), show_crlb=bool(show_crlb))
        # store ML toggle (NEW)
        st.session_state["use_ml"] = bool(use_ml)

col1, col2 = st.columns(2)

def process_and_display(cube, labels, cfgd, cfar_params):
    RD = range_doppler_map(cube)
    if cfar_params.get("use_denoiser", False):
        RD = gaussian_denoise(RD, 1.0)

    if cfar_params.get("use_cfar", True):
        det, thr, nmean = ca_cfar_2d(RD, guard=(cfar_params["guard_r"], cfar_params["guard_c"]),
                                     train=(cfar_params["train_r"], cfar_params["train_c"]), pfa=cfar_params["pfa"])
        peaks = nms_peaks(RD, det, max_peaks=cfar_params["max_peaks"], radius=cfar_params["nms_rad"])
    else:
        idx = np.unravel_index(np.argmax(RD), RD.shape)
        peaks = [(idx[0], idx[1], float(RD[idx]))]
        nmean = np.ones_like(RD) * (RD.mean()**2)

    with col1:
        st.subheader("Range–Doppler (with peaks)")
        fig1 = draw_rd_with_peaks(RD, peaks)
        st.pyplot(fig1)

    Nc, Ns = RD.shape
    lam = C / cfgd["fc"]
    results = []
    for r,c,val in peaks:
        R_est, v_est = bins_to_physical(c, r, Ns, Nc, cfgd["Fs"], cfgd["B"], cfgd["T_chirp"], cfgd["fc"])
        snr_db_est, sharp, conf = estimate_snr_confidence(RD, r, c, nmean, guard=cfar_params["guard_r"])
        M = cube.shape[0]
        snapshot = cube[:, r, c]
        doa_list, spectrum = estimate_doa_music(snapshot, num_sources=1, M=M, d=(cfgd['d_over_lambda']*lam), wavelength=lam)
        results.append({
            "doppler_bin": int(r), "range_bin": int(c),
            "R_est_m": float(R_est), "v_est_mps": float(v_est),
            "DoA_est_deg": float(doa_list[0]),
            "snr_db": float(snr_db_est), "sharpness": float(sharp), "confidence": float(conf),
            "peak_val": float(val)
        })

    # ====== ML baseline prediction (NEW) ======
    ml_row = None
    use_ml_flag = st.session_state.get("use_ml", False)
    if use_ml_flag and RF is not None:
        try:
            RD_small = downsample_rd(RD, D_BINS, R_BINS).flatten()[None, :]
            Rm = RF["R"].predict(RD_small)[0]
            Vm = RF["V"].predict(RD_small)[0]
            s  = RF["T_sin"].predict(RD_small)[0]
            c  = RF["T_cos"].predict(RD_small)[0]
            Th = float(np.degrees(np.arctan2(s, c)))
            ml_row = {"doppler_bin": None, "range_bin": None,
                      "R_est_m": float(Rm), "v_est_mps": float(Vm), "DoA_est_deg": Th,
                      "snr_db": None, "sharpness": None, "confidence": None, "peak_val": None, "source": "ML"}
        except Exception as e:
            st.warning(f"ML baseline error: {e}")
    # =========================================

    with col2:
        st.subheader("MUSIC Spectrum (first detected peak)")
        if len(results)>0:
            ang_grid = np.linspace(-90, 90, 721)
            M = cube.shape[0]
            r0,c0,_ = peaks[0]
            snapshot0 = cube[:, r0, c0]
            est_angles, spec = estimate_doa_music(snapshot0, num_sources=1, M=M, d=(cfgd['d_over_lambda']*lam), wavelength=lam, angle_grid=ang_grid)
            fig2, ax2 = plt.subplots(figsize=(6,4))
            ax2.plot(ang_grid, 10*np.log10(spec + 1e-9))
            ax2.set_xlabel("Angle (deg)")
            ax2.set_ylabel("Pseudo-spectrum (dB)")
            st.pyplot(fig2)
        else:
            st.info("No peaks detected above CFAR threshold. Increase SNR or relax Pfa/CFAR params.")

    st.markdown("### Detections and Ground Truth Matching")
    gR, gV, gT = labels["ranges_m"], labels["vels_mps"], labels["thetas_deg"]
    rows = []
    for res in results:
        dists = [np.hypot(res["R_est_m"]-gr, res["v_est_mps"]-gv) for gr,gv in zip(gR,gV)]
        gi = int(np.argmin(dists)) if len(dists)>0 else -1
        rows.append({**res,
                     "GT_R_m": gR[gi] if gi>=0 else None,
                     "GT_v_mps": gV[gi] if gi>=0 else None,
                     "GT_DoA_deg": gT[gi] if gi>=0 else None,
                     "matched_gt": gi,
                     "source": "DSP"})
    # Append ML global estimate row matched to nearest GT (for quick compare)
    if ml_row is not None and len(gR) > 0:
        gi_ml = int(np.argmin([np.hypot(ml_row["R_est_m"]-gr, ml_row["v_est_mps"]-gv) for gr,gv in zip(gR,gV)]))
        ml_row.update({"GT_R_m": gR[gi_ml], "GT_v_mps": gV[gi_ml], "GT_DoA_deg": gT[gi_ml], "matched_gt": gi_ml})
        rows.append(ml_row)

    st.write(rows)

    st.session_state["last_results"] = rows
    st.session_state["last_RD"] = RD
    st.session_state["last_cfg"] = cfgd

def evaluate_mae(n_frames=30):
    if "cfg" not in st.session_state or "cfar_params" not in st.session_state:
        st.warning("Simulate one frame first to capture the current config/CFAR settings.")
        return
    cfgd = st.session_state["cfg"]
    cp = st.session_state["cfar_params"]
    cfg = RadarConfig(**cfgd)
    errs_R, errs_V, errs_T = [], [], []
    misses = 0
    for _ in range(n_frames):
        cube, labels, _ = simulate_frame(cfg)
        RD = range_doppler_map(cube)
        if cp.get("use_denoiser", False):
            RD = gaussian_denoise(RD, 1.0)
        if cp.get("use_cfar", True):
            det, thr, nmean = ca_cfar_2d(RD, guard=(cp["guard_r"], cp["guard_c"]),
                                         train=(cp["train_r"], cp["train_c"]), pfa=cp["pfa"])
            peaks = nms_peaks(RD, det, max_peaks=cp["max_peaks"], radius=cp["nms_rad"])
        else:
            idx = np.unravel_index(np.argmax(RD), RD.shape)
            peaks = [(idx[0], idx[1], float(RD[idx]))]
        if len(peaks)==0:
            misses += 1
            continue
        Nc, Ns = RD.shape
        lam = C / cfg.fc
        pi = labels["primary_index"]
        gR, gV, gT = labels["ranges_m"][pi], labels["vels_mps"][pi], labels["thetas_deg"][pi]
        best = None; bestd = 1e9; best_theta = None
        for r,c,val in peaks:
            R_est, v_est = bins_to_physical(c, r, Ns, Nc, cfg.Fs, cfg.B, cfg.T_chirp, cfg.fc)
            d = np.hypot(R_est-gR, v_est-gV)
            if d < bestd:
                snapshot = cube[:, r, c]
                doa, _ = estimate_doa_music(snapshot, num_sources=1, M=cfg.M_ant, d=(cfg.d_over_lambda*lam), wavelength=lam)
                best = (R_est, v_est); bestd = d; best_theta = doa[0]
        if best is None:
            misses += 1
            continue
        errs_R.append(abs(best[0]-gR))
        errs_V.append(abs(best[1]-gV))
        aerr = abs(((best_theta - gT + 180) % 360) - 180)
        errs_T.append(aerr)
    if len(errs_R)==0:
        return {"frames": n_frames, "misses": misses, "note": "No detections across evaluation."}
    return {"frames": n_frames, "misses": misses,
            "MAE_R_m": float(np.mean(errs_R)),
            "MAE_V_mps": float(np.mean(errs_V)),
            "MAE_DoA_deg": float(np.mean(errs_T))}

if "cube" in st.session_state and "cfar_params" in st.session_state:
    process_and_display(st.session_state["cube"], st.session_state["labels"], st.session_state["cfg"], st.session_state["cfar_params"])
else:
    st.info("Use the sidebar to Simulate Frame / Optimize.")

st.markdown("---")
st.subheader("Evaluate Metrics and CRLB")
nf = st.number_input("Number of frames", value=30, step=10)
if st.button("Evaluate N frames"):
    metrics = evaluate_mae(int(nf))
    st.write(metrics)
    st.session_state["last_metrics"] = metrics

if st.checkbox("Plot CRLB vs SNR (primary target estimate)"):
    if "cube" in st.session_state:
        labels = st.session_state["labels"]
        cfgd = st.session_state["cfg"]
        pi = labels["primary_index"]
        gR, gV, gT = labels["ranges_m"][pi], labels["vels_mps"][pi], labels["thetas_deg"][pi]
        SNRs = np.linspace(0, 25, 11)
        sigR, sigV, sigT = [], [], []
        for sn in SNRs:
            sR, sV, sT = crlb_range_velocity_theta(cfgd, gR, gV, gT, sn)
            sigR.append(sR); sigV.append(sV); sigT.append(sT)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(SNRs, sigR, label="CRLB Range (m)")
        ax.plot(SNRs, sigV, label="CRLB Vel (m/s)")
        ax.plot(SNRs, sigT, label="CRLB DoA (deg)")
        ax.set_xlabel("SNR (dB)"); ax.set_ylabel("CRLB units")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Simulate one frame first.")

st.markdown("---")
st.subheader("Provenance quick tools")
if st.button("Hash current dataset file (if exists)"):
    test = "datasets/synth_v3.npz"
    try:
        h = sha256_file(test)
        st.write({"file": test, "sha256": h})
    except Exception as e:
        st.error(str(e))
