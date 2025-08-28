import os, sys, argparse, time, json
from pathlib import Path
import numpy as np
import joblib

# allow local imports
THIS = Path(__file__).resolve()
ROOT = THIS.parent
sys.path.insert(0, str(ROOT))

from simulate_isac import RadarConfig, simulate_frame
from estimators.dsp_baseline import range_doppler_map
from sklearn.ensemble import RandomForestRegressor

def mkdirp(p): Path(p).mkdir(parents=True, exist_ok=True)

def downsample_rd(RD, d_bins=24, r_bins=192):
    H, W = RD.shape  # H=Nc (doppler), W=Ns (range)
    dh = max(H // d_bins, 1)
    dw = max(W // r_bins, 1)
    RD_small = RD[:d_bins*dh, :r_bins*dw].reshape(d_bins, dh, r_bins, dw).mean(axis=(1,3))
    RD_small = RD_small / (RD_small.max() + 1e-9)
    return RD_small.astype(np.float32)

def features_from_cube(cube, d_bins=24, r_bins=192):
    RD = range_doppler_map(cube)
    RD_small = downsample_rd(RD, d_bins, r_bins)
    return RD_small.flatten()

def stream_generate_features(split, N, cfg_kwargs, outdir, d_bins=24, r_bins=192, sample_cubes=0):
    """
    Streams N samples: simulate -> feature -> write memmap; returns file paths.
    Optionally saves 'sample_cubes' raw cubes+labels to NPZ for provenance.
    """
    outdir = Path(outdir)
    featdir = outdir / "features"
    mkdirp(featdir)
    mkdirp(outdir / "samples")

    # First simulate one to know feature dim
    cfg = RadarConfig(**cfg_kwargs)
    cube0, lab0, _ = simulate_frame(cfg)
    f0 = features_from_cube(cube0, d_bins, r_bins)
    D = f0.size
    # Pre-create memmaps
    X_path = featdir / f"X_{split}.dat"
    yR_path = featdir / f"yR_{split}.npy"
    yV_path = featdir / f"yV_{split}.npy"
    yT_path = featdir / f"yT_{split}.npy"
    X = np.memmap(X_path, dtype=np.float32, mode="w+", shape=(N, D))
    yR = np.zeros(N, dtype=np.float32)
    yV = np.zeros(N, dtype=np.float32)
    yT = np.zeros(N, dtype=np.float32)

    # Write first sample
    X[0,:] = f0
    i0 = lab0["primary_index"]
    yR[0] = lab0["ranges_m"][i0]; yV[0] = lab0["vels_mps"][i0]; yT[0] = lab0["thetas_deg"][i0]

    # Optionally save a few raw cubes for provenance
    if sample_cubes > 0:
        np.savez_compressed(outdir / "samples" / f"{split}_sample_000.npz", cubes=np.array([cube0], dtype=object), labels=np.array([lab0], dtype=object))

    # Stream the rest
    for i in range(1, N):
        cube, lab, _ = simulate_frame(cfg)
        X[i,:] = features_from_cube(cube, d_bins, r_bins)
        pi = lab["primary_index"]
        yR[i] = lab["ranges_m"][pi]; yV[i] = lab["vels_mps"][pi]; yT[i] = lab["thetas_deg"][pi]
        if sample_cubes > 0 and i < sample_cubes:
            np.savez_compressed(outdir / "samples" / f"{split}_sample_{i:03d}.npz", cubes=np.array([cube], dtype=object), labels=np.array([lab], dtype=object))

    # Flush to disk
    X.flush(); np.save(yR_path, yR); np.save(yV_path, yV); np.save(yT_path, yT)
    return str(X_path), str(yR_path), str(yV_path), str(yT_path), D

def train_rf_from_features(X_path, yR_path, yV_path, yT_path, out_model):
    X_mm = np.memmap(X_path, dtype=np.float32, mode="r")
    yR = np.load(yR_path); yV = np.load(yV_path); yT = np.load(yT_path)
    N = yR.shape[0]; D = X_mm.size // N
    X = np.array(X_mm).reshape(N, D)

    R = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
    V = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
    Ts = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
    Tc = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)

    R.fit(X, yR); V.fit(X, yV)
    Ts.fit(X, np.sin(np.deg2rad(yT)))
    Tc.fit(X, np.cos(np.deg2rad(yT)))

    Path(out_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"R":R,"V":V,"T_sin":Ts,"T_cos":Tc}, out_model)
    return True

def eval_on_features(X_path, yR_path, yV_path, yT_path, model_path):
    X_mm = np.memmap(X_path, dtype=np.float32, mode="r")
    yR = np.load(yR_path); yV = np.load(yV_path); yT = np.load(yT_path)
    N = yR.shape[0]; D = X_mm.size // N
    X = np.array(X_mm).reshape(N, D)

    mdl = joblib.load(model_path)
    Rm, Vm, Ts, Tc = mdl["R"], mdl["V"], mdl["T_sin"], mdl["T_cos"]
    r = Rm.predict(X); v = Vm.predict(X)
    s = Ts.predict(X); c = Tc.predict(X)
    th = np.degrees(np.arctan2(s, c))

    errR = np.abs(r - yR).mean()
    errV = np.abs(v - yV).mean()
    ang = np.abs(th - yT) % 360.0
    ang[ang>180] = 360 - ang[ang>180]
    errT = ang.mean()
    return {"MAE_R(m)":float(errR), "MAE_V(m/s)":float(errV), "MAE_Theta(deg)":float(errT), "N":int(N)}

def main():
    ap = argparse.ArgumentParser(description="Efficient dataset builder and trainer for 8GB RAM")
    ap.add_argument("--outdir", default="datasets")
    ap.add_argument("--ntrain", type=int, default=300)
    ap.add_argument("--nval", type=int, default=80)
    ap.add_argument("--ntest", type=int, default=80)
    ap.add_argument("--d_bins", type=int, default=24, help="downsampled Doppler bins")
    ap.add_argument("--r_bins", type=int, default=192, help="downsampled Range bins")
    ap.add_argument("--samples", type=int, default=0, help="save this many raw cubes per split for provenance")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    mkdirp(outdir)

    cfg_train = dict(fc=77e9, B=200e6, T_chirp=5e-4, Fs=2e6, N_chirps=64, M_ant=8, d_over_lambda=0.5, snr_db=5,  max_targets=3, multipath=True)
    cfg_val   = dict(fc=77e9, B=200e6, T_chirp=5e-4, Fs=2e6, N_chirps=64, M_ant=8, d_over_lambda=0.5, snr_db=10, max_targets=3, multipath=True)
    cfg_test  = dict(fc=77e9, B=200e6, T_chirp=5e-4, Fs=2e6, N_chirps=64, M_ant=8, d_over_lambda=0.5, snr_db=15, max_targets=3, multipath=True)

    print("=== Streaming TRAIN features ===")
    Xtr, yRtr, yVtr, yTtr, D = stream_generate_features("train", args.ntrain, cfg_train, outdir, args.d_bins, args.r_bins, args.samples)
    print("=== Streaming VAL features ===")
    Xva, yRva, yVva, yTva, _ = stream_generate_features("val", args.nval, cfg_val, outdir, args.d_bins, args.r_bins, args.samples)
    print("=== Streaming TEST features ===")
    Xte, yRte, yVte, yTte, _ = stream_generate_features("test", args.ntest, cfg_test, outdir, args.d_bins, args.r_bins, args.samples)

    print("=== Training RandomForest ===")
    model_path = "ml/baseline.joblib"
    train_rf_from_features(Xtr, yRtr, yVtr, yTtr, model_path)

    print("=== Evaluating on TEST ===")
    metrics = eval_on_features(Xte, yRte, yVte, yTte, model_path)
    print(metrics)

    # Simple manifest (no heavy crypto; friendly for 8GB)
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "features": {"D": D, "d_bins": args.d_bins, "r_bins": args.r_bins},
        "splits": {"train": args.ntrain, "val": args.nval, "test": args.ntest},
        "paths": {"X_train": Xtr, "X_val": Xva, "X_test": Xte, "model": model_path},
        "metrics_test": metrics,
        "note": "Efficient streaming build for 8GB RAM"
    }
    Path("security").mkdir(exist_ok=True)
    Path("security/MANIFEST_min.json").write_text(json.dumps(manifest, indent=2))
    print("[OK] Wrote security/MANIFEST_min.json")
    print("[DONE]")

if __name__ == "__main__":
    main()
