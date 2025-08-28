# ml/eval_from_features.py
import numpy as np, joblib
from pathlib import Path

def ang_err_deg(pred_deg, gt_deg):
    diff = np.abs(pred_deg - gt_deg) % 360.0
    diff[diff > 180] = 360 - diff[diff > 180]
    return diff

def main():
    featdir = Path("datasets/features")

    # ---- Load TEST features + labels ----
    X_mm = np.memmap(featdir / "X_test.dat", dtype=np.float32, mode="r")
    yR = np.load(featdir / "yR_test.npy")
    yV = np.load(featdir / "yV_test.npy")
    yT = np.load(featdir / "yT_test.npy")

    N = yR.shape[0]; D = X_mm.size // N
    X = np.array(X_mm).reshape(N, D)

    # ---- Load trained model ----
    mdl = joblib.load("ml/baseline.joblib")
    R, V, Ts, Tc = mdl["R"], mdl["V"], mdl["T_sin"], mdl["T_cos"]

    # ---- Predict & metrics ----
    r = R.predict(X); v = V.predict(X)
    s = Ts.predict(X); c = Tc.predict(X)
    theta = np.degrees(np.arctan2(s, c))

    maeR = float(np.mean(np.abs(r - yR)))
    maeV = float(np.mean(np.abs(v - yV)))
    maeT = float(np.mean(ang_err_deg(theta, yT)))
    print({"MAE_R(m)": maeR, "MAE_V(m/s)": maeV, "MAE_Theta(deg)": maeT, "N": int(N)})

if __name__ == "__main__":
    main()
