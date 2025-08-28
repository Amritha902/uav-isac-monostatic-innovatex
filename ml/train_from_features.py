# ml/train_from_features.py
import numpy as np, joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

def main():
    featdir = Path("datasets/features")

    # ---- Load training features + labels ----
    X_mm = np.memmap(featdir / "X_train.dat", dtype=np.float32, mode="r")
    yR = np.load(featdir / "yR_train.npy")
    yV = np.load(featdir / "yV_train.npy")
    yT = np.load(featdir / "yT_train.npy")

    N = yR.shape[0]
    D = X_mm.size // N
    X = np.array(X_mm).reshape(N, D)

    # ---- Define models ----
    R  = RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1)
    V  = RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1)
    Ts = RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1)
    Tc = RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1)

    # ---- Train ----
    R.fit(X, yR)
    V.fit(X, yV)
    Ts.fit(X, np.sin(np.deg2rad(yT)))
    Tc.fit(X, np.cos(np.deg2rad(yT)))

    # ---- Save model bundle ----
    Path("ml").mkdir(exist_ok=True)
    joblib.dump({"R": R, "V": V, "T_sin": Ts, "T_cos": Tc}, "ml/baseline.joblib")
    print("[OK] Trained and saved model -> ml/baseline.joblib")

if __name__ == "__main__":
    main()
