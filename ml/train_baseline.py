import numpy as np
import argparse, joblib
from sklearn.ensemble import RandomForestRegressor
from estimators.dsp_baseline import range_doppler_map

def features_from_cube(cube):
    RD = range_doppler_map(cube)
    x = RD / (RD.max() + 1e-9)
    return x.flatten()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, required=True)
    ap.add_argument("--out", dest="outp", type=str, default="ml/baseline.joblib")
    args = ap.parse_args()

    data = np.load(args.inp, allow_pickle=True)
    cubes = data["cubes"]
    labels = data["labels"]
    X, yR, yV, yT = [], [], [], []
    for cube, lab in zip(cubes, labels):
        X.append(features_from_cube(cube))
        pi = lab["primary_index"]
        yR.append(lab["ranges_m"][pi])
        yV.append(lab["vels_mps"][pi])
        yT.append(lab["thetas_deg"][pi])

    X = np.array(X)
    from sklearn.ensemble import RandomForestRegressor
    R = RandomForestRegressor(n_estimators=200, random_state=0)
    V = RandomForestRegressor(n_estimators=200, random_state=0)
    T_sin = RandomForestRegressor(n_estimators=200, random_state=0)
    T_cos = RandomForestRegressor(n_estimators=200, random_state=0)

    R.fit(X, yR)
    V.fit(X, yV)
    ysin = np.sin(np.deg2rad(yT))
    ycos = np.cos(np.deg2rad(yT))
    T_sin.fit(X, ysin)
    T_cos.fit(X, ycos)

    joblib.dump({"R": R, "V": V, "T_sin": T_sin, "T_cos": T_cos}, args.outp)
    print("Saved model to", args.outp)

if __name__ == "__main__":
    main()
