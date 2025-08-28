import numpy as np

def _integral_image(A):
    return np.cumsum(np.cumsum(A, axis=0), axis=1)

def _rect_sum(ii, r0, c0, r1, c1):
    r0 = max(r0, 0); c0 = max(c0, 0)
    r1 = min(r1, ii.shape[0]-1); c1 = min(c1, ii.shape[1]-1)
    A = ii
    s = A[r1, c1]
    if r0>0: s -= A[r0-1, c1]
    if c0>0: s -= A[r1, c0-1]
    if r0>0 and c0>0: s += A[r0-1, c0-1]
    return s

def ca_cfar_2d(RD_mag, guard=(2,2), train=(8,8), pfa=1e-3):
    G_r, G_c = guard
    T_r, T_c = train
    H, W = RD_mag.shape
    power = RD_mag**2
    ii = _integral_image(power)
    det = np.zeros_like(RD_mag, dtype=bool)
    thr = np.zeros_like(RD_mag, dtype=float)
    nmean = np.zeros_like(RD_mag, dtype=float)
    N_train = (2*T_r+1)*(2*T_c+1) - (2*G_r+1)*(2*G_c+1)
    alpha = N_train * (pfa ** (-1.0 / max(N_train,1)) - 1.0)
    for r in range(H):
        for c in range(W):
            R0, C0 = r-(T_r+G_r), c-(T_c+G_c)
            R1, C1 = r+(T_r+G_r), c+(T_c+G_c)
            total = _rect_sum(ii, R0, C0, R1, C1)
            r0, c0 = r-G_r, c-G_c
            r1, c1 = r+G_r, c+G_c
            guard_cut = _rect_sum(ii, r0, c0, r1, c1)
            train_sum = max(total - guard_cut, 0.0)
            noise = train_sum / max(N_train, 1)
            nmean[r, c] = noise
            thr[r, c] = alpha * noise
            det[r, c] = power[r, c] > thr[r, c]
    return det, thr, nmean

def nms_peaks(RD_mag, det_mask, max_peaks=5, radius=3):
    coords = np.argwhere(det_mask)
    vals = [RD_mag[r,c] for r,c in coords]
    order = np.argsort(vals)[::-1]
    picked = []
    sup = np.zeros_like(det_mask, dtype=bool)
    for idx in order:
        r,c = coords[idx]
        if sup[r,c]:
            continue
        picked.append((r,c,float(RD_mag[r,c])))
        r0, r1 = max(0,r-radius), min(RD_mag.shape[0]-1, r+radius)
        c0, c1 = max(0,c-radius), min(RD_mag.shape[1]-1, c+radius)
        sup[r0:r1+1, c0:c1+1] = True
        if len(picked) >= max_peaks:
            break
    return picked
