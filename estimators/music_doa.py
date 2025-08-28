import numpy as np

def steering_vector_ula(theta_deg, M, d, wavelength):
    theta = np.deg2rad(theta_deg)
    k = 2*np.pi / wavelength
    m = np.arange(M)
    return np.exp(1j * k * m * d * np.sin(theta))

def estimate_doa_music(snapshot_vecs, num_sources=1, M=8, d=0.5, wavelength=1.0, angle_grid=np.linspace(-90, 90, 721)):
    X = np.asarray(snapshot_vecs)
    if X.ndim == 1:
        X = X[None, :]
    R = (X.conj().T @ X) / X.shape[0]
    vals, vecs = np.linalg.eigh(R)
    En = vecs[:, :M-num_sources] if num_sources < M else vecs[:, :1]
    P = []
    for ang in angle_grid:
        a = steering_vector_ula(ang, M, d, wavelength)[:, None]
        denom = np.linalg.norm(En.conj().T @ a)**2
        P.append(1.0/denom if denom > 1e-12 else 1e12)
    P = np.array(P).real
    est_angles = list(angle_grid[np.argsort(P)[-num_sources:]][::-1])
    return est_angles, P
