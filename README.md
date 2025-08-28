# UAV-ISAC Monostatic (v3) — Adaptive + CRLB + Security

**What’s new vs v2**
- Adaptive ISAC optimization (choose slope,B) minimizing cost = w1*CRLB_sum + w2*(-CommRate)
- CRLB module with MAE vs CRLB plots across SNR
- Simple denoiser (Gaussian filter) toggle before CFAR (CPU-only, no torch)
- Security stubs: dataset manifest (SHA-256), ECDSA sign/verify (if crypto present), Merkle-like ledger

## Quickstart (Windows PowerShell)
```
cd "<path>\uav_isac_win_kit_v3"
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
streamlit run app.py
```
