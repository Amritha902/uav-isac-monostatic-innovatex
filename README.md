# Samsung EnnovateX 2025 AI Challenge Submission

**Problem Statement**  
Estimation of UAV Parameters Using Monostatic Sensing in ISAC Scenario  
Develop an AI-based solution using monostatic integrated sensing and communication (ISAC) to estimate UAV range, velocity, and direction of arrival, leveraging advanced signal processing and machine learning. Utilize the channel model based on 3GPP TR 38.901-j00 (Rel-19) Section 7.9 for ISAC applications. Participants are expected to design models that extract these parameters from ISAC signals under the specified channel conditions.
App deployed link: https://amritha902-uav-isac-monostatic-innovatex-app-udc1zu.streamlit.app/
---

**Team Name**  
UAV-ISAC Monostatic Innovatex

**Team Members**  
- Amritha S  
- Yugeshwaran P  

## ✨ Features Implemented

1. **Signal Simulation**  
   Monostatic FMCW radar waveform generation.  
   Channel model based on **3GPP TR 38.901 Rel-19 §7.9** (urban ISAC scenario).  
   Configurable parameters: carrier frequency, bandwidth, chirp duration, sampling frequency, SNR, multipath, number of UAV targets, number of antennas.  

2. **Range–Doppler Processing**  
   FFT-based Range–Doppler map computation.  
   Optional **Gaussian denoiser** for noisy/low-SNR cases.  
   Visualization with interactive heatmaps.  

3. **Detection Pipeline**  
   2D CA-CFAR detector with configurable guard & training cells.  
   Adjustable probability of false alarm (Pfa).  
   Non-Maximum Suppression (NMS) to filter out duplicate peaks.  
   Real-time UAV detection displayed on Range–Doppler plots.  

4. **Direction of Arrival (DoA) Estimation**  
   MUSIC algorithm for high-resolution angle estimation.  
   Uniform Linear Array (ULA) support with configurable spacing.  
   Pseudo-spectrum plotting for detected UAV angles.  

5. **Adaptive Waveform Optimization**  
   Dynamic optimization of **bandwidth (B)** and **chirp duration (T_chirp)**.  
   Cost function combines **Cramér-Rao Lower Bound (CRLB)** + **Communication rate**.  
   Enables real-time **trade-off** between sensing accuracy and communication throughput.  

6. **Machine Learning Integration**  
   Feature extraction from Range–Doppler maps.  
   RandomForest baseline for UAV parameter regression (Range, Velocity, DoA).  
   Pre-trained model (`baseline.joblib`) included.  
   Training and evaluation scripts provided for reproducibility.  

7. **CRLB Performance Evaluation**  
   Computes **Cramér-Rao Lower Bounds** for Range, Velocity, and DoA.  
   Plots **CRLB vs SNR** alongside observed Mean Absolute Error (MAE).  
   Provides **scientific validation** of the solution.  

8. **Security & Provenance (Future-Ready)**  
   SHA-256 hashing of dataset files for integrity verification.  
   Merkle root manifests for scalable, blockchain-ready provenance.  
   Ensures datasets remain authentic and tamper-proof.  

9. **Interactive UI (Streamlit App)**  
   Full end-to-end UI built in Streamlit.  
   Sidebar for simulation controls & adaptive optimization.  
   Live plots for: Range–Doppler Map, MUSIC Spectrum, CRLB vs SNR.  
   Results table comparing detected UAV parameters vs ground truth.  

10. **Evaluation & Metrics**  
    Multi-frame evaluation pipeline.  
    Metrics include **MAE for Range, Velocity, DoA**, plus count of missed detections.  
    Judges can quickly verify robustness under different SNR and multipath conditions.  


## 📂 Repository Structure & File Explanations

- **app.py** – Streamlit UI for running full simulation, adaptive waveform optimization, CRLB plotting, CFAR detection, and DoA estimation.  
- **simulate_isac.py** – Core radar signal simulator (monostatic FMCW + 3GPP TR 38.901 channel model).  
- **estimators/**
  - `dsp_baseline.py` – Range–Doppler map and signal chain functions.  
  - `cfar.py` – 2D CA-CFAR detector.  
  - `peak_utils.py` – Peak picking, bin-to-physical conversion, SNR/confidence estimation.  
  - `music_doa.py` – MUSIC algorithm for DoA estimation.  
- **metrics/crlb.py** – Cramér-Rao Lower Bound (CRLB) computations for Range, Velocity, and DoA.  
- **optim/adaptive_isac.py** – Adaptive waveform optimizer (tunes chirp bandwidth & duration to minimize CRLB + comm rate cost).  
- **ml/**
  - `train_baseline.py` – Trains RandomForest regressors for range, velocity, and DoA (from features).  
  - `eval_from_features.py` – Evaluates trained model on test features.  
  - `train_from_features.py` – Alternative training script using saved `.dat` features.  
  - `baseline.joblib` – Pre-trained RandomForest model (~56 MB).  
- **security/**
  - `provenance.py` – SHA-256 hashing & Merkle root generation for dataset integrity.  
  - `MANIFEST_min.json` – Example security manifest output.  
- **requirements.txt** – Dependencies for reproducibility.  

---




## 📸 Screenshots



1. **Streamlit UI – Simulation Parameters**  
   <img width="1919" height="908" alt="image" src="https://github.com/user-attachments/assets/9a9e7016-7f0c-4f00-a6b4-c08179c34f53" />


2. **Range–Doppler Map with Peaks**  
   ![RD Map](docs/screenshots/rd_map.png)

3. **MUSIC Spectrum Plot**  
   ![MUSIC Spectrum](docs/screenshots/music_spectrum.png)

4. **CRLB vs MAE Comparison**  
   ![CRLB Plot](docs/screenshots/crlb_plot.png)

---

## 🚀 Quickstart

```bash
# Setup environment
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```
## 🏗️ Architecture Overview

Input: FMCW Monostatic ISAC signals simulated via 3GPP TR 38.901-j00.

DSP Chain: Range–Doppler processing → CFAR → Peak picking.

Angle Estimation: MUSIC DoA estimator.

ML Module: RandomForest baseline for regression of UAV parameters.

Adaptive Optimization: Waveform tuning (B, T_chirp) minimizing CRLB + rate cost.

Evaluation: MAE metrics vs CRLB bounds.

Security Layer: Provenance hashing for dataset authenticity.

## 📚 Attribution

Channel modeling aligned with 3GPP TR 38.901 Rel-19 Section 7.9.

Our contributions:

Adaptive ISAC waveform optimization.

Hybrid DSP + ML approach for UAV parameter estimation.

CRLB-grounded evaluation.

Security & provenance integration.




