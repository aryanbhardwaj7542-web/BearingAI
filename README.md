# ⚙️ BearingAI: Intelligent Machine Fault Detection

An AI-powered **bearing condition monitoring** dashboard built with Streamlit. It extracts statistical features from vibration signals in the **CWRU Bearing Dataset**, trains machine learning classifiers to recognise bearing faults, and presents the results through an interactive multi-page web app.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Charts-3F4F75?logo=plotly&logoColor=white)
![Status](https://img.shields.io/badge/Status-Educational%20Prototype-yellow)

> ⚠️ **Disclaimer:** BearingAI is an educational prototype built on a public lab dataset. It is not a validated industrial diagnostic tool, and its metrics should not be presented as real-world accuracy.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [How It Works](#-how-it-works)
- [Getting Started](#-getting-started)
- [App Pages](#-app-pages)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

Rolling-element bearings are among the most common failure points in rotating machinery. Faults leave signatures in vibration data, such as higher energy, sharper spikes and changes in signal shape. BearingAI turns those signatures into numbers and uses them to classify a bearing as healthy or as having one of three fault types.

**Fault classes:**

| Class | Description |
|---|---|
| 🟢 Normal | Healthy bearing |
| 🟠 Ball Fault | Defect on a rolling element |
| 🔴 Inner Race Fault | Defect on the inner ring |
| 🟣 Outer Race Fault | Defect on the outer ring |

---

## ✨ Features

- 🧮 **Feature extraction** from raw vibration signals using fixed 2048-sample windows
- 🤖 **Two classifiers compared**: Random Forest and an RBF-kernel SVM
- 📊 **Machine overview dashboard** with class distribution and per-feature box plots
- 📈 **Vibration analysis** with a time-domain waveform and a frequency spectrum (Welch power spectral density)
- 🔬 **AI fault prediction** with the predicted class, confidence, and per-class probabilities
- 🧠 **Model performance page** with accuracy, precision, recall, F1, classification reports and confusion matrices
- 🛡️ **Honest evaluation notes** built into the app, warning about optimistic metrics

---

## 🛠 Tech Stack

- **Language:** Python
- **Web app:** [Streamlit](https://streamlit.io/)
- **Machine learning:** scikit-learn (`RandomForestClassifier`, `SVC`, `StandardScaler`, cross-validation)
- **Signal processing:** SciPy (`loadmat`, `welch`, `kurtosis`, `skew`)
- **Data handling:** pandas, NumPy
- **Visualization:** Plotly (app), Matplotlib (saved confusion matrices)
- **Model persistence:** joblib

---

## 📂 Project Structure

```
BearingAI/
├── app.py              # Streamlit dashboard (4 pages)
├── extract_data.py     # Raw .mat signals → windowed features (bearing_features.csv)
├── train_model.py      # Cross-validated model comparison + final model (bearing_model.joblib)
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

**Files you provide or generate locally** (excluded by `.gitignore`):

| File | Source |
|---|---|
| `Normal_0.mat`, `B007_0.mat`, `IR007_0.mat`, `OR007_0.mat` | Downloaded from the CWRU Bearing Data Center |
| `bearing_features.csv` | Created by `extract_data.py` |
| `bearing_model.joblib` | Created by `train_model.py` |
| `model_comparison.csv` | Created by `train_model.py` |
| `random_forest_confusion_matrix.png`, `svm_confusion_matrix.png` | Created by `train_model.py` |

---

## 🗃 Dataset

**Case Western Reserve University (CWRU) Bearing Data Center**, a widely used benchmark for bearing fault diagnosis.

The project uses four drive-end (`DE`) vibration recordings:

| Label | File | Signal key in file |
|---|---|---|
| Normal | `Normal_0.mat` | `X097_DE_time` |
| Ball Fault | `B007_0.mat` | `X118_DE_time` |
| Inner Race Fault | `IR007_0.mat` | `X105_DE_time` |
| Outer Race Fault | `OR007_0.mat` | `X130_DE_time` |

The app assumes a **12 kHz sampling rate** when drawing the frequency spectrum. Verify this against the dataset documentation before drawing engineering conclusions.

---

## ⚙️ How It Works

### 1. Feature extraction (`extract_data.py`)

Each recording is split into **non-overlapping windows of 2048 samples**. For every window, six statistical features are computed:

| Feature | What it captures |
|---|---|
| `rms` | Overall vibration energy |
| `std` | Spread of the signal |
| `kurtosis` | "Spikiness", which is sensitive to impacts from defects |
| `skewness` | Asymmetry of the amplitude distribution |
| `peak_to_peak` | Total amplitude range |
| `crest_factor` | Peak amplitude relative to RMS |

Non-finite values are removed, and the result is saved to `bearing_features.csv` along with the fault label, source file and window start position.

### 2. Model training (`train_model.py`)

Two models are compared using **5-fold stratified cross-validation** (`shuffle=True`, `random_state=42`):

| Model | Configuration |
|---|---|
| Random Forest | 300 trees, `class_weight="balanced"` |
| SVM | `StandardScaler` + RBF kernel, `class_weight="balanced"` |

For each model the script prints accuracy, macro precision, macro recall and macro F1, plus a classification report, and saves a confusion-matrix image and `model_comparison.csv`. A final **Random Forest** is then trained on all samples and saved as `bearing_model.joblib`.

### 3. Dashboard (`app.py`)

Loads the features and the saved model and serves four pages (described below).

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or newer
- A recent version of Streamlit (the app uses the newer `width="stretch"` option for charts)

### 1. Clone the repository

```bash
git clone https://github.com/aryanbhardwaj7542-web/BearingAI.git
cd BearingAI
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download the four CWRU `.mat` files listed in the [Dataset](#-dataset) section and place them in the project root, next to `app.py`. The **Vibration Analysis** page also reads these files at runtime, so keep them there.

### 5. Extract features

```bash
python extract_data.py
```

### 6. Train the models

```bash
python train_model.py
```

### 7. Launch the app

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

---

## 🖥 App Pages

| Page | What you can do |
|---|---|
| **Machine Overview** | See total, normal and fault window counts, the class distribution, and box plots for any extracted feature |
| **Vibration Analysis** | Pick a machine condition, slide through the raw signal, and view the waveform and frequency spectrum |
| **AI Fault Prediction** | Choose a labelled sample and compare the model's predicted class and confidence against its dataset label |
| **Model Performance** | Run cross-validated evaluation of Random Forest vs SVM with metrics, reports and confusion matrices |

---

## ⚠️ Limitations

- **Optimistic metrics.** Windows from the same recording are split randomly across folds, so near-identical neighbouring windows can appear in both training and test data. Scores may be much higher than real-world performance.
- **Limited conditions.** Only four recordings are used (one operating condition and one fault size per fault type), so the model is not tested on other loads, speeds, fault sizes or machines.
- **Dataset-based predictions.** The prediction page classifies windows already in the dataset. It does not accept a new, uploaded signal.
- **Simple features.** Only six time-domain statistics are used. No frequency-domain or envelope features are included.
- **Not a diagnostic tool.** Model confidence is an estimate, not a guarantee of fault certainty.

---

## 🔮 Future Improvements

- [ ] Evaluate on held-out recordings or operating conditions (for example, train on some loads and test on others)
- [ ] Add more CWRU conditions, fault sizes and motor loads
- [ ] Add frequency-domain and envelope-analysis features
- [ ] Let users upload their own vibration signals for classification
- [ ] Try deep learning models such as 1D-CNNs
- [ ] Add remaining-useful-life estimation
- [ ] Add tests and deploy to Streamlit Community Cloud

---

## 👤 Author

**Aryan Bhardwaj**
2nd-year B.Tech student, Artificial Intelligence & Machine Learning, Dayananda Sagar University, Bengaluru

- GitHub: [@aryanbhardwaj7542-web](https://github.com/aryanbhardwaj7542-web)

---

## 🙏 Acknowledgements

- [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter) for the open bearing fault dataset
- The open-source communities behind Streamlit, scikit-learn, SciPy and Plotly

---

## 📄 License

No license has been specified yet. Add a `LICENSE` file (for example MIT) to define how others may use this project.
