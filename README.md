# ⚡ Electric Vehicle Sensor Analytics
### Predictive Machine Learning Dashboard for EV Performance, Safety & Battery Health

> **Turning raw OBD-II telemetry into actionable intelligence — without expensive hardware.**

A full-stack Python machine learning application that accepts real vehicle telemetry CSV files and runs **5 distinct predictive models** to optimize Electric Vehicle performance, safety, and battery longevity in real time.

---

## 📌 Project Overview

Modern EVs and hybrids generate thousands of sensor readings per second via their OBD-II port — yet almost none of this data is used intelligently. Current systems react to conditions *after* they happen. This project proves that a pure **software-only ML layer** can:

- **Proactively** throttle HVAC power *before* a steep hill drains the battery
- **Predict** how much free energy will be harvested via regenerative braking
- **Classify** a driver's behavior into Eco / Moderate / Aggressive in real-time
- **Detect** dangerous road hazard situations **without cameras or radar**
- **Track** micro-level battery chemical stress events before they cause permanent damage

---

## 🚀 Features

| # | Feature | Model | Goal |
|---|---------|-------|------|
| F1 | **Predictive HVAC Energy Optimizer** | Random Forest Regressor | Predict total HVAC power draw (W) and proactively throttle before battery-critical zones |
| F2 | **Gradient-Aware Regen Braking** | Gradient Boosting Regressor | Predict regenerative braking current harvestable from downhill / deceleration segments |
| F3 | **Driver Behavior & Eco-Score** | KMeans → RF Classifier | Cluster and classify driving style as Eco / Moderate / Aggressive with no labelled data |
| F4 | **Road Hazard Risk Predictor** | Random Forest Classifier | Flag Low / Medium / High collision risk using only OBD-II sensors — no camera, no radar |
| F5 | **Battery Health & Degradation Monitor** | XGBoost Classifier | Detect micro-stress degradation events in real-time based on battery physics |

---

## 🛠️ Technology Stack

| Technology | Role |
|---|---|
| **Python 3.10+** | Core language |
| **Streamlit** | Interactive web dashboard & CSV upload UI |
| **Scikit-learn** | Random Forest, Gradient Boosting, KMeans, StandardScaler |
| **XGBoost** | Battery health classification with class imbalance handling |
| **Pandas** | Full ETL pipeline — cleaning, imputation, feature engineering |
| **Plotly** | Interactive charts: histograms, scatter plots, timelines, pie charts |
| **Joblib** | Fast model serialization (3–5× faster than Pickle for NumPy arrays) |
| **NumPy** | Numerical computations and feature engineering |

---

## 📂 Project Structure

```
CAP_PROJ_2/
│
├── app.py                          # Main Streamlit dashboard
├── requirements.txt                # All Python dependencies
├── README.md
│
├── features/
│   ├── feature1_hvac_optimizer/
│   │   ├── preprocessing.py        # Data cleaning & feature engineering for F1
│   │   └── ml_model.py             # Random Forest Regressor training & evaluation
│   │
│   ├── feature2_regen_braking/
│   │   ├── preprocessing.py        # Regen-specific feature engineering
│   │   └── ml_model.py             # Gradient Boosting Regressor
│   │
│   ├── feature3_driver_behavior/
│   │   ├── preprocessing.py        # KMeans clustering + label generation
│   │   └── ml_model.py             # Random Forest Classifier
│   │
│   ├── feature4_road_hazard/
│   │   ├── preprocessing.py        # Physics-informed hazard labelling
│   │   └── ml_model.py             # Random Forest Classifier
│   │
│   └── feature5_battery_health/
│       ├── preprocessing.py        # Battery stress label creation
│       └── ml_model.py             # XGBoost Classifier
│
└── models/                         # Auto-generated on first run (gitignored)
    └── *.pkl                       # Saved trained models & scalers
```

---

## 📊 Dataset

This project uses the **eVED (Electric Vehicle Energy Dataset)**:
- ~**500,000** rows of real OBD-II telemetry (1-second samples)
- **37** sensor columns per row
- Collected from **5 real EV trips** across Michigan, USA
- Includes: speed, battery SOC, current, voltage, temperature, gradient, elevation, HVAC power, engine RPM, and more

> ⚠️ The raw CSV dataset is **not included** in this repository (too large). You must supply your own compatible OBD-II telemetry CSV through the dashboard upload interface.

**Compatible columns include:**
`Vehicle Speed[km/h]`, `HV Battery SOC[%]`, `HV Battery Current[A]`, `OAT[DegC]`, `Gradient`, `Elevation Smoothed[m]`, `Engine RPM[RPM]`, `Absolute Load[%]`, `Air Conditioning Power[Watts]`, `Heater Power[Watts]`, `Speed Limit[km/h]`, `Intersection`, and more.

---

## ⚙️ How It Works

### 1. Auto-Training Pipeline
On the **very first launch**, the dashboard automatically detects that no trained models exist and triggers background training:
1. Each feature's `preprocessing.py` is executed — cleans the dataset, engineers features, splits train/test
2. Each feature's `ml_model.py` is executed — trains the model, evaluates performance, saves `.pkl` files via Joblib
3. All subsequent launches load the saved models instantly (< 1 second)

### 2. CSV Upload & Inference
- Upload any compatible OBD-II telemetry CSV via the sidebar
- The uploaded data is passed through each feature's preprocessing pipeline
- Predictions are generated instantly using the pre-trained models
- Results are displayed as interactive Plotly charts and metric cards

### 3. Physics-Informed Labelling (No Manual Annotation Needed)
Because real accident or battery-failure records don't exist in the dataset, we construct labels using **domain physics**:
- **Hazard labels** (F4): Speed excess + intersection proximity + gradient + ice-risk temperature
- **Battery stress labels** (F5): SOC extremity + high current + thermal stress index

---

## 🖥️ How to Run

### Prerequisites
- Python **3.10 or higher**
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/ajais3000/vehicle-Energy-Intelligence.git
cd vehicle-Energy-Intelligence
```

### Step 2: Create & Activate a Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\activate

# Activate (Mac / Linux)
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Launch the Dashboard
```bash
streamlit run app.py
```

The dashboard will open automatically at **http://localhost:8501**

> 💡 **First Launch:** The app will auto-train all 5 models in the background (takes ~30–60 seconds). Subsequent launches are instant.

### Step 5: Upload Your Data
- Use the **sidebar file uploader** to upload a compatible OBD-II telemetry CSV
- Navigate between the 5 feature tabs using the sidebar menu
- Explore predictions, charts, and metrics!

---

## 📈 Model Performance

| Feature | Model | Metric | Score |
|---------|-------|--------|-------|
| F1 – HVAC Optimizer | Random Forest Regressor | R² | ~0.99 |
| F2 – Regen Braking | Gradient Boosting Regressor | R² | ~0.97 |
| F3 – Driver Behavior | RF Classifier (on KMeans labels) | Accuracy | ~94% |
| F4 – Road Hazard Risk | Random Forest Classifier | Accuracy | ~95%+ |
| F5 – Battery Health | XGBoost Classifier | Accuracy | ~93% |

---

## 🔬 Key Design Decisions

### Why Random Forest for HVAC (F1)?
HVAC demand follows a **non-linear U-curve** — the heater maxes out in extreme cold AND the AC maxes out in extreme heat. Linear models completely fail here. Random Forest handles this naturally without any mathematical reformulation.

### Why Gradient Boosting for Regen (F2)?
Regenerative braking energy spikes are sharp and non-monotonic (they depend on speed², not speed). Gradient Boosting's sequential error-correction mechanism captures these sharp energy spikes far better than parallel ensemble methods.

### Why XGBoost for Battery Health (F5)?
Only ~7% of rows are "High Stress" events. XGBoost's `scale_pos_weight` parameter forces the model to penalise itself heavily for missing rare dangerous events, while Random Forest would just predict "Low Stress" 93% of the time and call it done.

### Why Joblib over Pickle?
Our ML models contain millions of NumPy array values (decision tree splits). Joblib writes NumPy arrays directly to disk using memory-mapped files — **3–5× faster** saves/loads and **~60% smaller** file sizes than standard Pickle.

---

## 🔮 Future Scope

- [ ] Deploy to Raspberry Pi for live OBD-II port streaming
- [ ] Add LSTM / Transformer for time-series sequence prediction
- [ ] GPS map overlay for route-based hazard visualisation
- [ ] Cloud API deployment (FastAPI + Docker) for fleet telemetry
- [ ] Mobile app for driver eco-score gamification
- [ ] OTA model updates — weekly retraining on accumulated fleet data
- [ ] Weather API integration (rain, fog) for enhanced hazard prediction

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

---

## 👤 Author

**Ajai S**
- GitHub: [@ajais3000](https://github.com/ajais3000)
- Email: ajais4800@gmail.com

---

*Built as part of a Capstone Project (CAP_PROJ_2) — 2026*
