# Data & Énergie

## Project Overview

**Data & Énergie** is a data science project focused on the analysis, classification, prediction, and generation of residential energy consumption profiles.

The project uses half-hourly electricity consumption data and customer labels to study consumption behavior across different types of residences. The workflow combines exploratory data analysis, clustering, supervised classification, time-series forecasting, generative modeling, and an interactive Streamlit dashboard.

The main objective is to transform raw consumption data into meaningful insights and predictive tools that can support energy analysis and decision-making.

---

## Main Objectives

The project aims to:

* Analyze electricity consumption patterns over time.
* Identify typical customer consumption profiles through clustering.
* Classify customers according to their residence type.
* Forecast short-term energy consumption using machine learning and deep learning models.
* Generate synthetic consumption curves for analysis and experimentation.
* Provide an interactive dashboard to explore the data, models, and results.

---

## Dataset

The project uses two main data files stored in the `Data/` folder:

```text
Data/
├── RES2-6-9.csv
└── RES2-6-9-labels.csv
```

The consumption dataset contains half-hourly measurements for residential customers.
The labels dataset contains customer identifiers, residence labels, and cluster information.

Main information used in the project includes:

* Customer identifier
* Timestamp
* Consumption value
* Residence type label
* Cluster assignment

---

## Project Structure

```text
Data-et-energie/
│
├── APP/
│   └── app.py                         # Streamlit dashboard
│
├── Data/
│   ├── RES2-6-9.csv                   # Energy consumption dataset
│   └── RES2-6-9-labels.csv            # Customer labels and clusters
│
├── Notebooks/
│   ├── 01_clustering.ipynb            # Customer profile clustering
│   ├── 02_classification.ipynb        # Residence type classification
│   ├── 03_prediction.ipynb            # Energy consumption forecasting
│   └── 04_generation.ipynb            # Synthetic consumption generation
│
├── outputs/
│   ├── figures/                       # Generated visualizations
│   └── results/                       # CSV result files
│
├── README.md
├── Readme.pdf
└── .gitignore
└── requirements.txt
```

---

## Methodology

### 1. Data Preprocessing

The preprocessing step includes:

* Loading raw consumption and label data.
* Checking missing values and duplicated records.
* Converting timestamps into exploitable time-based features.
* Preparing customer-level and time-window-based datasets.
* Creating features related to time, seasonality, residence type, and consumption history.

### 2. Exploratory Data Analysis

Exploratory analysis is used to understand the structure of the data and detect consumption patterns.

The analysis includes:

* Consumption distribution.
* Daily and weekly consumption patterns.
* Differences between residence types.
* Seasonal and time-of-day effects.
* Customer variability and activity levels.

### 3. Clustering

The clustering notebook identifies groups of customers with similar consumption behavior.

The workflow includes:

* Feature extraction from consumption profiles.
* Standardization of customer-level features.
* K-Means clustering.
* Silhouette score analysis.
* PCA visualization.
* Interpretation of typical intraday profiles.

Generated outputs include clustering visualizations such as silhouette plots, PCA projections, and intraday consumption profiles.

### 4. Classification

The classification notebook aims to predict the residence type of customers using their consumption behavior.

The workflow includes:

* Feature engineering from consumption time series.
* Train/test splitting.
* Model comparison.
* Evaluation using classification metrics.
* Confusion matrix visualization.
* Feature importance analysis.

Several machine learning approaches are tested, including classical models and neural-network-based methods.

### 5. Prediction

The prediction notebook focuses on short-term energy forecasting.

The forecasting task is defined as:

* Input: past consumption history.
* Output: future consumption over a short prediction horizon.

The approach compares baseline methods and machine learning models, including:

* Recent consumption baseline.
* Weekly consumption baseline.
* Ridge regression.
* LightGBM résiduel
* Deep learning models such as CNN-based architectures.

The models are evaluated using forecasting metrics such as:

* WMAPE
* MAE
* RMSE
* sMAPE
* Error by prediction horizon
* Error by time of day

### 6. Generation

The generation notebook explores the creation of synthetic consumption curves.

The objective is to generate realistic energy consumption profiles that can be used for analysis, experimentation, or data augmentation.

The workflow includes deep learning models and generated output curves saved in the `outputs/results/` folder.

### 7. Interactive Dashboard

The Streamlit application provides an interactive interface to explore the project results.

The dashboard includes:

* Dataset overview.
* Consumption visualizations.
* Clustering results.
* Classification results.
* Prediction results.
* Generated figures and summary indicators.

---

## Outputs

The project generates several outputs stored in the `outputs/` folder.

```text
outputs/
├── figures/
│   ├── clustering/
│   ├── classification/
│   ├── prediction/
│   └── generation/
│
└── results/
    ├── cluster_results.csv
    └── courbes_generees_cvae.csv
```

Examples of generated outputs include:

* Clustering silhouette plot.
* PCA visualization of customer profiles.
* Intraday consumption profiles.
* Classification confusion matrices.
* Feature importance plots.
* Prediction curves.
* Error analysis by hour.
* Generated synthetic consumption curves.

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/MariamRekik/Data-et-energie.git
cd Data-et-energie
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies


```bash
pip install -r requirements.txt
```

### 4. Launch the notebooks

```bash
jupyter notebook
```

Then open the notebooks from the `Notebooks/` folder and run them in order:

```text
01_clustering.ipynb
02_classification.ipynb
03_prediction.ipynb
04_generation.ipynb
```

### 5. Launch the Streamlit dashboard

```bash
streamlit run APP/app.py
```

The dashboard will open locally in the browser, usually at:

```text
http://localhost:8501
```

---

## Results Summary

The project provides:

* A structured analysis of residential electricity consumption.
* Customer segmentation based on consumption behavior.
* Classification models for residence type prediction.
* Short-term forecasting models for future energy consumption.
* Synthetic consumption curve generation.
* An interactive dashboard for visualization and interpretation.

The results are mainly intended for academic analysis, model comparison, and understanding energy consumption patterns.

