# Enerlytics AI

**Unsupervised Energy Portfolio Segmentation & Optimization Analytics**

Production-ready machine learning system that segments large building portfolios using PCA + KMeans to uncover inefficiencies, cost drivers, and savings opportunities — deployed via an interactive Streamlit dashboard.

---

## 🚀 Problem Statement

Large energy portfolios often lack labeled optimization targets.
Without supervision, traditional modeling cannot identify where efficiency gaps exist.

Enerlytics AI solves this using unsupervised learning to:

* Discover latent energy behavior patterns
* Segment buildings by operational similarity
* Quantify potential savings per segment
* Translate cluster insights into actionable strategies

---

## 📊 Dataset Scope

* 52,000+ buildings
* 10+ engineered energy efficiency features
* PCA-based dimensionality reduction
* Dynamic clustering (`k = 2–10`) via dashboard control

---

## 🧠 Methodology Overview

### 1️⃣ Feature Engineering

Energy and efficiency indicators derived from raw consumption and operational metrics:

* Energy Intensity
* HVAC Ratio
* Renewable Ratio
* Carbon Load
* Efficiency Gap
* Savings Gap
* Grid Stress
* Temperature Deviation

---

### 2️⃣ Dimensionality Reduction (PCA)

* Reduced correlated feature space into orthogonal components
* Preserved majority variance while simplifying cluster geometry
* Variance contribution computed and displayed in dashboard

---

### 3️⃣ Clustering (KMeans)

* Model: `KMeans(random_state=42, n_init=20)`
* Interactive cluster count selection
* Segment naming based on centroid behavior

---

### 4️⃣ Cluster Validation

Multiple quality diagnostics:

* Mean Silhouette Score
* Per-sample silhouette distribution
* Calinski-Harabasz Index
* Inertia trend across k (elbow interpretation)

---

### 5️⃣ Business Translation Layer

Clusters are converted into:

* Segment-level profiles
* Cost concentration analysis
* Estimated savings impact
* Rule-based optimization recommendations

---

## 📈 Dashboard Capabilities

* Portfolio KPIs overview
* Cluster quality diagnostics
* PCA 2D interactive cluster map
* Segment-level radar signatures
* Business impact analysis (cost vs savings)
* Z-score anomaly detection
* CSV export for downstream reporting

---

## 🏗 System Architecture

Raw Dataset
→ Feature Engineering (Notebooks)
→ PCA Transformation
→ KMeans Clustering
→ Validation Metrics
→ Recommendation Engine
→ Streamlit Dashboard (Cloud Deployment)

---

## 📌 Why Unsupervised Learning?

The dataset does not contain labeled optimization outcomes.

Clustering enables:

* Discovery of hidden operational segments
* Strategic portfolio segmentation
* Data-driven prioritization
* Scalable optimization planning

---

## 💡 Key Outcomes

* Identified 4 distinct energy behavior segments
* Clear geometric separation validated via Silhouette & CH metrics
* Quantified potential savings per segment
* Enabled portfolio-level prioritization framework

---

## 🖥 Live App

[https://enerlytics-ai.streamlit.app](https://enerlytics-ai.streamlit.app)

---

## 🛠 Tech Stack

* Python
* Pandas / NumPy
* Scikit-learn
* PCA
* KMeans
* Streamlit
* Plotly
* Matplotlib

---

## 📂 Repository Structure

```
energy_cluster_dashboard/
│
├── app.py
├── requirements.txt
├── final_feature_engineered.csv
├── pca_transformed_data.csv
├── final_cluster_summary.csv
├── cluster_recommendations.csv
```

Notebooks:

* Data Cleaning
* Feature Engineering
* PCA Analysis
* KMeans Modeling
* Advanced Clustering
* Final Analysis

---

## ▶ Local Run

```
cd energy_cluster_dashboard
pip install -r requirements.txt
streamlit run app.py
```

---

## 📦 Deployment

Deployed via Streamlit Community Cloud.
Auto-redeploy on GitHub push.

---

## 📈 Future Enhancements

* Automated model retraining pipeline
* Cluster stability resampling
* Proximity-based confidence scoring
* SHAP-based feature contribution explanation
* CI integration and test coverage

---

## 👤 Author

Lavdeep Singh
Machine Learning | Energy Analytics | Applied AI Systems

