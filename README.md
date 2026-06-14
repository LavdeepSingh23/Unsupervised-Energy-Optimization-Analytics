# Unsupervised Energy Optimization Analytics

AI-powered portfolio segmentation for buildings using PCA + KMeans, with an interactive Streamlit dashboard for cluster quality, segment profiling, anomaly detection, recommendations, and export.

## Overview
This project analyzes energy behavior across buildings without labeled targets (unsupervised learning).  
It groups buildings with similar operational/efficiency patterns and translates those clusters into actionable business insights.

The workflow is:
1. Clean and engineer energy features.
2. Reduce dimensionality with PCA.
3. Run KMeans clustering (interactive `k` in dashboard).
4. Evaluate quality (Silhouette + Calinski-Harabasz + inertia/elbow trend).
5. Expose insights through a production-style Streamlit app.

## Key Features
- Interactive KMeans with adjustable cluster count (`k=2..10`).
- Cluster quality diagnostics:
  - Mean silhouette score
  - Per-sample silhouette distribution
  - Calinski-Harabasz index
  - Inertia trend across `k`
- Cluster explorer (PCA 2D map + segment-level summaries).
- Segment profiles (radar-style standardized feature signatures).
- Rule-based recommendations engine with estimated savings impact.
- Business impact views (cost vs savings by segment).
- Z-score anomaly detector for high-cost outliers.
- Filtered CSV export for downstream reporting.

## Live App (Streamlit Cloud)
If deployed, add your public app URL here:

`https://<your-streamlit-app>.streamlit.app`

## Repository Structure
```text
.
├── energy_cluster_dashboard/
│   ├── app.py
│   ├── requirements.txt
│   ├── pca_transformed_data.csv
│   ├── final_feature_engineered.csv
│   ├── final_cluster_summary.csv
│   ├── cluster_recommendations.csv
│   └── secrets.toml
├── data_cleaning.ipynb
├── feature_eng.ipynb
├── Pca.ipynb
├── k_means_clustering.ipynb
├── kmeans_final_clustering.ipynb
├── advanced_clustering.ipynb
├── analysis.ipynb
├── cleaned_unscaled.csv
├── electricity_consumption_optimization_dataset.csv
└── README.md
```

## Data & Feature Context
Primary dashboard input files:
- `energy_cluster_dashboard/final_feature_engineered.csv`
- `energy_cluster_dashboard/pca_transformed_data.csv`

Representative fields used in analysis:
- `Energy Consumption (kWh)`
- `Energy Price ($/kWh)`
- `Energy_Intensity`
- `HVAC_Ratio`
- `Lighting_Ratio`
- `Renewable_Ratio`
- `Carbon_Load`
- `Efficiency_Gap`
- `Savings_Gap`
- `Grid_Stress`
- `Temp_Deviation`

Derived dashboard metrics include:
- `Energy_Cost = Energy Consumption (kWh) * Energy Price ($/kWh)`
- `Potential_Savings_Value = Savings_Gap * Energy Consumption (kWh)`

## Methodology
### 1. Preprocessing
- Raw data is cleaned and transformed in notebooks.
- Features are engineered to represent energy efficiency and load behavior.

### 2. PCA
- Dimensionality reduction creates principal components (`PC1`, `PC2`, ...).
- Dashboard computes component variance contribution as a PCA quality proxy.

### 3. Clustering (KMeans)
- Model: `KMeans(random_state=42, n_init=20)` (core app path).
- `k` is user-controlled in the dashboard sidebar.
- Cluster labels are mapped to segment names dynamically.

### 4. Validation
- Silhouette average and per-point silhouette.
- Calinski-Harabasz score.
- Inertia across k-range for elbow interpretation.

### 5. Decision Layer
- Rule-based recommendation engine compares segment means vs portfolio means.
- Recommendations include priority, rationale, and estimated savings impact.

## Dashboard Pages
- `Overview`: Portfolio KPIs and cluster distribution.
- `Cluster Quality`: Silhouette/CH/inertia diagnostics.
- `Cluster Explorer`: PCA cluster scatter and segment stats.
- `Segment Profiles`: Deep dive for each segment.
- `Recommendations`: Portfolio-wide and segment-specific actions.
- `Business Impact`: Cost and savings breakdown.
- `Anomaly Detector`: Z-score based outlier identification.
- `Export Report`: Segment-filtered CSV download.

## Local Setup
### Prerequisites
- Python 3.9+ recommended
- `pip`

### Install
```bash
cd "energy_cluster_dashboard"
pip install -r requirements.txt
```

### Run
```bash
streamlit run app.py
```

Open the URL shown in terminal (usually `http://localhost:8501`).

## Streamlit Cloud Deployment Guide
### 1. Push to GitHub
- Ensure this repository is pushed to your GitHub account/org.

### 2. Create App in Streamlit Cloud
- Go to Streamlit Community Cloud.
- Select repo, branch, and set:
  - App file: `energy_cluster_dashboard/app.py`

### 3. Dependencies
- Keep `energy_cluster_dashboard/requirements.txt` updated.
- If Cloud fails to detect dependencies, add a root `requirements.txt` mirror.

### 4. Secrets
- Do not commit real secrets.
- Use Streamlit Cloud Secrets manager for production values.
- `energy_cluster_dashboard/secrets.toml` can be used locally as a template.

### 5. Redeploy
- Every push to selected branch triggers rebuild/redeploy.

## Configuration Notes
- `app.py` loads CSVs relative to its own directory, which is Cloud-safe.
- Default visual style is a custom dark-theme UI.
- Caching is used with `@st.cache_data` for performance.

## Troubleshooting
- `FileNotFoundError` on startup:
  - Confirm the two CSV inputs exist in `energy_cluster_dashboard/`.
- Slow first load:
  - Expected on cold starts; model/data caches warm after first run.
- Unicode symbol rendering issues in terminal:
  - Usually terminal encoding only; app rendering in browser is unaffected.
- Streamlit Cloud build issues:
  - Re-check `requirements.txt` compatibility and package names.

## Suggested Improvements
- Add unit tests for recommendation-rule logic.
- Add CI checks (`py_compile`, lint, notebook smoke tests).
- Track model drift and periodic retraining policy.
- Add role-based auth fully via Streamlit Secrets.
- Add explainability panel for cluster assignment confidence/proximity.

## License
Add your preferred license (MIT/Apache-2.0/etc.) in a `LICENSE` file.

## Author
Project by Lavdeepsingh (Enerlytics AI concept and implementation).
