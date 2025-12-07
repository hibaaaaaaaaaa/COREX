# SPY Options Market Analysis (2020–2022)

This project looks at how SPY options activity and market sentiment changed from 2020 to 2022. We use PySpark on the Roar cluster to process over 3.6 million option quote records, build daily sentiment features, explore patterns across market phases, and train models to predict high-volatility days.

The repo includes the final notebook, Python script, and an HTML export.

---

## Research Question

**How did SPY options activity and market sentiment evolve from 2020–2022?**

### Hypotheses
1. Put/Call Ratio, implied volatility, and large-trade frequency change across pandemic, optimism, and correction phases.  
2. Simple sentiment indicators (PCR, LTF) are weak predictors of volatility on their own.  
3. Richer option features (Greeks, strike distance, bid–ask spreads, returns) improve prediction of high-volatility days.

---

## Files in This Repo

### `ClusterFinal_Mode4.ipynb`
- Main notebook with all PySpark code  
- Data cleaning  
- Daily aggregation  
- PCA  
- KMeans clustering  
- Random Forest modeling  
- Visualizations  
- ROC curve evaluation  

### `ClusterFinal_Mode4.py`
- Script version of the notebook  
- Can be run directly on the Roar cluster  
- Includes the same analysis pipeline

### `ClusterFinal_Mode4.html`
- HTML export of the notebook  
- Easy to read without running anything

---

## Dataset

The dataset is too large to upload to GitHub. Download it here:

🔗 **SPY Daily EOD Options Quotes 2020–2022 (Kaggle)**  
https://www.kaggle.com/datasets/kylegraupe/spy-daily-eod-options-quotes-2020-2022?resource=download

This dataset includes all SPY call and put quotes, Greeks, bid–ask data, timestamps, and size/volume fields.

---

## Project Steps

### 1. Data Ingestion and Cleaning
- Loaded ~3.6M rows using PySpark DataFrames  
- Cleaned, filtered, and converted types  
- Removed missing/invalid rows  
- Created daily aggregated dataset (~700 rows)

### 2. Exploratory Visualizations
- Daily IV Proxy trends  
- Put/Call ratio trends  
- Large Trade Frequency  
- Boxplots across market phases  
- PCA scatter plot  
- Elbow and silhouette tests for clustering

### 3. Modeling
- PCA for dimensionality reduction  
- KMeans clustering  
- Random Forest classifier predicting high-volatility days  
- ROC curve and AUC score

### 4. Cluster Computing
- Ran the full workflow on the Roar HPC cluster  
- Used up to 4 CPU nodes  
- Faster processing for aggregation, PCA, clustering, and Random Forest training

---

## How to Run This Project

### Notebook
1. Open `ClusterFinal_Mode4.ipynb`  
2. Connect to a PySpark environment or cluster  
3. Update dataset file paths  
4. Run all cells  



