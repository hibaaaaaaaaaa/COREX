SPY Options Market Analysis (2020–2022)

This project looks at how SPY options activity and market sentiment changed from 2020 to 2022. We use PySpark on the Roar cluster to process over 3.6 million option quote records, build daily sentiment features, explore patterns across market phases, and train models to predict high-volatility days.

The repo includes the final notebook, Python script, and the HTML export of the analysis.

Research Question

How did SPY options activity and market sentiment evolve from 2020–2022?

Hypotheses

Put/Call Ratio, implied volatility, and large-trade frequency change noticeably across the pandemic, optimism, and correction phases.

Simple sentiment indicators (PCR, LTF) are weak predictors of volatility by themselves.

Richer option features (Greeks, strike distance, bid–ask spreads, returns) improve prediction of high-volatility days.

Files in This Repo
1. ClusterFinal_Modem4.ipynb

Main notebook with all the PySpark code

Data cleaning

Daily aggregation

PCA

KMeans clustering

Random Forest modeling

Visualizations

ROC curve evaluation

Best place to see the full workflow

2. ClusterFinal_Modem4.py

Script version of the main analysis

Can be run directly on the Roar cluster

Includes the same pipeline as the notebook

3. ClusterFinal_Modem4.html

HTML export of the notebook

Easy to read without running anything

Dataset

The full dataset is too large to upload to GitHub. You can download it here:
🔗 SPY Daily EOD Options Quotes 2020–2022 (Kaggle)
https://www.kaggle.com/datasets/kylegraupe/spy-daily-eod-options-quotes-2020-2022?resource=download

This dataset contains all SPY call and put quotes, Greeks, bid–ask data, timestamps, and size/volume fields.

Project Steps
1. Data Ingestion + Cleaning

Loaded ~3.6M rows using PySpark DataFrames

Cleaned, filtered, and converted types

Removed missing and invalid rows

Created daily aggregated dataset (~700 rows)

2. Exploratory Visualizations

Daily IV Proxy trends

Put/Call ratio trends

Large Trade Frequency (LTF)

Boxplots across market phases

PCA scatter plot

Clustering elbow and silhouette tests

3. Modeling

PCA for dimensionality reduction

KMeans clustering

Random Forest classifier to predict high-volatility days

ROC curve evaluation

4. Cluster Computing

Ran the full workflow on the Roar HPC cluster

Used up to 4 CPU nodes

Significant speed improvements for aggregation, PCA, KMeans, and Random Forest training

How to Run This Project
Option 1: Notebook

Open ClusterFinal_Modem4.ipynb

Connect to a PySpark environment or cluster

Update your file paths to the dataset

Run all cells

Option 2: Python Script
spark-submit ClusterFinal_Modem4.py


Make sure your Spark session has access to the dataset path.

Outputs

Running the notebook/script will generate:

Cleaned daily dataset

PCA plot

KMeans cluster labels

Full set of visualizations

Random Forest predictions

ROC curve and AUC score

These show how options sentiment changed over time and how well our models perform.
