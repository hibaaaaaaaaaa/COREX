#!/usr/bin/env python
# coding: utf-8

# # SPY Options Sentiment & Volatility Project
# # Team: (>^_^)> CoreX <(^_^<)

# #### Importing the needed libraries:

# In[26]:


import pyspark
#import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, LongType, IntegerType, FloatType
from pyspark.sql.functions import col, column, year, when, expr, split, array_contains
from pyspark.sql.window import Window

from pyspark.sql import Row
from pyspark.mllib.recommendation import ALS
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType, StringType, DateType, LongType
from pyspark.sql import functions as F

#for the regression model:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.sql.functions import mean
import matplotlib.pyplot as plt
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline


# ## 1. Pipeline and Data Processing Part:

# #### We created a SparkSession to read, clean, and analyze data.

# In[2]:


#create a spark session rather than spark context in order to use dataframe
ss=SparkSession.builder.appName("ClusterFinal").getOrCreate()


# #### Set the directory directory to ~/scratch where Spark stores temporary checkpoint files and then delete them to prevent our storage from getting full.

# In[3]:


ss.sparkContext.setCheckpointDir("~/scratch")


# #### We defined a schema for the options dataset to specify each column’s name and data type, ensuring Spark reads the file correctly and treats numeric and date values properly.

# In[4]:


options_schema = StructType([
    StructField("QUOTE_UNIXTIME", LongType(), True),
    StructField("QUOTE_READTIME", StringType(), True),
    StructField("QUOTE_DATE", DateType(), True),
    StructField("QUOTE_TIME_HOURS", FloatType(), True),
    StructField("UNDERLYING_LAST", FloatType(), True),
    StructField("EXPIRE_DATE", DateType(), True),
    StructField("EXPIRE_UNIX", StringType(), True),
    StructField("DTE", FloatType(), True),

    # Call option features
    StructField("C_DELTA", FloatType(), True),
    StructField("C_GAMMA", FloatType(), True),
    StructField("C_VEGA", FloatType(), True),
    StructField("C_THETA", FloatType(), True),
    StructField("C_RHO", FloatType(), True),
    StructField("C_IV", FloatType(), True),
    StructField("C_VOLUME", FloatType(), True),
    StructField("C_LAST", FloatType(), True),
    StructField("C_SIZE", StringType(), True),
    StructField("C_BID", FloatType(), True),
    StructField("C_ASK", FloatType(), True),
    
    StructField("STRIKE", FloatType(), True),
    # Put option features
    StructField("P_BID", FloatType(), True),
    StructField("P_ASK", FloatType(), True),
    StructField("P_SIZE", StringType(), True),
    StructField("P_LAST", FloatType(), True),
    StructField("P_DELTA", FloatType(), True),
    StructField("P_GAMMA", FloatType(), True),
    StructField("P_VEGA", FloatType(), True),
    StructField("P_THETA", FloatType(), True),
    StructField("P_RHO", FloatType(), True),
    StructField("P_IV", FloatType(), True),
    StructField("P_VOLUME", FloatType(), True),

    
    StructField("STRIKE_DISTANCE", FloatType(), True),
    StructField("STRIKE_DISTANCE_PCT", FloatType(), True)
])


# #### We loaded the SPY options dataset using the predefined schema so Spark reads each column with the correct data type and structure.

# In[5]:


df = ss.read.csv("spy_2020_2022.csv", schema=options_schema, header=True, inferSchema=False) # Make sure to use the small dataset

### checking the total number of rows ###
#df.count()


# In[6]:


### to display the dataset’s structure and confirm columns ###
#options_DF.printSchema()


# #### Cleaning the data by converting dates, removing missing values, and filtering out negative prices

# In[7]:


cleaned_df = df.withColumn("Quote_DATE", F.to_date("Quote_DATE")) #converting Quote_DATE column from string to date formate
cleaned_df = cleaned_df.dropna(subset=["QUOTE_DATE", "C_BID", "C_ASK", "P_BID", "P_ASK", "C_VOLUME", "P_VOLUME"]) #removed any NA values
cleaned_df = cleaned_df.filter((F.col("C_BID") >= 0) & (F.col("C_ASK") >= 0) & (F.col("P_BID") >= 0) & (F.col("P_ASK") >= 0)) #Filtered out invalid option quotes where prices are negative

cleaned_df = cleaned_df.withColumn("Year", F.year("QUOTE_DATE"))
cleaned_df = cleaned_df.withColumn("Month", F.month("QUOTE_DATE"))

### checking the total number of rows ###
#cleaned_df.count()


# In[8]:


### create a smaller sample that have 3% of the data (~100,000 rows) for testing
#sample_df = cleaned_df.sample(withReplacement=False, fraction=0.03, seed=42)


# #### Saving the sampled dataset as options_sample.csv

# In[9]:


# saving the small test dataset
# coalesce(1) is used to save one file only
#sample_df.coalesce(1).write.mode("overwrite").csv("spy_small_sample", header=True)


# #### Reading the sample file we saved earlier
# ##### we renamed the output from "part-00000" into "option_sample.csv" manually

# In[11]:


#options_DF = ss.read.csv("spy_small_sample/option_sample.csv", schema=options_schema, header=True, inferSchema=False)
options_DF = cleaned_df


# In[12]:


### Checking the cleaned dataframe ###
#options_DF.printSchema()
#options_DF.show(5)
print("Number of rows:", options_DF.count())


# #### Calculating the Put-Call Ratio (PCR) by dividing total put volume by total call volume — a key indicator of overall market sentiment for each day

# In[13]:


# There exist days that are equal to ZERO, code adjusted to safely compute Put-Call Ratio (PCR) and avoid divide-by-zero errors
pcr_df = options_DF.groupBy("QUOTE_DATE").agg(
    F.when(F.sum("C_VOLUME") != 0,
           F.sum("P_VOLUME") / F.sum("C_VOLUME"))
     .otherwise(None)
     .alias("Put_Call_Ratio")
)

#pcr_df.show()


# #### Estimating the implied volatility proxy (IV_Proxy) by measuring the relative bid–ask spread of call options — a wider spread suggests higher uncertainty.
# 
# #### Then we averaged this proxy by date to track daily market volatility sentiment.

# In[14]:


# Calculate the average of bid and ask to represent the fair market value
iv_df = options_DF.withColumn("MidPrice", (F.col("C_ASK") + F.col("C_BID")) / 2)

# Estimate the implied volatility proxy (a measure of uncertainty)
iv_df = iv_df.withColumn("IV_proxy", F.when(F.col("MidPrice") != 0, (F.col("C_ASK") - F.col("C_BID")) / F.col("MidPrice")).otherwise(None))

# Compute the daily average IV proxy to summarize market volatility sentiment by day
iv_df = iv_df.groupBy("Quote_DATE").agg(F.avg("IV_proxy").alias("IV_Proxy"))

# Show a few rows to verify the results
#iv_df.show()


# #### Splitting values like "10x5" and converting them into numeric totals

# In[15]:


# Split the option size columns into separate parts (e.g., "10x5" → ["10", "5"])
options_DF = options_DF.withColumn("C_SIZE_SPL", F.split(F.col("C_SIZE"), "x"))        .withColumn("P_SIZE_SPL", F.split(F.col("P_SIZE"), "x"))

# Convert call size strings into numeric values by multiplying split parts if needed
options_DF = options_DF.withColumn(
    "C_SIZE_NUM",
    F.when(F.size("C_SIZE_SPL") == 2,
           F.col("C_SIZE_SPL").getItem(0).cast("int") * F.col("C_SIZE_SPL").getItem(1).cast("int")
          ).otherwise(F.col("C_SIZE_SPL").getItem(0).cast("int"))
)

# Convert put size strings into numeric values using the same logic
options_DF = options_DF.withColumn(
    "P_SIZE_NUM",
    F.when(F.size("P_SIZE_SPL") == 2,
           F.col("P_SIZE_SPL").getItem(0).cast("int") * F.col("P_SIZE_SPL").getItem(1).cast("int")
          ).otherwise(F.col("P_SIZE_SPL").getItem(0).cast("int"))
)


# #### We defined large trades using the 95th percentile of trade size and calculated the daily ratio of large trades to total trades as a measure of big-investor activity.

# In[16]:


threshold = options_DF.approxQuantile("C_SIZE_NUM", [0.95], 0.01)[0]

options_DF = options_DF.withColumn(
    "IsLargeTrade",
    F.when((F.col("C_SIZE_NUM") > threshold) | (F.col("P_SIZE_NUM") > threshold), 1).otherwise(0))

# Same issue with days equal to ZERO. code adjusted to safely compute Large Trade Frequency Ratio (LTF_Ratio)
ltf_df = options_DF.groupBy("QUOTE_DATE").agg(
    F.when(F.count("*") != 0,
           F.sum("IsLargeTrade") / F.count("*"))
     .otherwise(None)
     .alias("LTF_Ratio"))

#ltf_df.show()


# #### We combined all daily sentiment metrics into one DataFrame by joining on the date column, creating a single dataset ready for analysis or visualization.

# In[17]:


result_df = pcr_df.join(iv_df, "QUOTE_DATE", "inner").join(ltf_df, "QUOTE_DATE", "inner")
result_df = result_df.orderBy("QUOTE_DATE")

#result_df.show()


# #### We added a Market_Phase column that labels each date as “Pandemic,” “Optimism,” or “Correction” based on the year to categorize market sentiment periods.

# In[18]:


result_df = result_df.withColumn(
    "Market_Phase",
    F.when((F.col("QUOTE_DATE") >= F.lit("2020-01-01")) & (F.col("QUOTE_DATE") <= F.lit("2020-12-31")), "Pandemic")
     .when((F.col("QUOTE_DATE") >= F.lit("2021-01-01")) & (F.col("QUOTE_DATE") <= F.lit("2021-12-31")), "Optimism")
     .when((F.col("QUOTE_DATE") >= F.lit("2022-01-01")) & (F.col("QUOTE_DATE") <= F.lit("2022-12-31")), "Correction")
     .otherwise("Other")
)

result_df = result_df.orderBy("QUOTE_DATE")
#result_df.show()


# In[ ]:


# result_df.select("QUOTE_DATE", "Put_Call_Ratio", "IV_Proxy", "LTF_Ratio", "Market_Phase").write.mode("overwrite").option("header", True).csv("/storage/home/xqm5143/work/FinalProject/daily_sentiment")


# #### We merge our daily sentiment metrics (PCR, IV_Proxy, LTF_Ratio, Market_Phase) back onto the full option-level dataset so every option record is enriched with the corresponding daily market signals.

# In[19]:


full_df = options_DF.join(result_df, "QUOTE_DATE", "left")
#full_df.show(5)


# #### Then we save the final DataFrame as a CSV file for future use.

# In[20]:


# full_df_pd = full_df.toPandas()
# # full_df_pd.to_csv("SPY_sample1_cleaData2.csv", index=False)
# full_df_pd.to_csv("SPY_cleaData2.csv", index=False)


# ## 2. Exploring and Analyzing Our Data + Modeling

# #### Loading our sampled and cleaned data

# In[21]:


# #full_df_pd = pd.read_csv("SPY_sample1_cleaData2.csv", parse_dates=["QUOTE_DATE"])
# full_df_pd = pd.read_csv("SPY_cleaData2.csv", parse_dates=["QUOTE_DATE"])


# print(full_df_pd.shape)          # number of rows and columns
# print(full_df_pd.dtypes)         # data types for each column
# #full_df_pd.head()                # show first few rows


# #### Viewing summary statistics

# In[22]:


# full_df_pd[["Put_Call_Ratio", "IV_Proxy", "LTF_Ratio"]].describe()


# #### Summarizing by market phase shows how each sentiment metric behaves in different market periods (Pandemic, Optimism, Correction).

# In[23]:


# phase_summary = full_df_pd.groupby("Market_Phase")[["Put_Call_Ratio", "IV_Proxy", "LTF_Ratio"]].agg(["mean", "std", "min", "max"])
# phase_summary


# #### We look at the correlations to check whether high put-call ratios are associated with higher implied volatility or large trade activity.

# In[24]:


# corr = full_df_pd[["Put_Call_Ratio", "IV_Proxy", "LTF_Ratio"]].corr()
# corr


# #### Creating a time series plots by date and phase
# 
# ##### These plots will show us how volatility and sentiment shifted during different phases.

# In[28]:


# full_df_pd = full_df_pd.sort_values("QUOTE_DATE")

# for phase in full_df_pd["Market_Phase"].unique():
#     subset = full_df_pd[full_df_pd["Market_Phase"] == phase]
#     plt.plot(
#         subset["QUOTE_DATE"].to_numpy(),   # or .values
#         subset["IV_Proxy"].to_numpy(),     # or .values
#         label=phase
#     )

# plt.title("Daily IV_Proxy Over Time by Market Phase")
# plt.xlabel("Date")
# plt.ylabel("IV_Proxy")
# plt.xticks(rotation=45)
# plt.legend()
# plt.tight_layout()
# plt.show()


# In[30]:


# for phase in full_df_pd["Market_Phase"].unique():
#     subset = full_df_pd[full_df_pd["Market_Phase"] == phase]
#     plt.plot(
#         subset["QUOTE_DATE"].to_numpy(),   # or .values
#         subset["Put_Call_Ratio"].to_numpy(),     # or .values
#         label=phase
#     )
    
# # Put-Call Ratio over time
# plt.title("Daily Put-Call Ratio Over Time by Market Phase")
# plt.xlabel("Date")
# plt.ylabel("Put_Call_Ratio")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


# In[31]:


# for phase in full_df_pd["Market_Phase"].unique():
#     subset = full_df_pd[full_df_pd["Market_Phase"] == phase]
#     plt.plot(
#         subset["QUOTE_DATE"].to_numpy(),   # or .values
#         subset["LTF_Ratio"].to_numpy(),     # or .values
#         label=phase
#     )

# # Large Trade Frequency over time
# plt.title("Daily Large Trade Frequency (LTF_Ratio) Over Time by Market Phase")
# plt.xlabel("Date")
# plt.ylabel("LTF_Ratio")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#.coalesce(1)


# ##### Extra plots (Boxplot), if needed, to see how each metric’s range and median differ between market regimes.

# In[32]:


# # IV_Proxy by phase
# phases = full_df_pd["Market_Phase"].unique()
# data = [full_df_pd[full_df_pd["Market_Phase"] == phase]["IV_Proxy"] for phase in phases]
# plt.boxplot(data, labels=phases)
# plt.title("IV_Proxy by Market Phase")
# plt.xlabel("Market Phase")
# plt.ylabel("IV_Proxy")
# plt.xticks(rotation=45)
# plt.show()

# # Put_Call_Ratio by phase
# data = [full_df_pd[full_df_pd["Market_Phase"] == phase]["Put_Call_Ratio"] for phase in phases]
# plt.boxplot(data, labels=phases)
# plt.title("Put-Call Ratio by Market Phase")
# plt.xlabel("Market Phase")
# plt.ylabel("Put_Call_Ratio")
# plt.xticks(rotation=45)
# plt.show()

# # LTF_Ratio by phase
# data = [full_df_pd[full_df_pd["Market_Phase"] == phase]["LTF_Ratio"] for phase in phases]
# plt.boxplot(data, labels=phases)
# plt.title("LTF_Ratio by Market Phase")
# plt.xlabel("Market Phase")
# plt.ylabel("LTF_Ratio")
# plt.xticks(rotation=45)
# plt.show()


# #### A simple regression model to see if we can predict volatility (IV_Proxy)

# In[33]:


# # Drop rows with missing values in model columns
# model_df = full_df.dropna(subset=["IV_Proxy", "Put_Call_Ratio", "LTF_Ratio"])

# # Define predictors and target
# X = model_df[["Put_Call_Ratio", "LTF_Ratio"]]
# y = model_df["IV_Proxy"]

# # Split data for evaluation
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train linear regression model
# lr = LinearRegression()
# lr.fit(X_train, y_train)

# # Predict and evaluate
# y_pred = lr.predict(X_test)
# r2 = r2_score(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# print("R²:", round(r2, 3))
# print("RMSE:", round(rmse, 3))
# print("Coefficients:", lr.coef_)
# print("Intercept:", lr.intercept_)


# ## 3. Preforming PCA (Principal Component Analysis) and finding KMeans

# we selected three sentiment indicators (Put_Call_Ratio, IV_Proxy, LTF_Ratio) as input variables for modeling

# In[34]:


feature_columns = ["Put_Call_Ratio", "IV_Proxy", "LTF_Ratio"]


# In[35]:


modelData = full_df
modelData = modelData.dropna(subset=["Put_Call_Ratio", "IV_Proxy", "LTF_Ratio"])
assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features")

assembled_data = assembler.transform(modelData)


# We standardize the data with StandardScaler to remove unit bias before PCA

# In[36]:


scaler = StandardScaler(inputCol="features",
                        outputCol="scaled_features",
                        withStd=True,
                        withMean=True)
scaler_model = scaler.fit(assembled_data)
scaled_data = scaler_model.transform(assembled_data)


# In[37]:


# PCA
pca = PCA(k=3, inputCol="scaled_features", outputCol="pcaFeatures")
model = pca.fit(scaled_data)
result = model.transform(scaled_data)

explained_variance = model.explainedVariance
print("Explained Variance: ", sum(explained_variance))


# In[38]:


#KMeans

def fit_kmeans(df_input,column_name='pcaFeatures',num_cluster_centers=3):
    kmeans = KMeans(featuresCol=column_name).setK(num_cluster_centers).setSeed(1)
    model = kmeans.fit(df_input)
    clustered_data = model.transform(df_input)
    evaluator = ClusteringEvaluator()
    silhouette_score = evaluator.evaluate(clustered_data)
    wcss = model.summary.trainingCost
    cluster_sizes = clustered_data.groupBy("prediction").count()
    # cluster_sizes.show()
    return clustered_data, silhouette_score, wcss


# In[39]:


k_values = range(2, 11) 
silhouette_scores = []
wcss_scores = []

for k in k_values:
    clustered_data, silh, wcss = fit_kmeans(result,column_name='pcaFeatures',num_cluster_centers=k)
    silhouette_scores.append(silh)
    wcss_scores.append(wcss)
    print(f"K = {k}, silhouette = {silh:.2f}, WCSS = {wcss:.2f}")
    
    
# plt.plot(k_values, wcss_scores, marker='o')
# plt.title("Elbow Method for Optimal K")
# plt.xlabel("Number of Clusters (K)")
# plt.ylabel("WCSS Scores")
# plt.show()

# plt.plot(k_values, silhouette_scores, marker='o')
# plt.xlabel("Number of Clusters (K)")
# plt.ylabel("Silhouette Scores")
# plt.show()


# With unsupervised learning (PCA + KMeans), we let the data decide how many regimes exist and which days belong to each cluster.
# This can show us whether the market behavior actually matches our existing labeled phases or if it will reveal something new.
# 
# Using unsupervised learning with PCA and KMeans, we allowed the sentiment data to reveal its own structure without using any year-based labels. The elbow plot shows a large drop in WCSS from K=2 to K=5, after which improvements slow down, while the silhouette scores climb steadily and peak at K=5, indicating this is the most stable and well-separated clustering structure. With five clusters, the model forms one large baseline cluster and several smaller clusters representing different intensities of market stress or speculative activity. This suggests that investor sentiment does not fall neatly into only three regimes (Pandemic, Optimism, Correction), but instead exhibits about five distinct behavioral patterns, ranging from normal trading to high-stress, high-uncertainty conditions. Clusters above K=6 produce negative silhouette values, indicating over-segmentation. Overall, PCA + KMeans reveals a richer regime structure than simple year-based labels, identifying five meaningful sentiment-driven market states hidden in the SPY options data.

# In[ ]:


clustered_data.select("QUOTE_DATE", "Put_Call_Ratio", "IV_Proxy", "LTF_Ratio", "prediction").write.mode("overwrite").option("header", True).csv("/storage/home/xqm5143/work/FinalProject/kmeans_clusters")


# In[40]:


iv_threshold = modelData.approxQuantile("IV_Proxy", [0.75], 0.01)[0]
modelData = modelData.withColumn(
    "HighStress",
    F.when(col("IV_Proxy") >= iv_threshold, 1).otherwise(0)
)

modelData = modelData.withColumn("Year", F.year(col("QUOTE_DATE")))
train_df = modelData.filter((col("Year") == 2020) | (col("Year") == 2021))
test_df = modelData.filter(col("Year") == 2022)


# In[41]:


feature_columns = ["Put_Call_Ratio", "LTF_Ratio"]

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

train_assembled = assembler.transform(train_df)
test_assembled = assembler.transform(test_df)

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withMean=True,
    withStd=True
)

scaler_model = scaler.fit(train_assembled)

train_scaled = scaler_model.transform(train_assembled)
test_scaled = scaler_model.transform(test_assembled)


# In[42]:


log_model = LogisticRegression(
    featuresCol = "scaled_features",
    labelCol = "HighStress"
)
log_model = log_model.fit(train_scaled)


# In[43]:


predictions = log_model.transform(test_scaled)

predictions.select(
    "QUOTE_DATE",
    "Put_Call_Ratio",
    "LTF_Ratio",
    "HighStress",
    "probability",
    "prediction"
)#.show(10, truncate=False)


# In[ ]:


predictions.select("QUOTE_DATE", "HighStress", "prediction", "probability").write.mode("overwrite").option("header", True).csv("/storage/home/xqm5143/work/FinalProject/rf_predictions")


# #### Observation
# We created a model using the same variables to predect the market volatility, but the model performance was weak because the sentiment features (PCR and LTF) alone are insufficient to accurately classify high-volatility days.

# ## 4. Attempting to building a stronger predictive model

# In[44]:


iv_threshold = modelData.approxQuantile("IV_Proxy", [0.75], 0.01)[0]

modelData = modelData.withColumn(
    "HighStress",
    when(col("IV_Proxy") >= iv_threshold, 1).otherwise(0))

modelData = modelData.withColumn("Year", year(col("QUOTE_DATE")))
train_df = modelData.filter((col("Year") == 2020) | (col("Year") == 2021))
test_df  = modelData.filter(col("Year") == 2022)


# In[45]:


rf_features = [
    "Put_Call_Ratio",
    "LTF_Ratio",
    "DTE",
    "STRIKE_DISTANCE_PCT",
    "C_VEGA",
    "C_GAMMA",
    "C_DELTA",
    "UNDERLYING_LAST"]

assembler = VectorAssembler(
    inputCols=rf_features,
    outputCol="rf_features")

train_rf = assembler.transform(train_df).dropna(subset=rf_features)
test_rf  = assembler.transform(test_df).dropna(subset=rf_features)


# In[46]:


rf = RandomForestClassifier(
    featuresCol="rf_features",
    labelCol="HighStress",
    numTrees=300,
    maxDepth=10,
    seed=42
)

rf_model = rf.fit(train_rf)
rf_predictions = rf_model.transform(test_rf)


# In[ ]:


rf_predictions.select("QUOTE_DATE", "HighStress", "prediction", "probability").write.mode("overwrite").option("header", True).csv("/storage/home/xqm5143/work/FinalProject/rf_highstress_predictions")


# In[47]:


evaluator = BinaryClassificationEvaluator(
    labelCol="HighStress",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

auc = evaluator.evaluate(rf_predictions)


# In[48]:


rf_predictions.select(
    "QUOTE_DATE", "HighStress", "prediction", "probability"
)#.show(20)


# The model is still weak because the features could be weak:
# Put_Call_Ratio, LTF_Ratio don't seem to be a volatility driver IV_Proxy sometimes correlated but very noisy.

# ### We will construct another model with different features to predect the market volatility

# We will reload the raw data that contains Greeks & price columns

# In[49]:


#fullDF = ss.read.csv("SPY_cleaData2.csv", header=True, inferSchema=True)


# In[50]:


result_daily = full_df.select(
    "QUOTE_DATE", "Put_Call_Ratio", "IV_Proxy", "LTF_Ratio", "Market_Phase"
).distinct()


# In[51]:


price_df = full_df.select("QUOTE_DATE", "UNDERLYING_LAST").distinct()

price_df = price_df.repartition(5, "QUOTE_DATE").sort("QUOTE_DATE")

window = Window.orderBy("QUOTE_DATE")

price_df = (
    price_df
    .withColumn("prev_price", F.lag("UNDERLYING_LAST").over(window))
    .withColumn(
        "daily_return",
        (F.col("UNDERLYING_LAST") - F.col("prev_price")) / F.col("prev_price")
    )
    .withColumn("abs_return", F.abs("daily_return"))
)


# In[52]:


greeks_df = full_df.groupBy("QUOTE_DATE").agg(
    F.avg("C_DELTA").alias("avg_delta"),
    F.avg("C_GAMMA").alias("avg_gamma"),
    F.avg("C_VEGA").alias("avg_vega"),
    F.avg("C_THETA").alias("avg_theta"),
    F.avg("STRIKE_DISTANCE_PCT").alias("avg_strike_pct"),
    F.avg("DTE").alias("avg_DTE"),
    F.avg(F.col("C_ASK") - F.col("C_BID")).alias("avg_C_bidask"),
    F.avg(F.col("P_ASK") - F.col("P_BID")).alias("avg_P_bidask"),
    F.avg("C_VOLUME").alias("avg_volume")
)


# In[53]:


finalDF = (
    result_daily
    .join(price_df, on="QUOTE_DATE", how="inner")
    .join(greeks_df, on="QUOTE_DATE", how="inner")
    .dropna()
)
finalDF = finalDF.repartition(5, "QUOTE_DATE")


# In[54]:


iv_threshold = finalDF.approxQuantile("IV_Proxy", [0.75], 0.01)[0]


# In[55]:


finalDF = finalDF.withColumn(
    "HighVol",
    F.when(F.col("IV_Proxy") >= iv_threshold, 1).otherwise(0)
)

# finalDF.show(5)
# print("Number of rows:", finalDF.count())


# Our final daily dataset contains 756 trading days from 2020–2022, and each row now represents a complete snapshot of market sentiment, price movement, and option-Greek behavior for that day. 
# The features we aggregated (Put-Call Ratio, IV_Proxy, LTF_Ratio, daily returns, Greeks, bid–ask spreads, and volume) behave exactly as expected across the three market regimes:
# 1. 2020 shows higher volatility and heavier large-trade activity
# 2. 2021 becomes calmer
# 3. 2022 rises again due to correction pressures.
# 
# We created the 'HighVol label' by marking the top 25% of IV_Proxy values, most days naturally ended up labeled 0 because extreme volatility only occurs in short bursts (pandemic crash and 2022 rate-hike shocks). 
# This confirms why the two earlier models struggled to capture the volatility dynamics.

# ### Now we will build a random forest model

# In[56]:


feature_cols = [
    "Put_Call_Ratio",
    "LTF_Ratio",
    "UNDERLYING_LAST",
    "daily_return",
    "abs_return",
    "avg_DTE",
    "avg_strike_pct",
    "avg_delta",
    "avg_gamma",
    "avg_theta",
    "avg_vega",
    "avg_C_bidask",
    "avg_P_bidask",
    "avg_volume"
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)


# In[57]:


rf = RandomForestClassifier(
    labelCol="HighVol",
    featuresCol="features",
    numTrees=200,
    maxDepth=10,
    seed=42
)

pipeline = Pipeline(stages=[assembler, rf])


# In[58]:


train = finalDF.filter(F.year("QUOTE_DATE").isin(2020, 2021))
test  = finalDF.filter(F.year("QUOTE_DATE") == 2022)


# In[59]:


model = pipeline.fit(train)


# In[60]:


preds = model.transform(test)

preds.select(
    "QUOTE_DATE",
    "HighVol",
    "prediction",
    "probability"
)#.show(20)


# In[ ]:


preds.select("QUOTE_DATE", "HighVol", "prediction", "probability").write.mode("overwrite").option("header", True).csv("/storage/home/xqm5143/work/FinalProject/rf_highvol_predictions")


# In[61]:


evaluator = BinaryClassificationEvaluator(
    labelCol="HighVol",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

auc = evaluator.evaluate(preds)
print("AUC =", auc)


# The final Random Forest model delivered an AUC of 0.718, meaning it correctly distinguishes high-volatility market days roughly 72% of the time. 
# This model is preforming better compared to our earlier models because we shifted from weak sentiment-only predictors (PCR, LTF) to stronger variables directly tied to volatility dynamics, such as daily returns, option Greeks, strike distance, bid–ask spreads, and underlying price levels. 
# These features seem to capture real structural changes in the options market, which allows the model to learn the mechanics of volatility more accurately.
# As we see in the prediction table: the model successfully flags high-volatility days that align with 2022 drawdowns, while occasional false positives and false negatives do exist.

# In[65]:


ss.stop()


# In[ ]:




