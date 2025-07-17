import pyspark
import findspark
import numpy as np

print("Dependencies installed successfully!")



# cleaning the data 
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, date_format
from pyspark.sql.functions import unix_timestamp, col, log, when, minute, month, lit, hour, dayofweek, sqrt, pow
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from distutils.version import LooseVersion

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("NYC Taxi Trip Prediction") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .getOrCreate()

# Load the dataset (replace 'path/to/dataset.csv' with your file path)
data = spark.read.csv("file_path", header=True, inferSchema=True)

# Inspect the data
data.show(5)


# Step 3: Data Preprocessing

# Remove outliers in trip_time_in_secs and trip_distance
data = data.filter((col("trip_time_in_secs") > 30) & (col("trip_time_in_secs") < 3000))  # Between 30 seconds and 2500 seconds
data = data.filter((col("trip_distance") > 0) & (col("trip_distance") < 50))  # Trip distance between 0 and 50

# Feature Engineering: Extract hour, minute, month, day of the week, and calculate distance
data = data.withColumn("pickup_hour", hour(col("pickup_datetime"))) \
           .withColumn("pickup_minute", minute(col("pickup_datetime"))) \
           .withColumn("pickup_month", month(col("pickup_datetime"))) \
           .withColumn("pickup_dayofweek", dayofweek(col("pickup_datetime"))) \
           .withColumn("log_trip_distance", log(col("trip_distance") + 1)) \
           .withColumn("Day_Monday", when(col("pickup_dayofweek") == 2, 1).otherwise(0)) \
           .withColumn("Day_Tuesday", when(col("pickup_dayofweek") == 3, 1).otherwise(0)) \
           .withColumn("Day_Wednesday", when(col("pickup_dayofweek") == 4, 1).otherwise(0)) \
           .withColumn("Day_Thursday", when(col("pickup_dayofweek") == 5, 1).otherwise(0)) \
           .withColumn("Day_Friday", when(col("pickup_dayofweek") == 6, 1).otherwise(0)) \
           .withColumn("Day_Saturday", when(col("pickup_dayofweek") == 7, 1).otherwise(0)) \
           .withColumn("Day_Sunday", when(col("pickup_dayofweek") == 1, 1).otherwise(0))

# Step 4: Assemble Features
feature_columns = ["passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude", 
                   "dropoff_latitude", "log_trip_distance", "pickup_month", "pickup_hour", "pickup_minute", 
                   "Day_Friday", "Day_Monday", "Day_Saturday", "Day_Sunday", 
                   "Day_Thursday", "Day_Tuesday", "Day_Wednesday"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data)

# Step 5: Standardize Features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(data)
data = scaler_model.transform(data)

# Step 6: Prepare Data for Model Training
# Keep only the scaled features and label
model_data = data.select("scaled_features", col("trip_time_in_secs").alias("label"))

# Split data into training and test sets
train_data, test_data = model_data.randomSplit([0.8, 0.2], seed=42)

# Step 7: Train Linear Regression Model
lr = LinearRegression(featuresCol="scaled_features", labelCol="label")
lr_model = lr.fit(train_data)

# Step 8: Evaluate the Model
# Make predictions on the test data
predictions = lr_model.transform(test_data)

# Evaluate with RMSE and R2
evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
evaluator_mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")
print(f"MAE: {mae}")


# Step 10: Save the Model
# lr_model.save("path_to_save_model")

# Stop SparkSession
spark.stop()


