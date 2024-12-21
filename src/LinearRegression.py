
# Marwane Zaoudi
import pyspark
import findspark
from pyspark.sql.functions import col, when, count
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id

import matplotlib.pyplot as plt
import seaborn as sns


from pyspark.sql.functions import hour, dayofweek

from pyspark.ml.feature import VectorAssembler


# Initialize Spark Session
spark = SparkSession.builder \
    .appName("NYC Taxi Trip") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.files.maxRecordLength", "5368709120") \
    .getOrCreate()


# Load the CSV File

# load all files 

file_paths = [
    "/Users/marwanezaoudi/Downloads/trip_data/trip_data_1.csv",
    "/Users/marwanezaoudi/Downloads/trip_data/trip_data_2.csv",
    "/Users/marwanezaoudi/Downloads/trip_data/trip_data_3.csv",
    "/Users/marwanezaoudi/Downloads/trip_data/trip_data_4.csv",
    "/Users/marwanezaoudi/Downloads/trip_data/trip_data_5.csv",
    "/Users/marwanezaoudi/Downloads/trip_data/trip_data_10.csv",
    "/Users/marwanezaoudi/Downloads/trip_data/trip_data_11.csv",
    "/Users/marwanezaoudi/Downloads/trip_data/trip_data_12.csv",
]


trip_data = spark.read.csv(file_paths, header=True, inferSchema=True)

# picking the features and extracting 
trip_data = trip_data.withColumn("hour_of_day", hour(trip_data["pickup_datetime"]))
# Extract the day of the week (1=Monday, 7=Sunday) from the pickup_datetime column
trip_data = trip_data.withColumn("day_of_week", dayofweek(trip_data["pickup_datetime"]))
trip_data = trip_data.withColumn("is_weekend", when((col("day_of_week") == 7) | (col("day_of_week") == 1), 1).otherwise(0))


trip_data.printSchema()

# need to plot data 


#sampled_data = trip_data.sample(fraction=0.01).toPandas()
# Sample a subset of the data (PySpark DataFrames can be large, so sample for plotting)
#plt.figure(figsize=(8, 6))
#sns.histplot(sampled_data["passenger_count"], bins=50, kde=True, color='blue')
#plt.title("Distribution of Passenger Count (After Outlier Removal)")
#plt.xlabel("Passenger Count")
#plt.ylabel("Frequency")
#plt.axvline(x=0.5, color='red', linestyle='--', label="Removed: Passenger Count = 0")  # Highlight outliers removed
#plt.legend()
#plt.show()


