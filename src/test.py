 
# Marwane Zaoudi

import pyspark
import findspark
from pyspark.sql.functions import col, when, count
from pyspark.sql import SparkSession


# Initialize Spark Session
spark = SparkSession.builder \
    .appName("NYC Taxi Trip Processing") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Load the CSV File
trip_data = spark.read.csv("/Users/marwanezaoudi/Downloads/trip_data/trip_data_5.csv", header=True, inferSchema=True)

# Trim and clean column names to remove leading/trailing whitespaces     , i did this since i figured that the files has a lot of spaces and stuff like that 
trip_data = trip_data.toDF(*[col_name.strip() for col_name in trip_data.columns])

selected_columns = [
    "vendor_id",
    "pickup_datetime",
    "dropoff_datetime",
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "trip_distance",
    "store_and_fwd_flag"
]

# Select the relevant columns
cleaned_data = trip_data.select(*selected_columns)

# Show the first few rows to confirm the changes
cleaned_data.show(10, truncate=False)

# Save the cleaned data
output_dir = "/Users/marwanezaoudi/Downloads/trip_data/cleaned_trip_data_5.csv"
cleaned_data.coalesce(1).write.option("header", "true").csv(output_dir)     # THIS MAKES PYSPARK SAVE ALL PART IN ONE SINGLE FILE .CSV 

print("Data saved successfully!")

