
# Marwane Zaoudi
import pyspark
import findspark
from pyspark.sql.functions import col, when, count
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("NYC Taxi Trip Processing") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.files.maxRecordLength", "5368709120") \
    .getOrCreate()


# Load the CSV File
trip_data = spark.read.csv("/Users/marwanezaoudi/Downloads/trip_data/trip_data_8.csv", header=True, inferSchema=True)

# Trim and clean column names to remove leading/trailing whitespaces     , i did this since i figured that the files has a lot of spaces and couldn't read at first 
trip_data = trip_data.toDF(*[col_name.strip() for col_name in trip_data.columns])

# Select the columns i want to drop using the drop in pyspark 
droppedColumns = ["medallion", "hack_license", "rate_code", "store_and_fwd_flag"]
cleaned_data = trip_data.drop(*droppedColumns)

# creating a new column called id ( increasing numbers )
increasingId = cleaned_data.withColumn("trip_id", monotonically_increasing_id() + 1 )

# Show the first few rows to confirm the changes
increasingId.show(5, truncate=False)


# Save the cleaned data
#output_dir = "/Users/marwanezaoudi/Downloads/trip_data/cleaned_trip_data_5.csv"
#increasingId.coalesce(1).write.option("header", "true").csv(output_dir)     # THIS MAKES PYSPARK SAVE ALL PART IN ONE SINGLE FILE .CSV 



#print("Data saved successfully!")   # for debugging purposes 
#print(increasingId[increasingId['trip_distance']==0].count())  # for debugging purposes 


