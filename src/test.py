# test pyspark env 
# Marwane Zaoudi




import pyspark
import findspark
print("Dependencies installed successfully!")




# we are going to be testing some pyspark using the data we have , cleaning it and transforming it 

# cleaning the data 
from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder.appName("NYC Taxi Trip Prediction").getOrCreate()

# Load datasets
trip_fare1 = spark.read.csv("./Datasets/trip_fare_1.csv", header=True, inferSchema=True)


# Check data : 
trip_fare1.show(5)
trip_fare1.printSchema()