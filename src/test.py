
# Marwane Zaoudi
import pyspark
import findspark
from pyspark.sql.functions import col, when, count
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id


from pyspark.sql.functions import hour, dayofweek

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

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
    "/Users/marwanezaoudi/Downloads/filetered_data/FilteredData1.csv",
    "/Users/marwanezaoudi/Downloads/filetered_data/FilteredData2.csv",
    "/Users/marwanezaoudi/Downloads/filetered_data/FilteredData3.csv",
    "/Users/marwanezaoudi/Downloads/filetered_data/FilteredData4.csv",
    "/Users/marwanezaoudi/Downloads/filetered_data/FilteredData5.csv",
    "/Users/marwanezaoudi/Downloads/filetered_data/FilteredData10.csv",
    "/Users/marwanezaoudi/Downloads/filetered_data/FilteredData11.csv",
    "/Users/marwanezaoudi/Downloads/filetered_data/FilteredData12.csv",
]


trip_data = spark.read.csv(file_paths, header=True, inferSchema=True)

# Trim and clean column names to remove leading/trailing whitespaces     , i did this since i figured that the files has a lot of spaces and couldn't read at first 
#trip_data = trip_data.toDF(*[col_name.strip() for col_name in trip_data.columns])

# Select the columns i want to drop using the drop in pyspark 
#droppedColumns = ["trip_id" , "store_and_fwd_flag" , "medallion" , "hack_license" , "rate_code", "id"]
#trip_data = trip_data.drop(*droppedColumns)

# creating a new column called id ( increasing numbers )
#trip_data = trip_data.withColumn("id", monotonically_increasing_id())

# make id the first column 
#trip_data = trip_data[['id'] + trip_data.columns[:-1]]

#trip_data = trip_data.filter(col("trip_distance") != 0) 
#trip_data = trip_data.filter(col("passenger_count") != 0)



# Save the cleaned data
#output_dir = "/Users/marwanezaoudi/Downloads/filetered_data/filtered_trip_data_2"
#trip_data.coalesce(1).write.option("header", "true").csv(output_dir)     # THIS MAKES PYSPARK SAVE ALL PART IN ONE SINGLE FILE .CSV 
#print("Data saved successfully!")   # for debugging purposes 



trip_data = trip_data.withColumn("hour_of_day", hour(trip_data["pickup_datetime"]))

# Extract the day of the week (1=Monday, 7=Sunday) from the pickup_datetime column
trip_data = trip_data.withColumn("day_of_week", dayofweek(trip_data["pickup_datetime"]))

trip_data = trip_data.withColumn("is_weekend", when((col("day_of_week") == 7) | (col("day_of_week") == 1), 1).otherwise(0))





features_Needed = ["trip_distance", "passenger_count", "pickup_longitude", "pickup_latitude",
                "dropoff_longitude", "dropoff_latitude", "hour_of_day", "day_of_week", "is_weekend"]


# in regression models , we need to specify the label or a target we wish to predict , in this case 
# we are trying to predict trip_times in secs 

label = "trip_time_in_secs"


# now we need to assmeble the features into a single vector 
# source used : https://www.machinelearningplus.com/pyspark/pyspark-gradient-boosting-model/ 

assembler = VectorAssembler(inputCols=features_Needed, outputCol="features")
trip_data = assembler.transform(trip_data)

#trip_data.printSchema()


# check if transformation is working 
# trip_data.select("features").show(5, truncate=False)

train_data, test_data = trip_data.randomSplit([0.8, 0.2], seed=42)


gbt = GBTRegressor(featuresCol="features", labelCol="trip_time_in_secs", maxIter=10 )

model = gbt.fit(train_data)

trip_data.printSchema()