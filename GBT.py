


## all the sources used for leanrning are cited in the report 
# Marwane Zaoudi
import pyspark
import findspark
from pyspark.sql.functions import col, when, count
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.feature import VectorAssembler, MinMaxScaler

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
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


#  
trip_data = trip_data.filter((col("trip_time_in_secs") > 60) & (col("trip_time_in_secs") <= 7200))  # Between  1 min and 2 hours

# 
trip_data = trip_data.filter((col("passenger_count") > 0) & (col("passenger_count") <= 6))

# 
# according to https://www.latlong.net/place/new-york-city-ny-usa-1848.html#:~:text=Satellite%20Map%20of%20New%20York%20City%2C%20NY%2C%20USA&text=The%20latitude%20of%20New%20York,°%2056'%206.8712''%20W.
# Approximate bounding box for NYC: (Lat: 40.4774 to 40.9176, Lon: -74.2591 to -73.7004)
trip_data = trip_data.filter((col("pickup_latitude") >= 40.4774) & (col("pickup_latitude") <= 40.9176) &
                             (col("pickup_longitude") >= -74.2591) & (col("pickup_longitude") <= -73.7004) &
                             (col("dropoff_latitude") >= 40.4774) & (col("dropoff_latitude") <= 40.9176) &
                             (col("dropoff_longitude") >= -74.2591) & (col("dropoff_longitude") <= -73.7004))



#CREATING NEW FEATURES ! 

trip_data = trip_data.withColumn("hour_of_day", hour(trip_data["pickup_datetime"]))
trip_data = trip_data.withColumn("day_of_week", dayofweek(trip_data["pickup_datetime"]))
trip_data = trip_data.withColumn("is_weekend", when((col("day_of_week") == 7) | (col("day_of_week") == 1), 1).otherwise(0))
trip_data = trip_data.withColumn("weekend_hour_interaction", col("hour_of_day") * col("is_weekend"))

# Define feature columns
feature_columns = ["trip_distance", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", 
                   "passenger_count", "weekend_hour_interaction"]


# Assemble features
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
trip_data = assembler.transform(trip_data)



# for debugging purposes 

#trip_data.printSchema()
#trip_data.select("features").show(5, truncate=False)

# Scale the features using mixmaxscaler


scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(trip_data)
trip_data = scaler_model.transform(trip_data)


# Ensure the label column is numeric
trip_data = trip_data.withColumn("trip_time_in_secs", trip_data["trip_time_in_secs"].cast("float"))


# now we split the data for training 
train_data, test_data = trip_data.randomSplit([0.8, 0.2], seed=42)

# start with the gbt 
gbt = GBTRegressor(featuresCol="scaled_features", labelCol="trip_time_in_secs")

# tuning the hyperparameters 
paramGrid = ParamGridBuilder().addGrid(gbt.maxDepth, [5, 10]).addGrid(gbt.maxIter, [10, 20]).build()

# start the eval 
evaluator_rmse = RegressionEvaluator(labelCol="trip_time_in_secs", predictionCol="prediction", metricName="rmse")

# Cross-validation
crossval = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid,
                         evaluator=evaluator_rmse, numFolds=3)

# Train with cross-validation
cv_model = crossval.fit(train_data)
best_model = cv_model.bestModel


# Make predictions on test data
predictions = best_model.transform(test_data)

# Define evaluators for metrics
evaluator_mae = RegressionEvaluator(labelCol="trip_time_in_secs", predictionCol="prediction", metricName="mae")
evaluator_r2 = RegressionEvaluator(labelCol="trip_time_in_secs", predictionCol="prediction", metricName="r2")

# Compute metrics
rmse = evaluator_rmse.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

# View metrics
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² (Coefficient of Determination): {r2}")

# Feature importances
# just for debugging and figuring out the best features 

print("Feature Importances:", best_model.featureImportances) 

best_model.save("/Users/marwanezaoudi/Downloads")