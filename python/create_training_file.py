from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, concat

# Create a SparkSession
spark = SparkSession.builder.appName("CSVToParquet").getOrCreate()

# Define the path to your CSV file
csv_file_path = "/content/cleaned_data_gpt.csv"

# Read the CSV file
data = spark.read.csv(csv_file_path, header=True)  # Assuming a header row

# Create the desired format with concat
data = data.withColumn(
    "text",
    concat(
        lit("<s>[INST]"),
        data["query"],
        lit("[/INST]"),
        data["query_content"],
        lit("</s>"),
    ),
)

# Select only the formatted column
data = data.select("text")

# Write the data to a parquet file
data.write.format("parquet").save("/content/drive/MyDrive/Advision_file/output.parquet")

# Stop the SparkSession
spark.stop()

# Create a SparkSession
spark = SparkSession.builder.appName("ReadParquet").getOrCreate()

# Define the path to your parquet file
parquet_file_path = "/content/drive/MyDrive/Advision_file/output.parquet/part-00000-48749a10-8fb6-4044-9919-2452cf0f3bc6-c000.snappy.parquet"
# Read the parquet file
data = spark.read.format("parquet").load(parquet_file_path)

# Show the first few rows (optional)
data.show(truncate=False)

# Print the schema (column structure)
data.printSchema()

# Stop the SparkSession
spark.stop()