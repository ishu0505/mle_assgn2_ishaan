# import os
# import argparse

# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col

# from utils.validators import validate_date, build_partition_name, pyspark_info
# from configs.data import source_data_files, bronze_data_dirs


# def add_data_bronze(date: str, type: str, spark: SparkSession):
#     """
#     Function to add data to a bronze table according to medallion architecture.
#     Args:
#         date (str): The date for which to add data, in the format "YYYY-MM-DD".
#         type (str): The data type to be processed.
#         spark (SparkSession): The Spark session object.
#     Returns nothing.
#     """

#     # prepare arguments
#     snapshot_date = validate_date(date, output_DateType=True)
#     input_file = source_data_files[type]
#     output_directory = bronze_data_dirs[type]
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)
    
#     # load data
#     print('Loading data for date:', date)
#     df = spark.read.csv(input_file, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
#     print(f"Loaded data from {input_file}. Row count: {df.count()}")

#     # Show DataFrame information
#     pyspark_info(df)

#     # save bronze table to datamart
#     partition_name = build_partition_name('bronze', type, date, 'csv')
#     filepath = os.path.join(output_directory, partition_name)
#     df.toPandas().to_csv(filepath, index=False)

#     print(f"Bronze table created for {type} on date {date} and saved to {filepath}")
#     return


# if __name__ == "__main__":

#     # get input arguments
#     parser = argparse.ArgumentParser(description='Add data to bronze table.')
#     parser.add_argument('--date', type=str, required=True, help='The date for which to add data, in the format YYYY-MM-DD')
#     parser.add_argument('--type', type=str, required=True, choices=['clickstream_data', 'customer_attributes', 'customer_financials', 'loan_data'],
#                         help='Type of data to process: clickstream_data, customer_attributes, customer_financials, or loan_data.')
#     args = parser.parse_args()

#     # validate input arguments
#     if not args.date or not args.type:
#         raise ValueError("All arguments --date and --type are required.")

#     # Initialize Spark session
#     spark = SparkSession.builder \
#         .appName("Bronze Processing") \
#         .master("local[*]") \
#         .getOrCreate()
#     spark.sparkContext.setLogLevel("ERROR")

#     # call function to add data to bronze table
#     add_data_bronze(date=args.date, type=args.type, spark=spark)

#     # Stop the Spark session
#     spark.stop()
#     print(f"\n\n---Bronze Store completed successfully---\n\n")













import os
import argparse
import glob
import shutil  # <-- IMPORTED: shutil for robust directory removal

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType

# Assuming these are in your python path, as configured in the DAG
from utils.validators import build_partition_name, validate_date
from configs.data import source_data_files, bronze_data_dirs, data_types


def add_data_bronze(date: str, type: str, spark: SparkSession):
    """
    Reads raw source data, filters for a specific snapshot date, and saves it
    to the bronze layer as a single CSV file. If no data exists for that date,
    it creates an empty CSV with the correct headers.

    Args:
        date (str): The snapshot date to process (e.g., "2023-01-01").
        type (str): The type of data to process (e.g., 'loan_data').
        spark (SparkSession): The active SparkSession.
    """
    # 1. Load the full raw dataset
    source_file_path = source_data_files.get(type)
    if not source_file_path:
        raise ValueError(f"Source file path not found for type: {type}")

    print(f"Loading raw data from: {source_file_path}")
    raw_df = spark.read.csv(source_file_path, header=True, inferSchema=True)
    
    # 2. Filter for the specific snapshot date
    df = raw_df.filter(F.col('snapshot_date') == date)

    # 3. Handle the case where the month has no data
    if df.rdd.isEmpty():
        print(f"Warning: No data found for type '{type}' on date {date}. Creating an empty bronze file.")
        
        # Get the schema from the raw, unfiltered dataframe to ensure headers are correct
        expected_schema = raw_df.schema
        
        # Create an empty DataFrame with the correct schema
        df = spark.createDataFrame([], schema=expected_schema)

    # 4. Prepare to save the output as a single named CSV file
    output_dir = bronze_data_dirs.get(type)
    if not output_dir:
        raise ValueError(f"Bronze directory not found for type: {type}")

    os.makedirs(output_dir, exist_ok=True)
    
    partition_name = build_partition_name('bronze', type, date, 'csv')
    final_output_path = os.path.join(output_dir, partition_name)
    
    # Spark writes to a directory by default, so we write to a temporary location
    # and then rename the single part-file.
    temp_output_dir = os.path.join(output_dir, "temp_" + partition_name)

    print(f"Writing data to temporary location: {temp_output_dir}")
    df.coalesce(1).write.mode("overwrite").option("header", "true").csv(temp_output_dir)
    
    # Find the temporary part-file created by Spark
    try:
        # Note: Spark might name the directory with .csv at the end. glob handles this.
        temp_file = glob.glob(f"{temp_output_dir}/*.csv")[0]
        # Rename the part-file to the desired final output file name
        shutil.move(temp_file, final_output_path)
        print(f"Successfully created bronze file: {final_output_path}")
    except IndexError:
        print(f"Warning: Could not find Spark output file in {temp_output_dir}. It's likely the DataFrame was empty.")
        # If the dataframe was empty, Spark might not create a part file.
        # In this case, we create an empty file with just the headers.
        header = ",".join(df.columns)
        with open(final_output_path, 'w') as f:
            f.write(header + '\n')
        print(f"Created an empty file with headers at: {final_output_path}")
    finally:
        # FIX: Use shutil.rmtree to safely remove the temporary directory and its contents.
        # This is more robust than os.rmdir and handles various edge cases.
        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)
            print(f"Cleaned up temporary directory: {temp_output_dir}")


# Main function to run the bronze processing scripts
if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process raw data and load into bronze layer.')
    parser.add_argument('--type', type=str, required=True, help='Type of data to process.')
    parser.add_argument('--date', type=str, required=True, help='Date for which data is being processed (format: YYYY-MM-DD).')
    args = parser.parse_args()

    # Validate the date argument
    date = validate_date(args.date)

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName(f"Bronze Processing - {args.type}") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Run the processing function
    add_data_bronze(date=date, type=args.type, spark=spark)

    # Stop the Spark session
    spark.stop()
    print(f"\n--- Bronze processing for '{args.type}' completed successfully for {date} ---\n")
