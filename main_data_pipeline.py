"""
quick function to run the data processing scripts for the data pipeline to play with the data
"""
import os
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame
from scripts.data_processing.bronze_processing import add_data_bronze
from scripts.data_processing.silver_processing import (
    silver_processing_clickstream_data,
    silver_processing_customer_attributes,
    silver_processing_customer_financials,
    silver_processing_label_store
)
from scripts.data_processing.gold_processing_label import create_label
from scripts.data_processing.gold_processing_features import join_feature_tables, one_hot_encode, encode_loan_types_with_counts

from utils.data import load_data, transform_data_in_column
from utils.validators import build_partition_name
from configs.data import bronze_data_dirs, silver_data_dirs, gold_data_dirs, data_types

# Initialize SparkSession
print("Initializing Spark session & generating snapshot dates...")
spark = SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

data = ['clickstream_data', 'customer_attributes', 'customer_financials', 'loan_data']


#### MAIN LOOP OVER ALL DATES ####
for date_str in dates_str_lst:
    print(f"\nProcessing date: {date_str}")
    

    # Bronze processing for all four sources
    for data_type in data:
        print(f"\nProcessing {data_type} for date {date_str} into bronze store...")
        add_data_bronze(date=date_str, type=data_type, spark=spark)
        print(f"\n\n---Bronze {data_type} Store completed successfully for date {date_str}---\n\n")


    # Silver processing for all four sources
    for data_type in data:
        print(f"\nProcessing {data_type} for date {date_str} into silver store...")

        # Load the data from the bronze store
        partition = build_partition_name('bronze', data_type, date_str, 'csv') # type: ignore
        df = load_data(spark, bronze_data_dirs[data_type], partition)
        bronze_count = df.count()

        # Type casting
        for column, dtype in data_types[data_type].items():
            df = transform_data_in_column(df, column, dtype)

        # Process data based on type
        if data_type == 'clickstream_data':
            df = silver_processing_clickstream_data(df)
        elif data_type == 'customer_attributes':
            df = silver_processing_customer_attributes(df)
        elif data_type == 'customer_financials':
            df = silver_processing_customer_financials(df)
        elif data_type == 'loan_data':
            df = silver_processing_label_store(df)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        # check data after cleaning
        silver_count = df.count()
        if silver_count != bronze_count:
            print(f"Warning: Row count changed from {bronze_count} to {silver_count} after cleaning for partition {partition}.")
        # pyspark_df_info(df)

        # Ensure that silver directory exists
        os.makedirs(os.path.dirname(silver_data_dirs[data_type]), exist_ok=True)

        # Save the cleaned data to the silver directory as parquet
        partition_name = build_partition_name('silver', data_type, date_str, 'parquet') # type: ignore
        silver_filepath = os.path.join(silver_data_dirs[data_type], partition_name)

        df.write.mode("overwrite").parquet(silver_filepath)
        print(f"Successfully processed {silver_count} rows and saved data to {silver_filepath}.")
        print(f"\n\n---Silver {data_type} Store completed successfully for {date_str}---\n\n")


    # Gold processing for label store
    print(f"\nProcessing label store for date {date_str} into gold store...")

    # Load data from silver directory
    partition = build_partition_name('silver', 'loan_data', date_str, 'parquet')
    df = load_data(spark, silver_data_dirs['loan_data'], partition)

    # Call the function to process loan data
    print(f"\nProcessing loan data for date: {date_str}...\n")
    df = create_label(df, dpd=30, mob=6)

    # check data after processing
    # pyspark_df_info(df)

    # Ensure that gold label store directory exists
    os.makedirs(os.path.dirname(gold_data_dirs['label_store']), exist_ok=True)

    # Save the cleaned data to the gold label store as parquet
    partition = build_partition_name('gold', 'label_store', date_str, 'parquet')
    gold_label_store_filepath = os.path.join(gold_data_dirs['label_store'], partition)

    df.write.mode("overwrite").parquet(gold_label_store_filepath)
    print(f"Successfully processed and saved data to {gold_label_store_filepath}.")
    print(f"\n\n---Gold Label Store completed successfully for {date_str}---\n\n")


    # Gold Processing for Feature Store
    print(f"\nProcessing feature store for date {date_str} into gold store...")

    # Join feature tables to create the gold feature store
    df = join_feature_tables(spark=spark, date=date_str) # type: ignore

    # Feature Engineering Steps
    print(f"\n\n---Feature Engineering for Gold Feature Store for {date_str}------\n\n")

    # Perform standard one-hot enconding for categorical columns
    print("Performing one-hot encoding for categorical columns...")
    df = one_hot_encode(df=df, column="Occupation", drop_label="Other")
    df = one_hot_encode(df=df, column="Credit_Mix", drop_label="Unknown")
    df = one_hot_encode(df=df, column="Payment_Behaviour", drop_label="Unknown")

    # Transform Loan Type into multiple type columns with respective counts
    print("Transforming Loan Type into multiple type columns with respective counts...")
    df = encode_loan_types_with_counts(df=df, loan_column_name="Type_of_Loan")

    print("Feature Engineering completed.")
    # pyspark_df_info(df)
    
    # Ensure that gold directory exists
    os.makedirs(os.path.dirname(gold_data_dirs['feature_store']), exist_ok=True)

    # Save the cleaned data to the gold directory as parquet
    partition_name = build_partition_name('gold', 'feature_store', date_str, 'parquet') # type: ignore
    gold_filepath = os.path.join(gold_data_dirs['feature_store'], partition_name)

    df.write.mode("overwrite").parquet(gold_filepath)
    print(f"Gold feature store saved to {gold_filepath}.")
    print(f"\n\n---Gold feature store completed successfully for {date_str}---\n\n")

# Stop the Spark session
spark.stop()
print("Data processing completed for all dates.")