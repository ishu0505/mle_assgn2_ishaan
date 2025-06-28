"""
This script prepares the training data for the machine learning model.
It reads data from the feature store and label store, processes it, and saves the training,
validation, test, and out-of-time (OOT) datasets in the specified directories.
"""

import os
import argparse
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from configs.data import gold_data_dirs, training_data_dir
from utils.data import load_data
from utils.validators import validate_date, pyspark_info, build_partition_name


def load_and_merge_data(spark: SparkSession, datelist: list) -> DataFrame:
    """
    Load feature and label data from the gold directory and merge them together.
    Args:
        spark (SparkSession): Spark session to use for loading data.
        datelist (list): List of dates for which to load data.
    Returns:
        pd.DataFrame: Merged DataFrame containing feature and label data.
    """
    # Load feature data from gold feature store & label data from gold label store for given date range
    print(f"\nLoading and merging data from gold feature store and label store...")
    feature_store, label_store = [], []
    for date in datelist:
        partition = build_partition_name('gold', 'feature_store', date, 'parquet')
        feature_store.append(load_data(spark=spark, input_directory=gold_data_dirs['feature_store'], partition=partition))
        partition = build_partition_name('gold', 'label_store', date, 'parquet')
        label_store.append(load_data(spark=spark, input_directory=gold_data_dirs['label_store'], partition=partition))

    if feature_store and label_store:
        df_feature_store = feature_store[0]
        df_label_store = label_store[0]

        for i in range(1, len(feature_store)):
            df_feature_store = df_feature_store.unionByName(feature_store[i])
            df_label_store = df_label_store.unionByName(label_store[i])

    else:
        raise ValueError("No data found in feature store or label store for the specified date range.")
    
    # Merge feature and label dataframes on customer_id
    print(f"\nMerging feature store and label store dataframes on customer_id...")
    df = df_feature_store.join(df_label_store[['Customer_ID', 'label']], on='Customer_ID', how='inner')
    pyspark_info(df)

    return df


if __name__ == "__main__":
    # get input arguments
    parser = argparse.ArgumentParser(description='Join silver feature tables to gold feature store.')
    parser.add_argument('--start', type=str, required=True, help='The start date (partition) from which to load data, in the format YYYY-MM-DD')
    parser.add_argument('--end', type=str, required=True, help='The end date (partition) until which to load data, in the format YYYY-MM-DD')
    parser.add_argument('--oot', type=int, default=3, help='Number of months to use for out-of-time (OOT) test set (default: 3 months)')
    args = parser.parse_args()

    # validate input arguments
    if not args.start or not args.end:
        raise ValueError("Argument --date is required.")
    start_date = validate_date(args.start)
    end_date = validate_date(args.end)

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Training Data Preparation") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # create list of dates for which to load data
    date_list = pd.date_range(start=start_date, end=end_date, freq='MS').strftime('%Y-%m-%d').tolist()

    train_size = len(date_list) - args.oot
    if train_size < 12:
        raise ValueError(f"Not enough data for training. \
                         At least 12 months of data are required, but only {train_size} months are available. \
                         Please adjust the date range or the OOT parameter.")

    # Load and merge data from feature store and label store
    df = load_and_merge_data(spark, date_list)

    # Split total data into training, validation, test, and out-of-time (OOT) sets
    print(f"\nSplitting data into training, validation, test, and out-of-time (OOT) sets...")

    datasets = {}
    # create separate oot datasets from the last months
    for i in range(args.oot):
        oot_date = df.select(F.max('snapshot_date')).collect()[0][0] - relativedelta(months=i)
        oot_df = df.filter(F.col('snapshot_date') == oot_date)
        datasets[f'oot_{args.oot - i}'] = oot_df
    
    # rest of the data is used for training, validation, and test sets
    df = df.filter(df.snapshot_date < oot_date) # type: ignore
    datasets['training'], datasets['validation'], datasets['test'] = df.randomSplit([0.7, 0.15, 0.15], seed=42)

    for name, dataset in datasets.items():
        print(f"Dataset {name} has {dataset.count()} records.")
    
    # Ensure that training directory exists
    os.makedirs(os.path.dirname(training_data_dir), exist_ok=True)

    # Save the data to the training directory as Parquet files
    for name, dataset in datasets.items():
        partition = f"{name}.parquet"
        filepath = os.path.join(training_data_dir, partition)
        dataset.write.mode("overwrite").parquet(filepath)
        print(f"Saved {name} dataset to {filepath}.")

    # Stop the Spark session
    spark.stop()
    print(f"\n\n---Training Data Preparation completed successfully---\n\n")