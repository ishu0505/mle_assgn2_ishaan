import os
import argparse 

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType

from utils.data import load_data
from utils.validators import build_partition_name, pyspark_info
from configs.data import silver_data_dirs, gold_data_dirs


def create_label(df: DataFrame, dpd=30, mob=6):
    '''
    Function to process loan data from silver table to the gold label store.
    Args:
        df (DataFrame): Input DataFrame containing loan data.
        dpd (int): Days past due for label creation. Default is 30.
        mob (int): Months on book for label creation. Default is 6.
    Returns:
        DataFrame: Processed DataFrame with labels and metadata.
    '''

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    return df


# main function to run the script
if __name__ == "__main__":

    # parse command line arguments
    parser = argparse.ArgumentParser(description='Create gold label store from silver layer.')
    parser.add_argument('--date', type=str, required=True, help='Date for which data is being processed (format: YYYY-MM-DD).')
    args = parser.parse_args()

    # Validate input arguments
    if not args.date:
        raise ValueError("Argument --date is required.")

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Gold Label Processing") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Load data from silver directory
    partition = build_partition_name('silver', 'loan_data', args.date, 'parquet')
    df = load_data(spark, silver_data_dirs['loan_data'], partition)

    # Call the function to process loan data
    print(f"\nProcessing loan data for date: {args.date}...\n")
    df = create_label(df, dpd=30, mob=6)

    # check data after processing
    pyspark_info(df)

    # Ensure that gold label store directory exists
    os.makedirs(os.path.dirname(gold_data_dirs['label_store']), exist_ok=True)

    # Save the cleaned data to the gold label store as parquet
    partition = build_partition_name('gold', 'label_store', args.date, 'parquet')
    gold_label_store_filepath = os.path.join(gold_data_dirs['label_store'], partition)

    df.write.mode("overwrite").parquet(gold_label_store_filepath)
    print(f"Successfully processed and saved data to {gold_label_store_filepath}.")

    # Stop the Spark session
    spark.stop()
    print(f"\n\n---Gold Label Store completed successfully for {args.date}---\n\n")