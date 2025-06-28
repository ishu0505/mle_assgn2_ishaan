import os
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import DataType, StringType, IntegerType, FloatType, DateType, DecimalType, BooleanType
from pyspark.sql.functions import regexp_extract, col

from .validators import validate_file_path


def load_data(spark: SparkSession, input_directory: str, partition: str | None) -> DataFrame:
    """
    Load data from a CSV file into a Spark DataFrame.
    Args:
        spark (SparkSession): The Spark session object.
        input_directory (str): The directory to the input files. Files should be in CSV or Parquet format.
        partition (str): A specific partition to load. If not provided, all data in the given directory will be loaded.
    Returns:
        DataFrame: A Spark DataFrame containing the loaded data.
    """
    # Check if input directory exists
    if not os.path.exists(input_directory):
        raise FileNotFoundError(f"Input directory {input_directory} does not exist.")
    # If partition is provided, check if file exists and load it
    if partition:
        file_path = validate_file_path(os.path.join(input_directory, partition))
        # Load CSV or parquet file based on the file extension
        if file_path.endswith('.parquet'):
            df = spark.read.parquet(file_path)
        elif file_path.endswith('.csv'):
            df = spark.read.csv(file_path, header=True, inferSchema=True)
        else:
            raise ValueError(f"Unsupported file format for {file_path}. Only CSV and Parquet are supported.")
        print(f"Loaded data from {file_path}. Row count: {df.count()}")
        return df
    
    # If no partition is provided, load all data in the directory
    else:
        # check which file types are in the directory
        file_types = set()
        for file in os.listdir(input_directory):
            if file.endswith('.csv'):
                file_types.add('csv')
            elif file.endswith('.parquet'):
                file_types.add('parquet')
            else:
                raise ValueError(f"Unsupported file format in {input_directory}. Only CSV and Parquet are supported.")
        # If both file types are present, raise an error
        if len(file_types) > 1:
            raise ValueError(f"Multiple file types found in {input_directory}. Please specify a partition.")
        
        # Load files based on the file type
        if 'parquet' in file_types:
            all_files_path = os.path.join(input_directory, "*.parquet")
            print(f"Loading parquet files with wildcard: {all_files_path}")
            df = spark.read.parquet(all_files_path)
        else:
            all_files_path = os.path.join(input_directory, "*.csv")
            print(f"Loading csv files with wildcard: {all_files_path}")
            df = spark.read.csv(all_files_path, header=True, inferSchema=True)
        
        print(f"Loaded data from all files from {input_directory}. Row count: {df.count()}")
        return df


def transform_data_in_column(df: DataFrame, column_name: str, dtype: DataType):
    """
    Transforms data from a specified column in a DataFrame.
    Args:
        df (DataFrame): The DataFrame to transform data in.
        column_name (str): The name of the column to transform the data in.
        dtype (DataType): The target data type to transform the column to.
    Returns:
        df (DataFrame): The DataFrame with the transformed column.
    """

    # Regex pattern depending on the data type
    if isinstance(dtype, StringType):
        pattern = r'.*'  # no selection (everything is allowed)
    elif isinstance(dtype, IntegerType):
        pattern = r'^\d+$'  # Digits only
    elif isinstance(dtype, FloatType) or isinstance(dtype, DecimalType):
        pattern = r'^\d+(\.\d+)?$'  # Digits with optional decimal point
    elif isinstance(dtype, BooleanType):
        pattern = r'^(true|false|1|0)$'
    elif isinstance(dtype, DateType):
        pattern = r'^\d{4}-\d{2}-\d{2}$'  # YYYY-MM-DD format
    else:
        raise ValueError(f"Unsupported data type: {dtype}")
    
    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    # Extract the matching part of the string using regexp_extract
    df = df.withColumn(
        column_name,
        regexp_extract(col(column_name).cast(StringType()), pattern, 0)
    )

    # Cast the column to the specified data type & return the DataFrame
    return df.withColumn(column_name, df[column_name].try_cast(dtype))


def check_partition_availability(store_dir: str, start_date: datetime, end_date: datetime) -> bool:
    """
    Checks if the respective store (feature or label) contains monthly partitions for the given time frame.
    Args:
        store_dir (str): Path to the directory containing feature store files.
        start_date (datetime): Start date of the time frame to check.
        end_date (datetime): End date of the time frame to check.
    Returns:
        bool: True if at monthly partitions exist for this time frame, False otherwise.
    """
    pattern = re.compile(r"(\d{4})_(\d{2})_(\d{2})")  # Pattern for YYYY_MM_DD format
    partition_dates = []

    for fname in os.listdir(store_dir):
        match = pattern.search(fname)
        if match:
            date_str = match.group(0)
            try:
                partition_date = datetime.strptime(date_str, "%Y_%m_%d")
                partition_dates.append(partition_date)
            except ValueError:
                continue

    if not partition_dates:
        print(f"No partitions found in {store_dir}.")
        return False

    partition_dates = sorted(set(partition_dates)) # Remove duplicates and sort

    current_date = start_date
    while current_date <= end_date:
        if current_date not in partition_dates:
            print(f"Missing partition for date: {current_date.strftime('%Y-%m-%d')}")
            return False
        current_date += relativedelta(months=1)  # Increment by one month

    return True