import os
import re
from datetime import datetime

from pyspark.sql import DataFrame
from pyspark.sql.functions import count


def validate_date(date_str: str, output_DateType: bool = False) -> datetime | str:
    """
    Validate the date string against the YYYY-MM-DD format.
    Args:
        date_str (str): The date string to validate.
        output_dtype (DataType): The expected output data type, either DateType or StringType. Default is String.
    Returns:
        datetime: The validated date as either a datetime object or a string.
    """
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        raise ValueError(f"Invalid date format: {date_str}. Expected format is YYYY-MM-DD.")
    
    return datetime.strptime(date_str, "%Y-%m-%d") if output_DateType else date_str


def validate_file_path(file_path):
    """
    Validate if the file path exists.
    Args:
        file_path (str): The file path to validate.
    Returns:
        str: The validated file path.
    """
    if not re.match(r'^[\w\-. /]+$', file_path) or not os.path.exists(file_path):
        raise ValueError(f"Invalid file path: {file_path}")
    
    return file_path


def build_partition_name(level: str, table_name: str, date: str, type: str) -> str:
    """
    Builds a partition name for saving data based on the table name and date.
    Args:
        table_name (str): The name of the table.
        date (str): The date in YYYY-MM-DD format.
    Returns:
        str: The partition name in the format 'bronze_<table_name>_<date>.csv'.
    """
    if not re.match(r'^[a-zA-Z0-9_]+$', table_name):
        raise ValueError(f"Invalid table name: {table_name}. Only alphanumeric characters and underscores are allowed.")

    return f"{level}_{table_name}_{date.replace('-', '_')}.{type}"


def pyspark_info(df: DataFrame):
    """
    Prints information about a PySpark DataFrame, similar to pandas.DataFrame.info().
    Includes total number of rows, and for each column: its name, 
    the number of non-null entries, and its data type.

    Args:
        df: The PySpark DataFrame to get information about.
    """

    if not isinstance(df, DataFrame):
        print("Error: Input is not a PySpark DataFrame.")
        return

    total_rows = df.count()
    print(f"\nTotal entries: {total_rows}")
    
    num_columns = len(df.columns)
    print(f"Data columns (total {num_columns} columns):")

    longest_col_name = max(len(col_name) for col_name in df.columns)
    if longest_col_name < 25:
        longest_col_name = 25
    
    header = f"{'#':<3} {'Column':<{longest_col_name}} {'Non-Null Count':<18} {'Dtype':<15}"
    print(header)
    print("--- " + "-"*longest_col_name + " " + "-"*18 + " " + "-"*15)

    # get non-null counts for all columns
    agg_exprs = [count(c).alias(c) for c in df.columns]
    non_null_counts_row = df.agg(*agg_exprs).collect()[0]

    for i, (col_name, col_type) in enumerate(df.dtypes):
        non_null_count = non_null_counts_row[col_name]
        print(f"{i:<3} {col_name:<{longest_col_name}} {non_null_count:<18} {col_type:<15}")
    print("\n")