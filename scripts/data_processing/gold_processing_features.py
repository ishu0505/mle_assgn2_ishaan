import os
import argparse
import re

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import IntegerType

from utils.data import load_data
from utils.validators import build_partition_name, validate_date, pyspark_info
from configs.data import silver_data_dirs, gold_data_dirs


def join_feature_tables(spark: SparkSession, date: str) -> DataFrame:
    """
    Function to join silver feature tables to the gold feature store for the given date.
    Args:
        spark (SparkSession): The Spark session.
        date (str): The date for which to join data, in the format YYYY-MM-DD.
    """
    # load all data
    print(f"\nLoading all feature data for date: {date} from silver directory...")
    table_dfs = {}
    for table_name, table_path in silver_data_dirs.items():
        partition_name = build_partition_name('silver', table_name, date, 'parquet')
        table_dfs[table_name] = load_data(spark=spark, input_directory=table_path, partition=partition_name)

    # Step 1: For customers with label, select only snapshot dates which correspond to loan application date (months on book = 0)
    print("\nExtracting customers with labels & their loan start date from label store...")
    relevant_dates = table_dfs['loan_data'] \
        .filter(F.col('mob') == 0) \
        .select("Customer_ID", "snapshot_date") \
        .withColumnRenamed("snapshot_date", "loan_start_date")
    print(f"Number of customers where loan start date is given: {relevant_dates.count()}")

    # Filter & aggregate clickstream data to include only relevant dates
    print("Filtering clickstream data for customers with label & aggregating clickstream data to loan start date...")
    clickstream_data_aggr = table_dfs['clickstream_data'] \
        .join(relevant_dates, on="Customer_ID", how="inner") \
        .filter(F.col('snapshot_date') <= F.col('loan_start_date')) \
        .groupBy("Customer_ID", "snapshot_date") \
        .agg(
            *[F.avg(F.col(f"fe_{i}")).alias(f"fe_{i}_avg") for i in range(1, 21)]
        )

    # Join all dataframes to Feature Store
    print("\nJoining current partitions...")
    df = table_dfs['customer_attributes'].join(table_dfs['customer_financials'], on=(["Customer_ID", "snapshot_date"]), how="inner")
    df = df.join(clickstream_data_aggr, on=(["Customer_ID", "snapshot_date"]), how="left_outer")
    print("Joined Customer Attributes, Customer Financials & Clickstream Data")
    pyspark_info(df)

    return df


def one_hot_encode(df: DataFrame, column: str, drop_label: str) -> DataFrame:
    '''
    Function to perform one-hot encoding on a specified column of a DataFrame.
    It creates new binary columns for each category in the input column.
    The new column names will be: original_column_name + "_" + category_value.
    The original column and the intermediate index column are kept in the DataFrame.
    Args:
        df (DataFrame): Input PySpark DataFrame.
        column (str): Column name of the string column to be one-hot encoded.
        drop_label (str): Label to be dropped from the one-hot encoding.
    Returns:
        DataFrame: DataFrame with the original column, the new index column, 
                   and the new one-hot encoded columns.
    '''

    # String Indexing to transform the string column into an index column
    indexed_col_name = column + "_index"
    indexer = StringIndexer(inputCol=column, outputCol=indexed_col_name, handleInvalid="skip")
    model = indexer.fit(df)
    df_with_index = model.transform(df)
    labels = model.labels  # Get the original string labels corresponding to the indices

    # If the drop_label is present in the labels, drop the corresponding column
    if drop_label in labels:
        labels.remove(drop_label)
    else:
        print(f"Warning: {drop_label} not found in labels. No column will be dropped.")    

    # Create a new DataFrame with the original column and the index column
    df_ohe = df_with_index
    for i, label in enumerate(labels):

        # Replace sequences of non-alphanumeric characters (excluding underscore) with a single underscore
        sanitized_label = re.sub(r'[^\w]+', '_', str(label))
        sanitized_label = sanitized_label.strip('_')
        
        # If sanitization results in an empty string, fallback
        if not sanitized_label:
            sanitized_label = f"category_{i}"

        ohe_column_name = f"{column}_{sanitized_label}"

        # Create the new binary (0 or 1) column for the current category
        df_ohe = df_ohe.withColumn(
            ohe_column_name,
            F.when(F.col(indexed_col_name) == float(i), 1).otherwise(0).cast(IntegerType())
        )
    
    # Drop original encoded column
    df_ohe = df_ohe.drop(column)
    
    return df_ohe


def encode_loan_types_with_counts(df: DataFrame, loan_column_name: str) -> DataFrame:
    """
    Encodes a column containing concatenated loan types into separate columns for each loan type,
    with each new column storing the count of that specific loan type.
    Args:
        df (DataFrame): The input PySpark DataFrame.
        loan_column_name (str): The name of the column containing the concatenated loan type strings.
    Returns:
        DataFrame: The DataFrame with new columns for each loan type count.
    """

    loan_types = [
        "Personal Loan", "Student Loan", "Payday Loan", "Auto Loan", "Home Equity Loan", 
        "Mortgage Loan", "Credit-Builder Loan", "Debt Consolidation Loan", "Not Specified"
    ]

    df_processed = df

    for loan_type in loan_types:
        # Sanitize the loan_type to make it suitable for a column name
        sanitized_loan_type = re.sub(r'[^\w]+', '_', loan_type)
        sanitized_loan_type = sanitized_loan_type.strip('_')
        if not sanitized_loan_type: # Fallback for purely special character names
            sanitized_loan_type = f"loan_type_{loan_types.index(loan_type)}"
            
        new_col_name = f"LoanType_{sanitized_loan_type}_Count"

        # Count occurrences of the loan_type string.
        df_processed = df_processed.withColumn(
            new_col_name,
            F.when(F.col(loan_column_name).isNotNull(), 
                 F.regexp_count(F.col(loan_column_name), F.lit(loan_type))
            ).otherwise(0).cast(IntegerType())
        )
    
    # Drop the original loan column
    df_processed = df_processed.drop(loan_column_name)
        
    return df_processed


if __name__ == "__main__":
    # get input arguments
    parser = argparse.ArgumentParser(description='Join silver feature tables to gold feature store.')
    parser.add_argument('--date', type=str, required=True, help='The date for which to join data, in the format YYYY-MM-DD')
    args = parser.parse_args()

    # validate input arguments
    if not args.date:
        raise ValueError("Argument --date is required.")
    date = validate_date(args.date)

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Gold Feature Joining") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Join feature tables to create the gold feature store
    df = join_feature_tables(spark=spark, date=date) # type: ignore

    # Feature Engineering Steps
    print(f"\n\n---Feature Engineering for Gold Feature Store for {date}------\n\n")

    # Perform standard one-hot enconding for categorical columns
    print("Performing one-hot encoding for categorical columns...")
    df = one_hot_encode(df=df, column="Occupation", drop_label="Other")
    df = one_hot_encode(df=df, column="Credit_Mix", drop_label="Unknown")
    df = one_hot_encode(df=df, column="Payment_Behaviour", drop_label="Unknown")

    # Transform Loan Type into multiple type columns with respective counts
    print("Transforming Loan Type into multiple type columns with respective counts...")
    df = encode_loan_types_with_counts(df=df, loan_column_name="Type_of_Loan")

    print("Feature Engineering completed.")
    pyspark_info(df)
    
    # Ensure that gold directory exists
    os.makedirs(os.path.dirname(gold_data_dirs['feature_store']), exist_ok=True)

    # Save the cleaned data to the gold directory as parquet
    partition_name = build_partition_name('gold', 'feature_store', date, 'parquet') # type: ignore
    gold_filepath = os.path.join(gold_data_dirs['feature_store'], partition_name)

    df.write.mode("overwrite").parquet(gold_filepath)
    print(f"Gold feature store saved to {gold_filepath}.")

    # Stop the Spark session
    spark.stop()
    print(f"\n\n---Gold feature store completed successfully for {date}---\n\n")