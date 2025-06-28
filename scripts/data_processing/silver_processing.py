import os
import argparse

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, DateType

from utils.data import load_data, transform_data_in_column
from utils.validators import build_partition_name, pyspark_info, validate_date
from configs.data import bronze_data_dirs, silver_data_dirs, data_types


def silver_processing_clickstream_data(df: DataFrame):
    '''
    Function to process clickstream data from bronze table to silver table.
    Args:
        df (DataFrame): The DataFrame containing the clickstream data.
    Returns:
        DataFrame: The processed DataFrame with cleaned and transformed data.
    '''

    # Handle null values (0 for numerical, remove if customer_id or snapshot_date are corrupted)
    for column in data_types['clickstream_data'].keys():
        if column == "Customer_ID" or column == "snapshot_date":
            df = df.filter(F.col(column).isNotNull())
        else:
            df = df.fillna(0, subset=[column])
    
    # Remove duplicates
    df = df.dropDuplicates(["Customer_ID", "snapshot_date"])

    return df


def silver_processing_customer_attributes(df: DataFrame):
    '''
    Function to process customer attributes data from bronze table to silver table. Enforces schema and data quality checks.
    Args:
        df (DataFrame): The DataFrame containing the customer attributes data.
    Returns:
        DataFrame: The processed DataFrame with cleaned and transformed data.
    '''

    # drop columns name and ssn
    df = df.drop("Name", "SSN")

    # Handle corrupted values (1 of 15 categories for occupation, sanity check for age, remove if customer_id or snapshot_date are corrupted)
    df = df.filter(F.col("Customer_ID").isNotNull() | F.col("snapshot_date").isNotNull())
    df = df.withColumn("Age", F.when((F.col("Age") >= 0) & (F.col("Age") <= 100), F.col("Age")).otherwise(0))
    df = df.withColumn("Occupation", 
                       F.when(F.col("Occupation").isin([
                           "Scientist", 
                           "Media_Manager", 
                           "Musician", 
                           "Lawyer", 
                           "Teacher", 
                           "Developer", 
                           "Writer", 
                           "Architect", 
                           "Mechanic", 
                           "Entrepreneur", 
                           "Journalist", 
                           "Doctor", 
                           "Engineer", 
                           "Accountant", 
                           "Manager", 
                           "Other"
                           ]), 
                        F.col("Occupation")).otherwise("Other"))

    # Remove duplicates
    df = df.dropDuplicates(["Customer_ID", "snapshot_date"])

    return df


def silver_processing_customer_financials(df: DataFrame):
    '''
    Function to process customer financials data from bronze table to silver table. Enforces schema and data quality checks.
    Args:
        df (DataFrame): The DataFrame containing the customer financials data.
    Returns:
        DataFrame: The processed DataFrame with cleaned and transformed data.
    '''

    # identifiers have to be present
    df = df.filter(F.col("Customer_ID").isNotNull() | F.col("snapshot_date").isNotNull())

    # enforce false for boolean values if not present
    df = df.withColumn("Payment_of_Min_Amount", F.when(F.col("Payment_of_Min_Amount").isNotNull(), F.col("Payment_of_Min_Amount")).otherwise(False))

    # numerical values checking (> 0 for all, < 100 for interest rate and credit utilization ratio)
    df = df.withColumn("Annual_Income", F.when(F.col("Annual_Income") > 0, F.col("Annual_Income")).otherwise(0))
    df = df.withColumn("Monthly_Inhand_Salary", F.when(F.col("Monthly_Inhand_Salary") > 0, F.col("Monthly_Inhand_Salary")).otherwise(0))
    df = df.withColumn("Num_Bank_Accounts", F.when(F.col("Num_Bank_Accounts") > 0, F.col("Num_Bank_Accounts")).otherwise(0))
    df = df.withColumn("Num_Credit_Card", F.when(F.col("Num_Credit_Card") > 0, F.col("Num_Credit_Card")).otherwise(0))
    df = df.withColumn("Interest_Rate", F.when((F.col("Interest_Rate") > 0) & (F.col("Interest_Rate") < 100), F.col("Interest_Rate")).otherwise(0))
    df = df.withColumn("Num_of_Loan", F.when(F.col("Num_of_Loan") > 0, F.col("Num_of_Loan")).otherwise(0))
    df = df.withColumn("Delay_from_due_date", F.when(F.col("Delay_from_due_date") > 0, F.col("Delay_from_due_date")).otherwise(0))
    df = df.withColumn("Num_of_Delayed_Payment", F.when(F.col("Num_of_Delayed_Payment") > 0, F.col("Num_of_Delayed_Payment")).otherwise(0))
    df = df.withColumn("Changed_Credit_Limit", F.when(F.col("Changed_Credit_Limit") > 0, F.col("Changed_Credit_Limit")).otherwise(0))
    df = df.withColumn("Num_Credit_Inquiries", F.when(F.col("Num_Credit_Inquiries") > 0, F.col("Num_Credit_Inquiries")).otherwise(0))
    df = df.withColumn("Outstanding_Debt", F.when(F.col("Outstanding_Debt") > 0, F.col("Outstanding_Debt")).otherwise(0))
    df = df.withColumn("Credit_Utilization_Ratio", F.when((F.col("Credit_Utilization_Ratio") > 0) & (F.col("Credit_Utilization_Ratio") < 100), F.col("Credit_Utilization_Ratio")).otherwise(0))
    df = df.withColumn("Total_EMI_per_month", F.when(F.col("Total_EMI_per_month") > 0, F.col("Total_EMI_per_month")).otherwise(0))
    df = df.withColumn("Amount_invested_monthly", F.when(F.col("Amount_invested_monthly") > 0, F.col("Amount_invested_monthly")).otherwise(0))
    df = df.withColumn("Monthly_Balance", F.when(F.col("Monthly_Balance") > 0, F.col("Monthly_Balance")).otherwise(0))

    # categorical values checking
    df = df.withColumn("Credit_Mix",
                       F.when(F.col("Credit_Mix").isin([
                           "Good", 
                           "Bad", 
                           "Standard", 
                           "Unknown"
                           ]), 
                        F.col("Credit_Mix")).otherwise("Unknown"))
    df = df.withColumn("Payment_Behaviour",
                       F.when(F.col("Payment_Behaviour").isin([
                           "Low_spent_Small_value_payments", 
                           "Low_spent_Medium_value_payments", 
                           "Low_spent_Large_value_payments", 
                           "High_spent_Small_value_payments", 
                           "High_spent_Medium_value_payments", 
                           "High_spent_Large_value_payments"
                           ]), 
                        F.col("Payment_Behaviour")).otherwise("Unknown"))
    
    # Transform credit history age to no of months 
    years_col = F.coalesce(
        F.regexp_extract(F.col("Credit_History_Age"), r"(\d+)\s+Year(s)?", 1).try_cast(IntegerType()), 
        F.lit(0)
    )
    months_col = F.coalesce(
        F.regexp_extract(F.col("Credit_History_Age"), r"(\d+)\s+Month(s)?", 1).try_cast(IntegerType()), 
        F.lit(0)
    )
    df = df.withColumn("Credit_History_Age", ((years_col * 12) + months_col).cast(IntegerType()))

    # Remove duplicates
    df = df.dropDuplicates(["Customer_ID", "snapshot_date"])
    
    return df


def silver_processing_label_store(df: DataFrame):
    '''
    Function to process loan data from bronze table to silver table. Enforces schema and data quality checks.
    Args:
        df (DataFrame): The DataFrame containing the loan data.
    Returns:
        DataFrame: The processed DataFrame with cleaned and transformed data.
    '''

    # augment data: add month on book
    df = df.withColumn("mob", F.col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(F.try_divide(F.col("overdue_amt"), F.col("due_amt"))).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(F.col("installments_missed") > 0, F.add_months(F.col("snapshot_date"), -1 * F.col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(F.col("overdue_amt") > 0.0, F.datediff(F.col("snapshot_date"), F.col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    return df


# Main function to run the silver processing scripts
if __name__ == "__main__":

    # parse command line arguments
    parser = argparse.ArgumentParser(description='Process data from bronze to silver layer.')
    parser.add_argument('--type', type=str, required=True, choices=['clickstream_data', 'customer_attributes', 'customer_financials', 'loan_data'],
                        help='Type of data to process: clickstream_data, customer_attributes, customer_financials, or loan_data.')
    parser.add_argument('--date', type=str, required=True, help='Date for which data is being processed (format: YYYY-MM-DD).')
    args = parser.parse_args()

    # Validate input arguments
    if not args.date or not args.type:
        raise ValueError("All arguments --date and --type are required.")
    date = validate_date(args.date)

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Silver Processing") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Load data from bronze directory
    partition = build_partition_name('bronze', args.type, date, 'csv') # type: ignore
    df = load_data(spark, bronze_data_dirs[args.type], partition)
    bronze_count = df.count()

    # Type casting
    for column, dtype in data_types[args.type].items():
        df = transform_data_in_column(df, column, dtype)

    # Process data based on type
    if args.type == 'clickstream_data':
        df = silver_processing_clickstream_data(df)
    elif args.type == 'customer_attributes':
        df = silver_processing_customer_attributes(df)
    elif args.type == 'customer_financials':
        df = silver_processing_customer_financials(df)
    elif args.type == 'loan_data':
        df = silver_processing_label_store(df)
    else:
        raise ValueError(f"Unsupported data type: {args.type}")

    # check data after cleaning
    silver_count = df.count()
    if silver_count != bronze_count:
        print(f"Warning: Row count changed from {bronze_count} to {silver_count} after cleaning for partition {partition}.")
    pyspark_info(df)

    # Ensure that silver directory exists
    os.makedirs(os.path.dirname(silver_data_dirs[args.type]), exist_ok=True)

    # Save the cleaned data to the silver directory as parquet
    partition_name = build_partition_name('silver', args.type, date, 'parquet') # type: ignore
    silver_filepath = os.path.join(silver_data_dirs[args.type], partition_name)

    df.write.mode("overwrite").parquet(silver_filepath)
    print(f"Successfully processed {silver_count} rows and saved data to {silver_filepath}.")

    # Stop the Spark session
    spark.stop()
    print(f"\n\n---Silver {args.type} Store completed successfully for {date}---\n\n")