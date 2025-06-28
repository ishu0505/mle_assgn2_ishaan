from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, DecimalType, BooleanType

BASE_DIR = "/opt/airflow/"
# BASE_DIR = "" # For local testing, set to empty string

# Source Data files
source_data_files = {
    'clickstream_data': f'{BASE_DIR}data/feature_clickstream.csv',
    'customer_attributes': f'{BASE_DIR}data/features_attributes.csv',
    'customer_financials': f'{BASE_DIR}data/features_financials.csv',
    'loan_data': f'{BASE_DIR}data/lms_loan_daily.csv'
}

# Datalake Medallion directories
bronze_data_dirs = {
    'clickstream_data': f'{BASE_DIR}datamart/bronze/clickstream_data/',
    'customer_attributes': f'{BASE_DIR}datamart/bronze/customer_attributes/',
    'customer_financials': f'{BASE_DIR}datamart/bronze/customer_financials/',
    'loan_data': f'{BASE_DIR}datamart/bronze/loan_data/'
}

silver_data_dirs = {
    'clickstream_data': f'{BASE_DIR}datamart/silver/clickstream_data/',
    'customer_attributes': f'{BASE_DIR}datamart/silver/customer_attributes/',
    'customer_financials': f'{BASE_DIR}datamart/silver/customer_financials/',
    'loan_data': f'{BASE_DIR}datamart/silver/loan_data/'
}

gold_data_dirs = {
    'feature_store': f'{BASE_DIR}datamart/gold/feature_store/',
    'label_store': f'{BASE_DIR}datamart/gold/label_store/'
}

# Training Data directory
training_data_dir = f'{BASE_DIR}datamart/training/'

# Model Registry directory
model_store_dir = f'{BASE_DIR}model_store/'

# Data Type configurations
data_types = {
    'clickstream_data': {
        "fe_1": IntegerType(),
        "fe_2": IntegerType(),
        "fe_3": IntegerType(),
        "fe_4": IntegerType(),
        "fe_5": IntegerType(),
        "fe_6": IntegerType(),
        "fe_7": IntegerType(),
        "fe_8": IntegerType(),
        "fe_9": IntegerType(),
        "fe_10": IntegerType(),
        "fe_11": IntegerType(),
        "fe_12": IntegerType(),
        "fe_13": IntegerType(),
        "fe_14": IntegerType(),
        "fe_15": IntegerType(),
        "fe_16": IntegerType(),
        "fe_17": IntegerType(),
        "fe_18": IntegerType(),
        "fe_19": IntegerType(),
        "fe_20": IntegerType(),
        "Customer_ID": StringType(),
        "snapshot_date": DateType()
    },
    'customer_attributes': {
        "Age": IntegerType(),
        "Occupation": StringType(),
        "Customer_ID": StringType(),
        "snapshot_date": DateType()
    },
    'customer_financials': {
        "Customer_ID": StringType(),
        "Annual_Income": DecimalType(18, 2),
        "Monthly_Inhand_Salary": DecimalType(18, 2),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": DecimalType(18, 2),
        "Credit_Utilization_Ratio": DecimalType(4, 2),
        "Credit_History_Age": StringType(),
        "Payment_of_Min_Amount": BooleanType(),
        "Total_EMI_per_month": DecimalType(18, 2),
        "Amount_invested_monthly": DecimalType(18, 2),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": DecimalType(18, 2),
        "snapshot_date": DateType()
    },
    'loan_data': {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType()
    }
}

