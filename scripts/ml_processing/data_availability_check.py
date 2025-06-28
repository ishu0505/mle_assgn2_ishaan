import os
import argparse

from utils.validators import validate_date
from utils.data import check_partition_availability



if __name__ == "__main__":
    # get input arguments
    parser = argparse.ArgumentParser(description='Check availability of partitions in feature or label store.')
    parser.add_argument('--store_dir', type=str, required=True, help='Path to the store directory (feature or label store)')
    parser.add_argument('--start_date', type=str, default=True, help='Start date for checking partitions (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=True, help='End date for checking partitions (YYYY-MM-DD)')
    args = parser.parse_args()

    # validate input arguments
    if not os.path.exists(args.store_dir):
        raise ValueError(f"Store directory {args.store_dir} does not exist.")
    
    start_date = validate_date(date_str=args.start_date, output_DateType=True)
    end_date = validate_date(date_str=args.end_date, output_DateType=True)

    # check partition availability
    is_available = check_partition_availability(store_dir=args.store_dir, start_date=start_date, end_date=end_date) # type: ignore

    if is_available:
        print(f"Partitions are available from {args.start_date} to {args.end_date}.")
    else:
        raise ValueError(f"Partitions are not available from {args.start_date} to {args.end_date}.")
