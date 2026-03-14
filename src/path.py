import pandas as pd
from pathlib import Path

# 1. Get the directory where this script lives (src/)
# 2. Get its parent (the project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 3. Point to the data
data_path = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
clean_data_path = PROJECT_ROOT / "data" / "processed" / "clean_telco.csv"
processed_data_path = PROJECT_ROOT / "data" / "processed" / "processed_telco.csv"

def get_data_path():
    return data_path

def get_clean_data_path():
    return clean_data_path

def get_processed_data_path():
    return processed_data_path