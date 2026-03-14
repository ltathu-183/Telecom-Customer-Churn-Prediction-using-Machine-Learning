import sys

import pandas as pd
from data.preprocessing import DataPreprocessor
import sys
from pathlib import Path
import pandas as pd

# Add root to sys.path
root_path = str(Path.cwd().parent)
if root_path not in sys.path:
    sys.path.append(root_path)

# Force a reload in case the notebook is "remembering" an old version of the file
import importlib
import src.path
importlib.reload(src.path)

from src.path import get_data_path
from src.path import get_clean_data_path

def main():

    df = pd.read_csv(get_data_path())

    # 1. Initialize and process
    preprocessor = DataPreprocessor(df)
    X_clean, y_clean = preprocessor.run_all() # Unpack the tuple here!

    # 2. Combine them back into one DataFrame
    df_final = X_clean.copy()
    df_final['Churn'] = y_clean

    # 3. Save
    df_final.to_csv(get_clean_data_path(), index=False)     


if __name__ == "__main__":
    main()