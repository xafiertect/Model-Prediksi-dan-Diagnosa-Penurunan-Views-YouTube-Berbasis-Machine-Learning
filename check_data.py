import pandas as pd
import json

df = pd.read_csv('data/Data_Merged_Fix.csv')
report = {}
report['shape'] = df.shape
missing = df.isnull().sum()
report['missing_values'] = missing[missing > 0].to_dict()

# Check for duplicates
report['duplicate_rows'] = df.duplicated().sum()

# Check for data types
report['data_types'] = df.dtypes.astype(str).to_dict()

# print to output
for k, v in report.items():
    print(f"--- {k} ---")
    if isinstance(v, dict):
        for col, val in v.items():
            print(f"{col}: {val}")
    else:
        print(v)
    print()
