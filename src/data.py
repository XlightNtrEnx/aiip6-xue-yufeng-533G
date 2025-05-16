import logging
import numpy as np
import pandas as pd
import sqlite3
from enum import Enum

from logger import formatter

conn = sqlite3.connect('data/bmarket.db')

tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
raw_df: pd.DataFrame = None
for table in tables['name']:
    print(f"Table: {table}")
    query = f"SELECT * FROM {table};"
    raw_df = pd.read_sql_query(query, conn)
    print("\n" + "-" * 50 + "\n")

processed_df = raw_df.copy()
target = 'target'


# Processing
def string_to_bool(value):
    if value == 'yes':
        return True
    elif value == 'no':
        return False
    else:
        return None


def get_features(df: pd.DataFrame):
    features = df.columns[df.columns != target]
    bool_features = df.columns[df.apply(lambda col: col.dropna().isin([True, False]).all()) & (df.columns != target)]
    num_features = df.select_dtypes(include=[np.number]).columns
    cat_features = features.difference(bool_features.union(num_features))
    return features, bool_features, cat_features, num_features


# Everything
processed_df.replace('unknown', None, inplace=True)

# Client ID
processed_df.drop('Client ID', axis=1, inplace=True)

# Age (years)
processed_df['Age'] = processed_df['Age'].str.extract(r'(\d+)').astype(int)
processed_df.rename(columns={'Age': 'Age (years)'}, inplace=True)
processed_df = processed_df[(processed_df['Age (years)'] >= 0) & (processed_df['Age (years)'] <= 122)]
processed_df['Age (years)'] = np.log1p(processed_df['Age (years)'])
processed_df.rename(columns={"Age (years)": "Age (years)_log"}, inplace=True)

# Credit Default
processed_df['Credit Default'] = processed_df['Credit Default'].apply(string_to_bool)

# Housing Loan
processed_df.drop('Housing Loan', axis=1, inplace=True)

# Personal Loan
processed_df.drop('Personal Loan', axis=1, inplace=True)

# Contact Method
processed_df['Contact Method'] = processed_df['Contact Method'].str.lower()


class ContactMethod(Enum):
    CELL = "cell"
    TELEPHONE = "telephone"


contact_method_map = {method.value: method.value for method in ContactMethod}
contact_method_map.update({
    'cellular': ContactMethod.CELL.value
})
processed_df['Contact Method'] = processed_df['Contact Method'].apply(
    lambda x: contact_method_map.get(x) if contact_method_map.get(x, None) else None
)

# Previous Contact Days
processed_df['Previous Contact Days'] = processed_df['Previous Contact Days'].apply(
    lambda x: np.nan if x >= 999 else x
)
processed_df['Previous Contact Days'] = np.log1p(processed_df['Previous Contact Days'])
processed_df.rename(columns={"Previous Contact Days": "Previous Contact Days_log"}, inplace=True)

# Subscription Status
processed_df['Subscription Status'] = processed_df['Subscription Status'].apply(string_to_bool)
processed_df.rename(columns={'Subscription Status': target}, inplace=True)

# Campaign Calls
processed_df["Campaign Calls"] = processed_df["Campaign Calls"].apply(lambda x: abs(x))
processed_df['Campaign Calls'] = np.log1p(processed_df['Campaign Calls'])
processed_df.rename(columns={"Campaign Calls": "Campaign Calls_log"}, inplace=True)

# Data for export
features, bool_features, cat_features, num_features = get_features(processed_df)
X = processed_df.drop(columns=[target])
y = processed_df[target]

# Log data
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(f"logs/data.log", mode="w")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def log_column_uniques(df: pd.DataFrame):
    log = ""
    for col in df.columns:
        log += f"\nColumn: {col}\n{df[col].unique()}\n"
    logger.info(log)


log_column_uniques(processed_df)
