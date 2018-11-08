import numpy as np
import pandas as pd
from IPython.display import display

def analyze_base_data(train, export = False, unique_cols = ["ID"], output_dir=""):
    # show train data head
    print("Train:\n")
    display(train.head())
    print(train.describe())

    # drop unique columns for better analysis
    train_without_unique = train.drop(unique_cols, axis=1)

    # exclude non numeric columns
    non_numeric_columns = train_without_unique.select_dtypes(exclude=[np.number]).columns.tolist()
    train_numeric = train_without_unique.drop(non_numeric_columns, axis=1)

    avg = train_numeric.mean(axis=0)
    display(avg.head())
    if export: avg.to_csv(output_dir + "x_train_column_avg.csv")

    # how many features are enabled in general
    avg_of_rows = train_numeric.mean(axis=1)
    avg_of_rows['ID'] = train['ID']
    display(avg_of_rows.head())
    if export: avg_of_rows.to_csv(output_dir + "avg_of_rows.csv")



