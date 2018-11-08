# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display


def cleanupData(train, verbose = False, export = False, unique_cols = ["ID"], output_dir=""):
    x_train = train
    cols_to_drop = []
    # try to find columns where the value is constant (no need to analyse those)
    for col in x_train:
        if (x_train[col] == x_train[col][0]).all():
            # now we know column value is constant
            if verbose: print(f"Column: {col} - Value {x_train[col][0]}")
            cols_to_drop.append(col)

    if verbose: print("Columns that are constant:")
    if verbose:  display(cols_to_drop)

    pd.DataFrame(cols_to_drop).to_csv(output_dir+"x_train_constant_columns.csv")
    # drop constant columns from training data
    x_train = x_train.drop(cols_to_drop,axis=1)

    if export: x_train.to_csv(output_dir+"x_train_without_constant_columns.csv")

    # get data only so drop ID
    x_train_without_id = x_train.drop(unique_cols,axis=1)

    # find identical rows - do not mark first occurence
    x_train_duplicates_rows = x_train_without_id.duplicated(keep='first')
    x_train_duplicates_rows_df = pd.DataFrame(x_train_duplicates_rows)

    if verbose: print(f"Has duplicate rows: {x_train_duplicates_rows.any()}")
    if export: x_train_duplicates_rows_df.to_csv(output_dir+"x_train_duplicate_rows.csv")

    # get row index for duplicate rows
    row_index_duplicate_rows = x_train_duplicates_rows_df.index[x_train_duplicates_rows_df[0] == True].tolist()
    # remove duplicate rows from set
    x_train = x_train.drop(row_index_duplicate_rows, axis=0)
    #TODO: CHECK IF THIS REALLY IS CORRECT!!

    if verbose: print(f"Duplicate Rows: {row_index_duplicate_rows}")
    # find identical columns - do not mark first occurence
    # transpose dataframe first, find duplicate "rows", transpose again
    x_train_duplicates_cols = x_train_without_id.T.duplicated(keep='first').T
    x_train_duplicates_cols_df = pd.DataFrame(x_train_duplicates_cols)

    if verbose: print(f"Has duplicate cols: {x_train_duplicates_cols.any()}")
    if export: x_train_duplicates_cols_df.to_csv(output_dir+"x_train_duplicate_cols.csv")

    # get column headers for duplicate columns
    column_headers_duplicate_columns = x_train_duplicates_cols_df.index[x_train_duplicates_cols_df[0] == True].tolist()
    # remove duplicate columns from set
    x_train = x_train.drop(column_headers_duplicate_columns,axis=1)

    if verbose: print(f"Duplicate columns: {column_headers_duplicate_columns}")
    if export: x_train.to_csv(output_dir+"x_train_after_removing_all_duplicates.csv")

    if verbose: print(f"Intermediate x_train after removing duplicates: \n{x_train.head()}")

    # get non numeric columns
    non_numeric_columns = x_train.select_dtypes(exclude=[np.number]).columns.tolist()
    if verbose: print(f"Non numeric cols: {non_numeric_columns}")

    # get only boolean values from the dataset
    x_train_bool = x_train.drop(non_numeric_columns, axis=1).astype(bool)

    if verbose: display(x_train_bool.head())
    if export: x_train_bool.to_csv(output_dir+"x_train_boolean_only.csv")

    # get data only so drop ID
    x_train_bool_without_id = x_train_bool.drop(unique_cols, axis=1)
    if verbose: display(x_train_bool_without_id.head())
    if export: x_train_bool_without_id.to_csv(output_dir+"x_train_boolean_only_without_ID.csv")

    # now get the inverted of all 1/0 values
    # we want to check if we get additional correlations (e.g. a 1 in column X1 always results in a 0 in column X17)
    x_train_bool_inverted = ~x_train_bool_without_id
    if verbose: display(x_train_bool_inverted.head())
    if export: x_train_bool_inverted.to_csv(output_dir+"x_train_boolean_inverted_only.csv")

    inverted_duplicate_columns = []
    # iterate through all "normal" bool columns
    for normal_column in x_train_bool:
        # compare to each inverted column
        # if identical inverted is found remember column header
        if normal_column not in inverted_duplicate_columns:
            for inverted_column in x_train_bool_inverted:
                if x_train_bool[normal_column].equals(x_train_bool_inverted[inverted_column]):
                    if inverted_column not in inverted_duplicate_columns:
                        inverted_duplicate_columns.append(inverted_column)

    if verbose: print(f"Inverted duplicates: {inverted_duplicate_columns}")
    # TODO: CHECK IF THIS REALLY IS CORRECT!!

    # drop newly found duplicates
    x_train = x_train.drop(inverted_duplicate_columns, axis=1)
    if export: x_train.to_csv(output_dir+"x_train_after_inversion_cleanup.csv")

    # perform one-hot-encoding
    # first get column list relevant for one hot encoding
    non_numeric_columns_without_unique = non_numeric_columns
    for col_to_remove in unique_cols:
        if col_to_remove in non_numeric_columns: non_numeric_columns_without_unique = non_numeric_columns_without_unique.remove(col_to_remove)

    if verbose: print(f"columns for one-hot-encoding: {non_numeric_columns_without_unique}")
    # perform one hot encoding
    x_train = pd.get_dummies(x_train, columns=non_numeric_columns_without_unique)
    if export: x_train.to_csv(output_dir+"x_train_after_one_hot_encoding.csv")

    # return cleaned up data
    return x_train
