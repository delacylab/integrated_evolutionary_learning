########################################################################################################################
# Apache License 2.0
########################################################################################################################
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2025 Nina de Lacy
########################################################################################################################

########################################################################################################################
# Overview: Features with too many missing values (i.e., NaN) tend to under-represent the population distribution and
# are not reliable predictors even after imputation. This script provides a simple function to remove those features
# by a user-specified threshold.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import pandas as pd
from typing import List, Optional

########################################################################################################################
# Define the function for removing features with more than a user-specified percentage of missing values
########################################################################################################################


def remove_null(df: pd.DataFrame,
                nan_threshold: float,
                exclude_cols: Optional[List] = None):
    """
    Remove columns with the percentage of nulls greater than the pre-specified threshold.
    :param df: A pandas.DataFrame
           The feature dataset to be cleaned.
    :param nan_threshold: A float in the closed unit-interval [0, 1].
           The percentage of samples required to preserve a column. In other words, a columns will be removed if it
           has less than nan_threshold * df.shape[0] samples.
    :param exclude_cols: A list of strings or None.
           The list of column names in df that will not be considered in the cleaning process.
           Default setting: None
    :return:
    A pandas.DataFrame modified from df where columns (not in exclude_cols) with >nan_threshold missingness are removed.
    """

    ####################################################################################################################
    # Type and value check
    ####################################################################################################################
    if exclude_cols is None:
        exclude_cols = []
    else:
        assert isinstance(exclude_cols, list), 'exclude_cols must be a list'
    assert isinstance(df, pd.DataFrame), 'df must be a pandas.DataFrame.'
    try:
        nan_threshold = float(nan_threshold)
    except TypeError:
        raise TypeError('nan_threshold must be (convertible to) a float.')
    assert 0 <= nan_threshold <= 1, f'nan_threshold must be in the closed unit-interval [0, 1].'

    ####################################################################################################################
    # Warn users of the non-existing columns in exclude_cols
    ####################################################################################################################
    if exclude_cols is not None:
        exclude_cols_non_exist = set(exclude_cols) - set(df.columns)
        if len(exclude_cols_non_exist) > 0:
            print(f'{len(exclude_cols_non_exist)} columns in exclude_cols are not in df. You may want to verify the '
                  f'columns in exclude_cols and run this function again.')
    else:
        exclude_cols = []

    ####################################################################################################################
    # Identify the columns to be removed
    ####################################################################################################################
    cols_to_remove = [col for col in df.columns
                      if (col not in exclude_cols)
                      and (df[col].isna().sum() >= nan_threshold * df.shape[0])]
    print(f"Number of features to be removed (>{nan_threshold * 100}% null): {len(cols_to_remove)}", flush=True)
    return df.drop(columns=cols_to_remove)


########################################################################################################################
# Test run
########################################################################################################################
if __name__ == '__main__':

    import numpy as np
    np.random.seed(42)

    # Simulate a toy dataset with missing values
    X = np.random.rand(100, 500)
    nan_percents = np.random.uniform(low=0, high=0.5, size=X.shape[1])
    for col_ in range(X.shape[1]):
        percent = nan_percents[col_]
        n_missing = int(np.floor(percent * X.shape[0]))
        if n_missing > 0:
            missing_indices = np.random.choice(X.shape[0], n_missing, replace=False)
            X[missing_indices, col_] = np.nan
    df_X = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])
    df_X.insert(0, 'ID', [f'ID_{i}' for i in range(X.shape[0])])

    # Perform Nan thresholding
    df_X_new = remove_null(df_X, 0.3, exclude_cols=['ID', 'Dummy'])
    print(f'Dimension of the cleaned dataset: {df_X_new.shape}', flush=True)
