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
# Overview: Features with varying ranges of values can adversely affect learning, leading to vanishing or exploding
# gradients when applying deep-learning methods. This script provides a simple implementation of the min-max
# normalization such that feature values are scaled (to the unit interval [0, 1]) to facilitate learning.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional

########################################################################################################################
# Define the function for min-max normalization
########################################################################################################################


def minMaxScale(df: pd.DataFrame,
                var_range: tuple = (0, 1),
                exclude_cols: Optional[List] = None,
                round_cols: Optional[List] = None):
    """
    :param df: A pandas.DataFrame.
           The (feature) dataset to be scaled.
    :param var_range: A length-2 tuple.
           The lower and upper bounds of the scaled values.
           Default setting: (0, 1)
    :param exclude_cols: A list of strings or None.
           The list of column names in df that will not be considered in the scaling process.
           Default setting: None
    :param round_cols: A list of strings or None.
           The list of columns names in df that will be rounded to the nearest integer after scaling. This helps avoid
           floating point errors that may occur during rounding.
           Default setting: None
    :return:
    A pandas.DataFrame modified from df where variables have been scaled.
    """

    ####################################################################################################################
    # Type and value check
    ####################################################################################################################
    assert isinstance(df, pd.DataFrame), f'df must be a pandas.DataFrame.'
    assert isinstance(var_range, tuple), f'var_range must be a length-2 tuple.'
    assert len(var_range) == 2, f'var_range must be a length-2 tuple.'
    assert var_range[0] < var_range[1], f'The 1st element of var_range must be strictly smaller than the 2nd element.'
    if exclude_cols is None:
        exclude_cols = []
    else:
        assert isinstance(exclude_cols, list), 'exclude_cols must be a list'
    if round_cols is None:
        round_cols = []
    else:
        assert isinstance(round_cols, list), 'round_cols must be a list'

    ####################################################################################################################
    # Warn users of the non-existent columns and update exclude_cols and round_cols
    ####################################################################################################################
    exclude_cols_non_exist = set(exclude_cols) - set(df.columns)
    if len(exclude_cols_non_exist) > 0:
        print(f'{len(exclude_cols_non_exist)} columns in exclude_cols are not in df. You may want to verify the '
              f'columns in exclude_cols and run this function again.')
    exclude_cols = [feat for feat in exclude_cols if feat in df.columns]

    round_cols_non_exist = set(round_cols) - set(df.columns)
    if len(round_cols_non_exist) > 0:
        print(f'{len(round_cols_non_exist)} columns in round_cols are not in df. You may want to verify the '
              f'columns in round_cols and run this function again.')
    round_cols = [feat for feat in round_cols if feat in df.columns]

    exclude_round_intersect_cols = set(round_cols).intersection(exclude_cols)
    if len(exclude_round_intersect_cols) > 0:
        print(f'{len(exclude_round_intersect_cols)} columns exist in both exclude_cols and round_cols. They will not'
              f'be scaled.')
    round_cols = [feat for feat in round_cols if feat not in exclude_cols]

    ####################################################################################################################
    # Step 0. Make a copy of df to avoid Pandas inplace revision
    ####################################################################################################################
    df = deepcopy(df)  # Make a copy to avoid Pandas inplace revision

    ####################################################################################################################
    # Step 1. Identify the columns to be scaled
    ####################################################################################################################
    cols_to_scale = [col for col in df.columns if col not in exclude_cols]

    ####################################################################################################################
    # Step 2. Fit with MinMaxScaler
    ####################################################################################################################
    scaler = MinMaxScaler(feature_range=var_range)
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    ####################################################################################################################
    # Step 3. Round the specified variables if needed
    ####################################################################################################################
    df[round_cols] = df[round_cols].round(0).astype('Int64')

    ####################################################################################################################
    # Step 4. Return the scaled dataset
    ####################################################################################################################
    return df


########################################################################################################################
# Test run
########################################################################################################################
if __name__ == '__main__':

    import numpy as np
    np.random.seed(42)

    # Specify the sample size of the dataset
    n_ = 100

    # Simulate the toy continuous data
    cont_list_ = [f'C{i}' for i in range(3)]       # 3 continuous variables
    cont_data_ = np.random.rand(n_, len(cont_list_)) * 100
    df_ = pd.DataFrame(cont_data_, columns=cont_list_)

    # Simulate another 2 binary variables which have values in [0, 1]
    for i in range(2):
        df_[f'B{i}'] = np.random.randint(0, 2, size=n_)

    # Simulate another 2 binary variables which do not have values in [0, 1]
    for i in range(2):
        df_[f'B{i+2}'] = np.random.choice([1.0, 2.0], size=n_)

    # Simulate 2 extra columns which will not be winsorized
    df_.insert(0, 'ID1', [f'ID{i}' for i in range(n_)])
    df_.insert(1, 'ID2', np.random.choice(range(10), size=n_))

    df_new = minMaxScale(df=df_,
                         var_range=(0, 1),
                         exclude_cols=['ID1', 'ID2'],
                         round_cols=[col for col in df_.columns if col.startswith('B')])
    print(f'Data (before scaling):\n{df_}')
    print('*'*120)
    print(f'Data (after scaling):\n{df_new}')
