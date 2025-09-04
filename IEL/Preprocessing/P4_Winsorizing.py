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
# Overview: Outliers make learning hard. A common solution is to clip the values of features within a particular range
# such that the outliers will have a more normalized value. This is also known as winsorization in statistics. Below we
# define a function to winsorize two types of variables: continuous variables and ordinal variables. Continuous
# variables will be winsorized to s standard deviations away from the mean (where s is a user-specified parameter). On
# the other hand, ordinal variables are clipped to a list of user-specified range of values, usually obtained by the
# metadata. This avoids the case that uncleaned values (e.g., 999) will not cause of distribution shift to the expected
# distribution (e.g., with a support as [0, 1, 2]).
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Literal, Optional, Union

########################################################################################################################
# Define the function for winsorizing a dataset
########################################################################################################################


def winsorize(df: pd.DataFrame,
              cont_list: list[str],
              ord_dict: Optional[dict] = None,
              cont_std: Union[int, float] = 3,
              verbose: Literal[0, 1] = 1):
    """
    Winsorizing continuous and ordinal variables in the dataset df.
    :param df: A pandas.DataFrame.
           The (feature) dataset to be winsorized.
    :param cont_list: A list of strings.
           Names of the columns that are continuous variables in df.
    :param ord_dict: A dictionary with keys as strings and values as 2-element tuples.
           Each key in ord_dict is the column name of an ordinal variable in df. Its corresponding value is a tuple of
           the form (min, max) where min refers to the expected minimum value of the ordinal variable, and max to the
           expected maximum value of the variable.
           Default setting: None
    :param cont_std: A positive integer or float.
           The standard deviation away from a continuous variable to be winsorized.
           Default setting: 3
    :param verbose: An integer in [0, 1].
           Logging about the numbers of winsorized features and values will be provided if 1.
           Default setting: 1
    :return:
    A pandas.DataFrame modified from df where continuous and ordinal variables have been winsorized.
    """

    ####################################################################################################################
    # Type and value check
    ####################################################################################################################
    assert isinstance(df, pd.DataFrame), f'df must be a pandas.DataFrame.'
    assert isinstance(cont_list, list), f'cont_list must be a list.'
    assert [isinstance(c, str) for c in cont_list], f'Each element in cont_list must be a string.'
    if ord_dict is not None:
        assert isinstance(ord_dict, dict), f'ord_dict, if not None, must be a dictionary.'
        for k, v in ord_dict.items():
            assert isinstance(k, str), f'Every key of ord_dict (if not None) must be a string.'
            assert isinstance(v, tuple), f'Every value of ord_dict (if not None) must be a tuple.'
            assert len(v) == 2, f'Every value of ord_dict (if not None) must be a length-2 tuple.'
            assert v[0] < v[1], (f'Every value of ord_dict (if not None) must be a length-2 tuple with first element '
                                 f'smaller than the second element.')
    try:
        cont_std = float(cont_std)
    except TypeError:
        raise TypeError(f"cont_sd_away must be (convertible to) a float.")
    assert cont_std > 0, f'cont_ad_away must be positive.'
    assert verbose in [0, 1], f'verbose must be in [0, 1].'

    ####################################################################################################################
    # Warn users of the non-existent columns and update cont_list and ord_dict
    ####################################################################################################################
    cont_cols_non_exist = set(cont_list) - set(df.columns)
    if len(cont_cols_non_exist) > 0:
        print(f'{len(cont_cols_non_exist)} columns in cont_list are not in df. You may want to verify the '
              f'columns in cont_list and run this function again.')
    cont_list = [feat for feat in cont_list if feat in df.columns]

    ord_cols_non_exist = set(ord_dict.keys()) - set(df.columns)
    if len(ord_cols_non_exist) > 0:
        print(f'{len(ord_cols_non_exist)} columns in the keys of ord_dict are not in df. You may want to verify the '
              f'keys in ord_dict and run this function again.')
    ord_dict = {k: v for k, v in ord_dict.items() if k in df.columns}

    ####################################################################################################################
    # Step 0. Make a copy of df to avoid Pandas inplace revision
    ####################################################################################################################
    df = deepcopy(df)  # Make a copy to avoid Pandas inplace revision

    ####################################################################################################################
    # Step 1. Obtain a data subset of the continuous variables to winsorize to the upper bound
    ####################################################################################################################
    cont_data = df[cont_list].values
    mu, sigma = np.nanmean(cont_data, axis=0), np.nanstd(cont_data, axis=0)
    upper, lower = mu + (cont_std * sigma), mu - (cont_std * sigma)

    # Perform clipping to the upper bound
    cont_clip_upper = np.clip(cont_data, a_min=None, a_max=upper)
    if verbose:
        winsor_feat_list_0 = np.sum(np.not_equal(cont_data, cont_clip_upper), axis=0)
        winsor_feat_list_1 = np.sum(np.logical_and(np.isnan(cont_data), np.isnan(cont_clip_upper)), axis=0)
        winsor_feat_list = winsor_feat_list_0 - winsor_feat_list_1
        winsor_feat_count = np.sum(winsor_feat_list > 0)
        winsor_feat_pct = winsor_feat_count * 100 / len(winsor_feat_list)
        winsor_value_count = np.sum(winsor_feat_list)
        winsor_value_pct = winsor_value_count * 100 / np.sum(~np.isnan(cont_data))
        print(f"Continuous variables winsorized [upper]: {winsor_feat_count} ({winsor_feat_pct:.2f}%)", flush=True)
        print(f"Continuous values winsorized [upper]: {winsor_value_count} ({winsor_value_pct:.2f}%)", flush=True)

    ####################################################################################################################
    # Step 2. Winsorize the continuous variables to the lower bound
    ####################################################################################################################
    # Perform clipping to the lower bound
    cont_clip_lower = np.clip(cont_clip_upper, a_min=lower, a_max=None)
    if verbose:
        winsor_feat_list_0 = np.sum(np.not_equal(cont_clip_upper, cont_clip_lower), axis=0)
        winsor_feat_list_1 = np.sum(np.logical_and(np.isnan(cont_clip_upper), np.isnan(cont_clip_lower)), axis=0)
        winsor_feat_list = winsor_feat_list_0 - winsor_feat_list_1
        winsor_feat_count = np.sum(winsor_feat_list > 0)
        winsor_feat_pct = winsor_feat_count * 100 / len(winsor_feat_list)
        winsor_value_count = np.sum(winsor_feat_list)
        winsor_value_pct = winsor_value_count * 100 / np.sum(~np.isnan(cont_clip_upper))
        print(f"Continuous variables winsorized [lower]: {winsor_feat_count} ({winsor_feat_pct:.2f}%)", flush=True)
        print(f"Continuous values winsorized [lower]: {winsor_value_count} ({winsor_value_pct:.2f}%)", flush=True)

    ####################################################################################################################
    # Step 3: Update df
    ####################################################################################################################
    df.loc[:, cont_list] = cont_clip_lower

    ####################################################################################################################
    # Step 4: Obtain a data subset of the continuous variables to winsorize to the upper bound
    ####################################################################################################################
    ord_data = df[list(ord_dict.keys())].values
    min_values, max_values = [v[0] for v in ord_dict.values()], [v[1] for v in ord_dict.values()]

    # Perform clipping to the upper bound
    ord_clip_upper = np.clip(ord_data, a_min=None, a_max=max_values)
    if verbose:
        winsor_feat_list_0 = np.sum(np.not_equal(ord_data, ord_clip_upper), axis=0)
        winsor_feat_list_1 = np.sum(np.logical_and(np.isnan(ord_data), np.isnan(ord_clip_upper)), axis=0)
        winsor_feat_list = winsor_feat_list_0 - winsor_feat_list_1
        winsor_feat_count = np.sum(winsor_feat_list > 0)
        winsor_feat_pct = winsor_feat_count * 100 / len(winsor_feat_list)
        winsor_value_count = np.sum(winsor_feat_list)
        winsor_value_pct = winsor_value_count * 100 / np.sum(~np.isnan(ord_data))
        print(f"Ordinal variables winsorized [upper]: {winsor_feat_count} ({winsor_feat_pct:.2f}%)", flush=True)
        print(f"Ordinal values winsorized [upper]: {winsor_value_count} ({winsor_value_pct:.2f}%)", flush=True)

    ####################################################################################################################
    # Step 5. Winsorize the ordinal variables to the lower bound
    ####################################################################################################################
    # Perform clipping to the lower bound
    ord_clip_lower = np.clip(ord_clip_upper, a_min=min_values, a_max=None)
    if verbose:
        winsor_feat_list_0 = np.sum(np.not_equal(ord_clip_upper, ord_clip_lower), axis=0)
        winsor_feat_list_1 = np.sum(np.logical_and(np.isnan(ord_clip_upper), np.isnan(ord_clip_lower)), axis=0)
        winsor_feat_list = winsor_feat_list_0 - winsor_feat_list_1
        winsor_feat_count = np.sum(winsor_feat_list > 0)
        winsor_feat_pct = winsor_feat_count * 100 / len(winsor_feat_list)
        winsor_value_count = np.sum(winsor_feat_list)
        winsor_value_pct = winsor_value_count * 100 / np.sum(~np.isnan(ord_clip_upper))
        print(f"Ordinal variables winsorized [lower]: {winsor_feat_count} ({winsor_feat_pct:.2f}%)", flush=True)
        print(f"Ordinal values winsorized [lower]: {winsor_value_count} ({winsor_value_pct:.2f}%)", flush=True)

    ####################################################################################################################
    # Step 6: Return the winsorized dataset
    ####################################################################################################################
    df.loc[:, list(ord_dict.keys())] = ord_clip_lower
    return df


########################################################################################################################
# Test run
########################################################################################################################
if __name__ == '__main__':

    np.random.seed(42)

    # Specify the sample size of the dataset
    n_ = 1000

    # Specify the names of the continuous variables
    cont_list_ = [f'C{i}' for i in range(10)]    # 10 continuous variables

    # Simulate the toy continuous data
    cont_data_ = np.random.rand(n_, len(cont_list_)) * 100
    cont_data_ += np.random.normal(0, 100, cont_data_.shape)
    df_ = pd.DataFrame(cont_data_, columns=cont_list_)

    # Specify the ordinal variables and their expected range of values
    ord_dict_ = {'O0': (2, 9),
                 'O1': (0, 8),
                 'O2': (2, 7),
                 'O3': (3, 9),
                 'O4': (1, 7)}   # 5 ordinal variables, with user-specified expected range of values

    # Simulate the toy ordinal data
    for i in range(len(ord_dict_)):
        bins = [-np.inf] + list(range(10, 91, 10)) + [np.inf]
        labels = range(1, len(bins))
        df_[f'O{i}'] = pd.cut(np.random.rand(n_) * 100, bins=bins, labels=labels).codes.astype(int)

    # Simulate some extra columns which will not be winsorized
    for i in range(20):
        df_[f'B{i}'] = np.random.randint(0, 2, size=n_)
    df_.insert(0, 'ID', [f'ID_{i}' for i in range(n_)])

    # Perform winsorization
    df_new = winsorize(df=df_,
                       cont_list=cont_list_,
                       ord_dict=ord_dict_,
                       cont_std=3,
                       verbose=1)
