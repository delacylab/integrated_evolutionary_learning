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
# Overview: Target labels with imbalance samples tend to be harder to learn in classification tasks. Sample balancing
# (over the target labels) address the problem by ensuring that the negative labels and positive labels are of the same
# length. Notice that the function below is defined only for binary labels. A fuller version for multiclass
# classification will be provided in the future.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import pandas as pd
from typing import Optional

########################################################################################################################
# Define the function for sample balancing for binary classification tasks
########################################################################################################################


def sample_balancing(df: pd.DataFrame,
                     target_col: str,
                     match_col: Optional[str] = None,
                     tie_break_col: Optional[str] = None,
                     maximize_tie_break: bool = True):
    """
    Balance the binary labels in the target dataset df by matching positive and negative cases. Additional matching
    with respect to a certain background (e.g., sex and discretized age) can be done as well. A tie-breaking mechanism
    is embedded to determine which samples from the majority class are selected.
    :param df: A pandas.DataFrame.
           The dataset that includes the binary target to be balanced.
    :param target_col: A string.
           The name of the column in df that encodes the binary target.
    :param match_col: A string or None.
           The name of the column in df that encodes the column for matching.
           Default setting: None
    :param tie_break_col: A string or None.
           The name of the column in df that encodes the tie-breaker for samples from the majority class.
           Default setting: None
    :param maximize_tie_break: boolean.
           Used only when tie_break_col is not None. Whether large values in tie_break_col are chosen first for the
           majority class.
           Default setting: True
    :return:
    A pandas.DataFrame modified from df where samples of target_col have been balanced.
    """

    ####################################################################################################################
    # Type and value check
    ####################################################################################################################
    assert isinstance(df, pd.DataFrame), 'df must a a pandas.DataFrame.'
    assert isinstance(target_col, str), 'target_col must be a string.'
    assert target_col in df.columns, 'target_col must be a column in df.'
    if match_col is not None:
        assert isinstance(match_col, str), 'match_col must be a string.'
        assert match_col in df.columns, 'match_col must be a column in df.'
    if tie_break_col is not None:
        assert isinstance(tie_break_col, str), 'tie_break_col must be a string.'
        assert tie_break_col in df.columns, 'tie_break_col must be a column in df.'
    assert isinstance(maximize_tie_break, bool), 'maximize_tie_break must be a boolean.'
    assert df[target_col].dropna().nunique() == 2, 'Only binary labels are supported by this function.'
    assert set(df[target_col].unique()) == {0, 1}, ('Only binary labels 0/1 are supported by this function. Replace the'
                                                    'labels before using the function.')

    ####################################################################################################################
    # Warn users of the missing values in target_col
    ####################################################################################################################
    if df[target_col].isna().sum() > 0:
        df = df[df[target_col].notna()]
        print('target_col contains missing values. Only samples with non-missing values in target_col are considered.')

    ####################################################################################################################
    # Step 0. Create a dummy identifier column for convenience
    ####################################################################################################################
    df['ID_Dummy'] = df.index

    ####################################################################################################################
    # Step 1. Extract the positive and negative cases
    ####################################################################################################################
    df_pos, df_neg = df[df[target_col] == 0], df[df[target_col] == 1]

    ####################################################################################################################
    # Step 2. Loop over each minority sample to identify a matching majority sample
    ####################################################################################################################
    df_major = df_pos if df_pos.shape[0] >= df_neg.shape[0] else df_neg
    df_minor = df_pos if df_pos.shape[0] < df_neg.shape[0] else df_neg
    ID_major_list, ID_minor_list = [], []

    for i, minor in df_minor.iterrows():
        df_major_sub = df_major[df_major[match_col] == minor[match_col]] if match_col is not None \
            else df_major.copy(deep=True)
        if tie_break_col is not None:
            df_major_sub = df_major_sub.sort_values(by=tie_break_col,
                                                    ascending=not maximize_tie_break).reset_index(drop=True)
        if df_major_sub.shape[0] == 0:
            continue                    # No sample from the majority class can be matched. Skipping.
        id_minor = minor['ID_Dummy']
        id_major = df_major_sub.iloc[0]['ID_Dummy']
        ID_minor_list.append(id_minor)
        ID_major_list.append(id_major)
        df_major = df_major[df_major['ID_Dummy'] != id_major]     # Remove the matched subject from consideration

    ####################################################################################################################
    # Step 3. Return the balanced (and matched) dataset
    ####################################################################################################################
    assert len(ID_major_list) == len(ID_minor_list)
    return df[df['ID_Dummy'].isin(ID_minor_list+ID_major_list)].drop(columns='ID_Dummy').reset_index(drop=True)


########################################################################################################################
# Test run
########################################################################################################################
if __name__ == '__main__':

    import numpy as np
    np.random.seed(42)

    # Specify the number of subjects
    n_ = 100

    # Simulate a toy target dataset with the specified number of subjects
    y_ = pd.DataFrame({'ID': [f'ID_{i}' for i in range(n_)],
                       'target': np.array([1] * 30 + [0] * 70),
                       'sex': np.random.choice(['M', 'F'], size=n_),
                       'age': np.random.choice([10, 11, 12], size=n_),
                       'target_confidence': np.random.rand(n_)})
    print(f'Target distribution (before balancing): {y_.target.value_counts().to_dict()}')

    # In addition to balancing the binary label 'target', suppose we want to match the subjects with 'age' and 'sex'.
    # To do so, we start with creating an extra column combining 'sex' and 'age' for matching.
    y_['sex_age'] = list(zip(y_['sex'], y_['age']))

    # Perform sample balancing with matching
    y_new = sample_balancing(df=y_,
                             target_col='target',
                             match_col='sex_age',
                             tie_break_col='target_confidence',
                             maximize_tie_break=True)     # Maximize our confidence of the target when subsampling
    print(f'Target distribution (after balancing): {y_new.target.value_counts().to_dict()}')

    # Examine the results of matching
    y_new.drop(columns='sex_age', inplace=True)     # Not needed anymore after matching
    summary = y_new[['sex', 'age', 'target']].value_counts().reset_index(name='count')
    summary = summary.sort_values(by=['sex', 'age', 'target'], ascending=[True, True, False]).reset_index(drop=True)
    print(f'Matching results:\n{summary}')
