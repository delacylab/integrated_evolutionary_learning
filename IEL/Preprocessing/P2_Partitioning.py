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
# Overview: Partition a time-series dataset (separated into multiple datasets by timestamps) into training and test
# partitions for subsequent model fitting and evaluation. This script can be easily adopted for cross-sectional analysis
# with a single timestamp.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Literal, Optional, Union

########################################################################################################################
# Define the function for partitioning time-series datasets
########################################################################################################################


def partitioning(X_dict: dict[Union[str, int, float], pd.DataFrame],
                 y: pd.DataFrame,
                 train_ratio: float,
                 identifier_col: Optional[str] = None,
                 stratify_col: Optional[str] = None,
                 X_subject_include: Optional[Literal['union', 'intersect']] = 'union',
                 random_state: Optional[Union[int, np.random.Generator]] = None):
    """
    Partition the time-series data into training and test data partitions.
    :param X_dict: A dictionary with keys as the timestamp identifiers (in string/integer/float) and values as
                   the dataset of the corresponding timestamp (in pandas.DataFrame). If your feature dataset is not
                   time-series, make it a value of X_dict with an arbitrary key.
    :param y: A pandas.DataFrame.
           The target dataframe.
    :param train_ratio: A float in [0, 1].
           The proportion of the training set to the full set.
    :param identifier_col: A string or None.
           The name of the identifier column in each dataset in X_dict's values and y.
           Default setting: None
    :param stratify_col: A string or None.
           Stratify y with respect to the stratify_col column in y if not None.
           Default setting: None
    :param X_subject_include: A string in ['union', 'intersect'] or None.
           Used only when identifier_col is not None. Determine whether the intersection or the union of the subjects
           across different timestamps in X_dict is considered.
           Default setting: 'union'
    :param random_state: An integer or a random state instance.
           Random state controlling the partitioning procedure.
           Default setting: None
    :return:
    (a) A dictionary with keys as the keys in X_dict concatenated with the suffix '_train' and values as the
        pandas.DataFrame associated with the feature set in the training partition of the corresponding timestamp.
    (b) A dictionary with keys as the keys in X_dict concatenated with the suffix '_test' and values as the
        pandas.DataFrame associated with the feature set in the test partition of the corresponding timestamp.
    (c) A pandas.DataFrame of the target in the training partition.
    (d) A pandas.DataFrame of the target in the test partition.
    """

    ####################################################################################################################
    # Type and value check
    ####################################################################################################################
    assert isinstance(X_dict, dict), 'X_dict must be a dictionary.'
    for k, v in X_dict.items():
        assert isinstance(k, str) or isinstance(k, int) or isinstance(k, float), \
            'Each key in X_dict must be a string or integer.'
        assert isinstance(v, pd.DataFrame), 'Each value in X_dict must be a pandas.DataFrame.'
        if identifier_col is not None:
            assert identifier_col in v.columns, "identifier_col must be a column in each X_dict's value."
    assert isinstance(y, pd.DataFrame), 'y must be a pandas.DataFrame.'
    assert 0 < train_ratio < 1, 'train_ratio must be in the interval [0, 1].'
    if identifier_col is not None:
        assert isinstance(identifier_col, str), 'identifier_col must be a string.'
        assert identifier_col in y.columns, 'identifier_col (if not None) must be a column in y.'
        assert X_subject_include in ['union', 'intersect'], ('X_subject_include must be in ["union", "intersect"] '
                                                             'when identifier_col is not None.')
    if stratify_col is not None:
        assert isinstance(stratify_col, str), 'stratify_col (if not None) must be a string.'
        assert stratify_col in y.columns, 'stratify_col (if not None) must be a column in y.'

    ####################################################################################################################
    # Check for dimensionality
    ####################################################################################################################
    if identifier_col is None:
        n = y.shape[0]
        for k, df in X_dict.items():
            assert df.shape[0] == n, (f'Sample size of the feature dataset X in timestamp {k} does not match that with '
                                      f'the target dataset y (n={n}).')

    ####################################################################################################################
    # Obtain the intersecting set of subjects when identifier_col is not None
    ####################################################################################################################
    if identifier_col is not None:
        y = y.drop_duplicates(subset=identifier_col, keep='first')
        X_IDs = set()
        for idx, (k, df) in enumerate(X_dict.items()):
            df = df.drop_duplicates(subset=identifier_col, keep='first')
            X_dict[k] = df
            X_subjects_cur = set(df[identifier_col])
            if idx == 0:
                X_IDs = X_subjects_cur
            elif X_subject_include == 'union':
                X_IDs = X_IDs.union(X_subjects_cur)
            else:
                X_IDs = X_IDs.intersection(X_subjects_cur)
        Xy_IDs = set(y[identifier_col]).intersection(X_IDs)
        y = y[y[identifier_col].isin(Xy_IDs)]

    ####################################################################################################################
    # Perform (stratified) partitioning over y
    ####################################################################################################################
    y_train, y_test = train_test_split(y,
                                       train_size=train_ratio,
                                       stratify=y[stratify_col] if stratify_col is not None else None,
                                       random_state=random_state)

    ####################################################################################################################
    # Perform partitioning over X and return the partitioned feature and target datasets
    ####################################################################################################################
    X_train_dict, X_test_dict = dict(), dict()
    if identifier_col is not None:
        for k, X in X_dict.items():
            X_train_dict[f'{k}_train'] = pd.merge(left=X, right=y_train[[identifier_col]], on=identifier_col,
                                                  how='right')
            X_test_dict[f'{k}_test'] = pd.merge(left=X, right=y_test[[identifier_col]], on=identifier_col,
                                                how='right')
            assert X_train_dict[f'{k}_train'][identifier_col].to_list() == y_train[identifier_col].to_list()
            assert X_test_dict[f'{k}_test'][identifier_col].to_list() == y_test[identifier_col].to_list()
    else:
        y_train_idxs, y_test_idxs = y_train.index, y_test.index
        for k, X in X_dict.items():
            X_train_dict[f'{k}_train'] = X.iloc[y_train_idxs]
            X_test_dict[f'{k}_test'] = X.iloc[y_test_idxs]
    return X_train_dict, X_test_dict, y_train, y_test


########################################################################################################################
# Test run
########################################################################################################################
if __name__ == '__main__':
    np.random.seed(42)

    # Specify the number of subjects
    n_ = 100

    # Simulate a toy target dataset with the specified number of subjects
    y_ = pd.DataFrame({'ID': [f'ID_{i}' for i in range(n_)],
                       'target': np.random.randint(0, 2, size=n_)})

    # Simulate a toy time-series feature dataset with 5 timestamps
    ts = range(5)   # Try 1 for the regular cross-sectional case
    X_dict_ = {}
    for t in ts:
        X_ = pd.DataFrame({'ID': [f'ID_{i}' for i in range(n_)],
                           'X1': np.random.rand(n_),
                           'X2': np.random.rand(n_)})

        # Randomly drop 10% rows in X_ to simulate that some subjects have no samples in a given timestamp
        drop_indices = X_.sample(n=int(0.1*n_)).index
        X_dict_[t] = X_.drop(index=drop_indices).reset_index(drop=True)

    # Perform partitioning
    results = partitioning(X_dict=X_dict_,
                           y=y_,
                           train_ratio=0.7,
                           identifier_col='ID',
                           stratify_col='target',
                           X_subject_include='intersect',    # Try 'intersect' to see the difference
                           random_state=42)
    X_train_dict_, X_test_dict_, y_train_, y_test_ = results

    # Check the dimensionality
    print(f'y_train has a dimension of {y_train_.shape}')
    print(f'y_test has a dimension of {y_test_.shape}')
    print('*'*120)
    for t in ts:
        X_train_ = X_train_dict_[f"{t}_train"]
        X_test_ = X_test_dict_[f"{t}_test"]
        n_nan_train = X_train_.drop(columns='ID').isnull().all(axis=1).sum()
        n_nan_test = X_test_.drop(columns='ID').isnull().all(axis=1).sum()
        print(f'X_train (at timestamp {t}) has a dimension of {X_train_dict_[f"{t}_train"].shape} with {n_nan_train} '
              f'subjects having all NaN values.')
        print(f'X_test (at timestamp {t}) has a dimension of {X_test_dict_[f"{t}_test"].shape} with {n_nan_test} '
              f'subjects having all NaN values.')
        print('*'*120)
