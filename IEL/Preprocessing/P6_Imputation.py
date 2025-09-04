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
# Overview: Real-world datasets often contain missing values. While many machine-learning or deep-learning algorithms
# cannot be run with datasets involving missing values, data imputation is required to make those algorithms work.
#
# Data missingness comes in two different kinds: missingness at random (MAR) and missingness not at random (MNAR). For
# MAR, the probability of the missing value is related to other observed variables in the dataset but not the missing
# value itself. For MNAR, the missingness is related to the missing data itself (e.g., subjects refused to provide
# the data).
#
# Non-negative matrix factorization (NNMF) is proven to be a robust imputation method that works for both kinds of
# missingness, and will be implemented in this script. However, the off-the-shelf implementation from sklearn is often
# slow (with the multiplicative update solver). This script provides a PyTorch-empowered version of NNMF that
# drastically speed up the imputation process.
#
# On the other hand, Multiple Imputation by Chained Equations (MICE) is a widely adopted imputation method in the
# literature. It performs round-robin predictions to impute missing values. However, there are 3 main shortcomings.
# (a) MICE is slow because each iteration means p rounds of modeling (where p = number of features). So, users are
# suggested to use a small number of iterations (e.g., 10).
# (b) MICE cannot handle large data matrix. For example, MICE easily run into out-of-memory (OOM) issue when imputing
# >1000 features. While some may suggest using feature-batching to solve OOM problems, this violates the feature
# dependency required by MICE.
# (c) MICE assumes MAR theoretically. So, use it with caution when you are not certain if MNAR is involved in your data.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4096"
import pandas as pd
import torch
from copy import deepcopy
from sklearn.decomposition import NMF
from sklearn.experimental import enable_iterative_imputer       # Needed by sklearn implicitly
from sklearn.impute import IterativeImputer
from time import time
from typing import List, Literal, Optional, Union


########################################################################################################################
# Define a function for data imputation
########################################################################################################################


def impute_nnmf(df: pd.DataFrame,
                exclude_cols: Optional[List] = None,
                round_cols: Optional[List] = None,
                n_components: Optional[Union[int, Literal['auto']]] = None,
                tol: Optional[float] = 0.0001,
                max_iter: int = 200,
                random_state: Optional[int] = 42,
                method: Literal['sklearn', 'pytorch'] = 'pytorch',
                verbose: Literal[0, 1] = 1):
    """
    Perform non-negative matrix factorization to impute missing data.
    :param df: A pandas.DataFrame.
           The feature dataset to be imputed.
    :param exclude_cols: A list of strings or None.
           The list of column names in df that will not be considered in the imputation process.
           Default setting: None
    :param round_cols: A list of strings or None.
           The list of columns names in df that will be rounded to the nearest integer after scaling.
           Default setting: None
    :param n_components: An integer, 'auto', or None.
           The number of components in the embedded latent feature space. Use 'auto' only when method == 'sklearn'. If
           set as None, the embedding space has the same dimension as the number of variables to be imputed.
           Default setting: None
    :param tol: A float or None.
           The tolerance level of the stopping condition.
           Default setting: 0.0001
    :param max_iter: An integer.
           The maximum number of iterations to run.
           Default setting: 200
    :param random_state: An integer.
           The random state used for initialization of the embedded latent feature space.
           Default setting: 42
    :param method: A string in ['sklearn', 'pytorch'].
           Use 'sklearn' for the standard implementation available in sklearn, and 'pytorch' for our optimized version.
           Also, users are suggested to set a larger value for max_iter when using 'pytorch'.
           Default setting: 'pytorch'
    :param verbose: An integer in [0, 1].
           Default setting: 1
    :return:
    A pandas.DataFrame modified from df where variables have been imputed by NNMF.
    """

    ####################################################################################################################
    # Type and value check
    ####################################################################################################################
    assert isinstance(df, pd.DataFrame), f'df must be a pandas.DataFrame.'
    if exclude_cols is None:
        exclude_cols = []
    else:
        assert isinstance(exclude_cols, list), 'exclude_cols must be a list'
    if round_cols is None:
        round_cols = []
    else:
        assert isinstance(round_cols, list), 'round_cols must be a list'
    if n_components is not None:
        assert n_components == 'auto' or isinstance(n_components, int), 'n_components must be an integer or "auto".'
        if isinstance(n_components, int):
            assert n_components >= 1, f'n_components, when set as an integer, must be positive.'
    if tol is not None:
        assert isinstance(tol, float), f'tol, if not None, must be a float.'
        assert tol > 0, f'tol, if not None, must be a positive float.'
    assert isinstance(max_iter, int), f'max_iter must be an integer.'
    assert max_iter > 0, f'max_iter must be a positive integer.'
    assert isinstance(random_state, int), f'random_state must be an integer'
    assert method in ['sklearn', 'pytorch'], f'method must be in ["sklearn", "pytorch"]'
    if method in 'pytorch':
        assert n_components != 'auto', f'n_components="auto" is only implemented for method="sklearn"'
    assert verbose in [0, 1], f'verbose must be in [0, 1].'

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
    # Step 0. Prepare dataset for imputation and initialize imputation with mean values
    ####################################################################################################################
    feats_to_impute = [feat for feat in df.columns if feat not in exclude_cols]
    if n_components is None:
        n_components = len(feats_to_impute)
    df_feat = df[feats_to_impute]
    if verbose:
        print(f'Number of NaNs before NNMF ({method}): {df_feat.isnull().sum().sum()}', flush=True)
    feat_copy = deepcopy(df_feat)
    mean_data = np.nanmean(feat_copy, axis=0)
    for col_idx, col in enumerate(feat_copy.columns):
        feat_copy[col] = feat_copy[col].fillna(mean_data[col_idx])
    feat_copy_np = feat_copy.to_numpy()

    ####################################################################################################################
    # Step 1A. Perform NNMF imputation with method=='sklearn'
    ####################################################################################################################
    if method == 'sklearn':
        M = NMF(solver='mu', max_iter=max_iter, l1_ratio=0, verbose=verbose, init='random', random_state=random_state,
                n_components=n_components)
        W = M.fit_transform(feat_copy_np)
        H = M.components_
        nmf_output = pd.DataFrame(np.dot(W, H), columns=feats_to_impute)
        df_feat = df_feat.mask(df_feat.isnull(), nmf_output)

    ####################################################################################################################
    # Step 1B. Perform NNMF imputation with method=='pytorch'
    ####################################################################################################################
    else:
        device = 'cuda:0'
        dtype = torch.float32  # May need to change to torch.float16 if having out-of-memory issues
        torch.cuda.empty_cache()
        torch.manual_seed(random_state)
        feat_copy_torch = torch.tensor(feat_copy_np, dtype=dtype).to(device)
        n, m = feat_copy_torch.shape
        W = torch.rand(n, n_components, dtype=dtype, device=device)
        H = torch.rand(n_components, m, dtype=dtype, device=device)
        epsilon = 1e-5  # Avoid division-by-zero error

        ################################################################################################################
        # Update the latent components recursively
        ################################################################################################################
        t0 = time()
        with torch.no_grad():
            for round_i in range(max_iter):
                H_numerator = W.T @ feat_copy_torch
                H_denominator = (W.T @ W @ H) + epsilon
                H = H * (H_numerator / H_denominator)
                H = torch.clamp(H, min=epsilon)

                W_numerator = feat_copy_torch @ H.T
                W_denominator = (W @ H @ H.T) + epsilon
                W = W * (W_numerator / W_denominator)
                W = torch.clamp(W, min=epsilon)

                # Compute Frobenius loss
                frob_loss = torch.norm(feat_copy_torch - W @ H, p='fro').item() ** 2
                frob_per_sample = frob_loss / feat_copy_torch.shape[0]
                if tol is not None:
                    if frob_per_sample < tol:
                        t1 = time()
                        print(f"[Iteration {round_i + 1}] Elapsed time: {t1 - t0:.2f}s; Per-sample Frobenius loss: "
                              f"{frob_per_sample:.2f} < tol.", flush=True)
                        break
                if verbose and (round_i + 1) % 10 == 0:
                    t1 = time()
                    print(f"[Iteration {round_i + 1}] Elapsed time: {t1 - t0:.2f}s; Per-sample Frobenius loss: "
                          f"{frob_per_sample:.2f}", flush=True)

        WH = torch.matmul(W, H)
        torch.cuda.empty_cache()
        nan_mask = torch.isnan(torch.tensor(df_feat.to_numpy(), dtype=dtype, device=WH.device))
        orig_tensor = torch.tensor(df_feat.to_numpy(), dtype=WH.dtype, device=WH.device)
        imputed_tensor = torch.where(nan_mask, WH, orig_tensor)
        df_feat = pd.DataFrame(imputed_tensor.cpu().numpy(), columns=feats_to_impute)

    if verbose:
        print(f"Number of NaNs after NNMF ({method}): {df_feat.isnull().sum().sum()}", flush=True)

    ####################################################################################################################
    # Step 3. Round the specified variables if needed
    ####################################################################################################################
    df_feat[round_cols] = df_feat[round_cols].round(0).astype('Int64')

    ####################################################################################################################
    # Step 4. Return the imputed dataset
    ####################################################################################################################
    df = deepcopy(df)
    df.loc[:, feats_to_impute] = df_feat.values
    return df


########################################################################################################################
# Define the function for MICE imputation
########################################################################################################################


def impute_mice(df: pd.DataFrame,
                exclude_cols: Optional[List] = None,
                round_cols: Optional[List] = None,
                max_iter: int = 10,
                tol: float = 0.001,
                random_state: Optional[int] = 42,
                verbose: Literal[0, 1] = 1):
    """
    Perform non-negative matrix factorization to impute missing data.
    :param df: A pandas.DataFrame.
           The feature dataset to be imputed.
    :param exclude_cols: A list of strings or None.
           The list of column names in df that will not be considered in the imputation process.
           Default setting: None
    :param round_cols: A list of strings or None.
           The list of columns names in df that will be rounded to the nearest integer after scaling.
           Default setting: None
    :param max_iter: An integer.
           The maximum number of iterations to run.
           Default setting: 10
    :param tol: A float.
           The tolerance level of the stopping condition.
           Default setting: 0.001
    :param random_state: An integer.
           The random state used for initialization of the embedded latent feature space.
           Default setting: 42
    :param verbose: An integer in [0, 1].
           Default setting: 1
    :return:
    A pandas.DataFrame modified from df where variables have been imputed by MICE.
    """

    ####################################################################################################################
    # Type and value check
    ####################################################################################################################
    assert isinstance(df, pd.DataFrame), f'df must be a pandas.DataFrame.'
    if exclude_cols is None:
        exclude_cols = []
    else:
        assert isinstance(exclude_cols, list), 'exclude_cols must be a list'
    if round_cols is None:
        round_cols = []
    else:
        assert isinstance(round_cols, list), 'round_cols must be a list'
    assert isinstance(max_iter, int), f'max_iter must be an integer.'
    assert max_iter > 0, f'max_iter must be a positive integer.'
    assert isinstance(tol, float), f'tol, if not None, must be a float.'
    assert tol > 0, f'tol, if not None, must be a positive float.'
    assert isinstance(random_state, int), f'random_state must be an integer'
    assert verbose in [0, 1], f'verbose must be in [0, 1].'

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
    # Step 0. Prepare dataset for imputation and initialize imputation with mean values
    ####################################################################################################################
    feats_to_impute = [feat for feat in df.columns if feat not in exclude_cols]
    df_feat = df[feats_to_impute]
    if verbose:
        print(f'Number of NaNs before MICE: {df_feat.isnull().sum().sum()}', flush=True)
    feat_copy = df_feat.to_numpy()

    ####################################################################################################################
    # Step 1. Perform MICE imputation
    ####################################################################################################################
    M = IterativeImputer(max_iter=max_iter, sample_posterior=True, initial_strategy='mean', random_state=random_state,
                         verbose=verbose)
    mice_output = M.fit_transform(feat_copy)
    df_feat = df_feat.mask(df_feat.isnull(), mice_output)
    if verbose:
        print(f"Number of NaNs after MICE: {df_feat.isnull().sum().sum()}", flush=True)

    ####################################################################################################################
    # Step 3. Round the specified variables if needed
    ####################################################################################################################
    df_feat[round_cols] = df_feat[round_cols].round(0).astype('Int64')

    ####################################################################################################################
    # Step 4. Return the imputed dataset
    ####################################################################################################################
    df = deepcopy(df)
    df.loc[:, feats_to_impute] = df_feat.values
    return df


########################################################################################################################
# Test run
########################################################################################################################
if __name__ == '__main__':

    np.random.seed(42)

    # Specify the sample size of the dataset
    n_ = 1000

    # Simulate the toy dataset with missing values
    cont_data = np.random.rand(n_, 300)                                        # 300 continuous variables
    cont_data[np.random.rand(*cont_data.shape) < 0.1] = np.nan
    bin_data = np.random.randint(0, 2, size=(n_, 100)).astype(float)           # 100 binary variables
    bin_data[np.random.choice(n_, size=int(n_ * 0.2), replace=False), :] = np.nan
    cont_colnames = [f"C{i}" for i in range(cont_data.shape[1])]
    bin_colnames = [f"B{i}" for i in range(bin_data.shape[1])]
    df_ = pd.DataFrame(np.concatenate([cont_data, bin_data], axis=1), columns=cont_colnames + bin_colnames)

    # Insert an extra identifier column
    df_.insert(0, 'ID', [f'ID_{i}' for i in range(n_)])

    # Specify the imputation method you want to run
    impute_method: str = ['nnmf_sklearn', 'nnmf_pytorch', 'mice'][1]

    # Perform data imputation
    t0_ = time()
    if impute_method.startswith('nnmf'):
        imputed_df = impute_nnmf(df=df_,
                                 exclude_cols=['ID'],
                                 round_cols=bin_colnames,
                                 n_components=None,
                                 tol=None,
                                 max_iter=200,
                                 random_state=42,
                                 method=impute_method.split('_')[-1],
                                 verbose=1)
    else:
        imputed_df = impute_mice(df=df_,
                                 exclude_cols=['ID'],
                                 round_cols=bin_colnames,
                                 max_iter=3,
                                 random_state=42,
                                 verbose=1)
    t1_ = time()
    print(f'Elapsed time of imputation: {t1_ - t0_:.1f} seconds.')

    ####################################################################################################################
    # Runtime comparison
    ####################################################################################################################
    # Machine used: Lambda Tensorbook laptop
    # • 11th Gen Intel® Core™ i7-11800H @ 2.30GHz 8-core CPU
    # • 64GB DDR4 memory
    # • NVIDIA RTX 3080 Mobile Max-Q 16GB

    # Dimension of simulated dataset: (1000 samples, 300 features)
    # NNMF (pytorch implementation) with 200 iterations     ->       0.4 seconds
    # NNMF (sklearn implementation) with 200 iterations     ->       2.7 seconds
    # MICE with 3 iterations                                ->      53.0 seconds
