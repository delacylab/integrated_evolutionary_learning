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
# Overview: This script codes the Integrated Evolutionary Learning (IEL) algorithm.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import gc
import numpy as np
import pandas as pd
import torch
from time import time
from .ann_embedded import ANN_Classifier_Embedded
from .ann_base import EarlyStopping
from typing import Literal, Union

########################################################################################################################
# Define a helper function to sample hyperparameters from a given range
########################################################################################################################


def sample_hp(low: float,
              hi: float,
              size: int):
    """
    :param low: A float or an integer.
           The lower bound of the hyperparameter range to be sampled from.
    :param hi: A float or an integer.
           The upper bound of the hyperparameter range to be sampled from.
    :param size: An integer.
           The number of hyperparameters to be sampled.
    :return: A list of floats.
    """
    # Type and value check
    assert isinstance(low, (float, int)), 'low must be a float or an integer.'
    assert isinstance(hi, (float, int)), 'hi must be a float or an integer.'
    assert low < hi, 'low must be smaller than hi.'
    assert isinstance(size, int), 'size must be an integer.'
    assert size >= 1, 'size must be a positive integer.'

    return list(np.random.uniform(low=low, high=hi, size=size))

########################################################################################################################
# Define a helper function to sample indices of features.
########################################################################################################################


def sample_feat_idx(n_feat: int,
                    low: int,
                    hi: int,
                    size: int):
    """
    :param n_feat: A positive integer.
           The total number of features.
    :param low: A positive integer.
           The lower bound of the number of features to be sampled from.
    :param hi: A positive integer.
           The upper bound of the number of features to be sampled from.
    :param size: A positive integer.
           The number of feature subsets to be sampled.
    :return: A list of lists of feature indices.
    """
    # Type and value check
    assert isinstance(n_feat, int), 'n_feat must be an integer.'
    assert n_feat >= 1, 'n_feat must be a positive integer.'
    assert isinstance(low, int), 'low must be an integer.'
    assert isinstance(hi, int), 'hi must be an integer.'
    assert low < hi <= n_feat, 'low must be smaller than hi, which is smaller than or equal to n_feat.'
    assert isinstance(size, int), 'size must be an integer.'
    assert size >= 1, 'size must be a positive integer.'

    return [sorted((np.random.choice(range(n_feat), size=np.random.randint(low, hi + 1), replace=False)))
            for _ in range(size)]

########################################################################################################################
# Define a helper function to mutate each hyperparameter with a specific offset
########################################################################################################################


def mutate_hps(hps: list[float],
               mutate: float):
    """
    :param hps: A list of floats.
                The list of hyperparameters to be mutated.
    :param mutate: A float.
           The offset of the hyperparameter.
    :return: A list of floats, representing the hyperparameters in hps after offsetting.
    """
    # Type and value check
    assert isinstance(hps, list), 'hps must be a list.'
    assert all(isinstance(hp, float) for hp in hps), 'Each element in hps must be a float.'
    assert isinstance(mutate, float), 'mutate must be a float.'

    return [hp + mutate if np.random.rand() < 0.5 else hp - mutate for hp in hps]

########################################################################################################################
# Define the IEL as a class
########################################################################################################################


class IELClassifier:
    def __init__(self,
                 n_max_gen: int = 100,
                 n_models_per_gen: int = 100,
                 n_top_mate: int = 40,
                 n_top_mutate: int = 20,
                 feat_config: tuple[int, int] = (2, 50),
                 lr_config: tuple[float, float, float] = (0.00001, 0.01, 0.0001),
                 beta1_config: tuple[float, float, float] = (0.9, 0.999, 0.001),
                 beta2_config: tuple[float, float, float] = (0.9, 0.999, 0.001)):
        """
        :param n_max_gen: A positive integer.
               The maximum number of generations to be run.
        :param n_models_per_gen: A positive integer.
               The number of models to be sampled in each generation.
        :param n_top_mate: A positive integer.
               The number of top models to be used for crossover in the genetic procedure.
        :param n_top_mutate: A positive integer.
               The number of top models (following n_top_mate) to be used for mutation in the genetic procedure.
        :param feat_config: A tuple of integers.
               The first (second) element represents the minimum (maximum) number of features to be sampled from.
        :param lr_config: A tuple of floats.
               The first (second) element represents the lower (upper) bound of the learning rates to be sampled from.
               The third element represents the mutation offset.
        :param beta1_config: A tuple of floats.
               The first (second) element represents the lower (upper) bound of the beta1 values to be sampled from.
               The third element represents the mutation offset.
        :param beta2_config: A tuple of floats.
               The first (second) element represents the lower (upper) bound of the beta2 values to be sampled from.
               The third element represents the mutation offset.
        """
        # Type and value check
        assert isinstance(n_max_gen, int), 'n_max_gen must be an integer.'
        assert n_max_gen > 1, 'n_max_gen must be greater than 1.'
        assert isinstance(n_models_per_gen, int), 'n_models_per_gen must be an integer.'
        assert n_models_per_gen >= 2 and n_models_per_gen % 2 == 0, ('n_models_per_gen must be greater than '
                                                                     'and divisible by 2')
        assert isinstance(n_top_mate, int), 'n_top_mate must be an integer.'
        assert n_top_mate >= 2 and n_top_mate % 2 == 0, ('n_top_mate must be greater than '
                                                         'and divisible by 2')
        assert n_models_per_gen >= n_top_mate, 'n_top_mate must be in the range [2, n_models_per_gen].'
        assert isinstance(n_top_mutate, int), 'n_top_mutate must be an integer.'
        assert n_top_mutate >= 2 and n_top_mutate % 2 == 0, ('n_top_mutate must be greater than '
                                                             'and divisible by 2')
        assert n_models_per_gen >= n_top_mutate, 'n_top_mutate must be in the range [2, n_models_per_gen].'
        assert n_top_mate + n_top_mutate <= n_models_per_gen, \
            f'Requirement: n_top_mate + n_top_mutate <= n_models_per_gen'
        assert isinstance(feat_config, tuple) and len(feat_config) == 2, 'feat_config must be a tuple of length 2.'
        assert all(isinstance(i, int) for i in feat_config), 'Each element in feat_config must be an integer.'
        assert 1 <= feat_config[0] <= feat_config[1], 'feat_config was not set up with the right bounds.'
        for idx, config_name in enumerate(['lr_config', 'beta1_config', 'beta2_config']):
            config = [lr_config, beta1_config, beta2_config][idx]
            assert isinstance(config, tuple) and len(config) == 3, \
                f'{config_name} must be a tuple of length 3'
            assert all(config[i] > 0 for i in range(3)), \
                f'{config_name} must be a tuple of positive elements.'
            assert config[0] < config[1], \
                f'{config_name}: 1st element must be less than the 2nd'
            assert config[2] < config[1] - config[0], \
                f'{config_name}: 3rd element must be less than the 2nd - 1st'

        # Define properties of the class
        self.n_max_gen: int = n_max_gen
        self.n_models_per_gen: int = n_models_per_gen
        self.n_top_mate: int = n_top_mate
        self.n_top_mutate: int = n_top_mutate
        self.hp_config_dict: dict[str, tuple] = {'n_feat': list(feat_config),
                                                 'lr': list(lr_config),
                                                 'beta1': list(beta1_config),
                                                 'beta2': list(beta2_config)}
        self.g_all_result: pd.DataFrame = pd.DataFrame(None)    # Initialize the storage for performance statistics
        print(f'An IEL model is created and ready to be fitted.', flush=True)
        print('*'*120, flush=True)

    ####################################################################################################################
    # Define a simple method to get the performance statistics
    ####################################################################################################################
    def get_performance(self):
        if self.g_all_result.shape[0] == 0:
            print(f'The IEL model has not been fitted yet.', flush=True)
        return self.g_all_result

    ####################################################################################################################
    # Define the model fitting process of IEL as a method.
    ####################################################################################################################
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            n_splits: int = 5,
            n_units: int = 300,
            n_classes: int = 2,
            n_layers: int = 3,
            n_epochs: int = 100,
            earlyStopper: EarlyStopping = EarlyStopping(patience=3),
            metric: Literal['Accuracy', 'AUROC', 'F1', 'NPV', 'Precision', 'Recall', 'Specificity',
                            'AIC', 'BIC', 'NLL'] = 'AUROC',
            maximize: bool = True,
            stop_cond: tuple[int, int, int, float] = (50, 20, 3, 0.01)):
        """
        :param X_train: A 2-dimensional numpy array (or Torch.Tensor).
               The training feature data with dimension of (sample size, number of features).
        :param y_train: A 1-dimensional numpy array (or Torch.Tensor).
               The training target 1-dimensional data with length as the sample size.
        :param n_splits: Integer.
               Number of folds (k) used in cross-validation.
               Default setting: n_splits=5
        :param n_units: A positive integer.
               The number of hidden units in each hidden layer.
               Default setting: n_units=300
        :param n_classes: A positive integer greater than 1.
               The number of classes in the target.
               Default setting: n_classes=2
        :param n_layers: A positive integer.
               The number of hidden layers in the model.
               Default setting: n_layers=3
        :param n_epochs: A positive integer.
               The number of maximum epochs to be run in training the model.
               Default setting: n_epochs=100
        :param earlyStopper: An EarlyStopping object or None.
               Default setting: earlyStopper=EarlyStopping(patience=3)
        :param metric: A string.
               The metric used to rank the models in each generation. See the options above.
               Default setting: metric='AUROC'
        :param maximize: A boolean.
               True if the metric specified is larger the better, otherwise False (e.g., AIC, BIC, NLL).
               Default setting: maximize=True
        :param stop_cond: A tuple of the form (integer, integer, integer, float), or None.
               A tuple used to define the stopping condition of the genetic procedure.
               The first element (A) specifies the minimum number of generations to be run before stopping.
               The second element (B) specifies the patience level to allow continuation of the genetic algorithm.
               The third element (C) specifies the number of top models to obtain summarized statistics.
               The fourth element (D) specifies the minimum required improvement in the specified metric.
               So, the condition is defined as "IEL will stop if the current generation is greater than (A), and the
               average validation statistics of the top-(C) models in each generation has not been improving for at
               least (D) for (B) generations since the best average validation statistic last observed. If None, no
               stoping condition will be enforced.
               Default setting: stop_cond=(50, 20, 3, 0.01)
        :return: None
        """
        # Type and value check --- most left to ann_embedded.py
        assert metric in ['Accuracy', 'AUROC', 'F1', 'NPV', 'Precision', 'Recall', 'Specificity',
                          'AIC', 'BIC', 'NLL'], \
            f'See the docstring for the possible values of metric.'
        assert isinstance(maximize, bool), 'maximize must be a boolean.'
        if stop_cond is not None:
            assert isinstance(stop_cond, tuple) and len(stop_cond) == 4, 'stop_cond must be a length-4 tuple.'
            assert isinstance(stop_cond[0], int) and 0 <= stop_cond[0] < self.n_max_gen, \
                'stop_cond[0] must be an integer in the range [0, n_max_gen].'
            assert isinstance(stop_cond[1], int) and 1 <= stop_cond[1] < self.n_max_gen, \
                'stop_cond[1] must be an integer in the range [1, n_max_gen].'
            assert isinstance(stop_cond[2], int) and 1 <= stop_cond[2] <= self.n_models_per_gen, \
                'stop_cond[2] must be an integer in the range [1, n_models_per_gen].'
            assert isinstance(stop_cond[3], (float, int)) and stop_cond[3] > 0, \
                'stop_cond[3] must be a positive float or integer.'

        ################################################################################################################
        # Set up the configurations before model fitting
        ################################################################################################################
        n_feat: int = X_train.shape[1]      # Number of features
        # Update the greatest number of features to be sampled
        self.hp_config_dict['n_feat'][0] = min(n_feat, self.hp_config_dict['n_feat'][0])
        hpc: dict = self.hp_config_dict               # The initialized configurations of hyperparameters
        n_models: int = self.n_models_per_gen         # The initialized number of models to be run per generation
        metric_rank_str: str = f'Val_{metric}_Rank'   # The name of the metric used for ranking models per generation

        ################################################################################################################
        # Clear memory
        ################################################################################################################
        gc.collect()                # Clear cpu cache
        torch.cuda.empty_cache()    # Clear gpu cache

        ################################################################################################################
        # [Generation 0] Model fitting (Chromosome initialization)
        ################################################################################################################
        log_head: str = f'[Gen. 0] '
        g0_t0: float = time()
        g0_results: list[dict[str, float]] = []     # Store the performance statistics for each model
        n_top: int = min(3, n_models)               # Number of top-models to be logged

        # Initialize the populations of learning rates, beta1, and beta2 for the AdamW optimizer, also the indices of
        # the features being sampled
        lr_pop: list[float] = sample_hp(hpc['lr'][0], hpc['lr'][1], n_models)
        beta1_pop: list[float] = sample_hp(hpc['beta1'][0], hpc['beta1'][1], n_models)
        beta2_pop: list[float] = sample_hp(hpc['beta2'][0], hpc['beta2'][1], n_models)
        feat_idx_pop: list[list[int]] = sample_feat_idx(n_feat, hpc['n_feat'][0], hpc['n_feat'][1], n_models)

        # Train the models with ANN_Classifier_Embedded defied in ann_embedded.py
        for model_idx, (lr, beta1, beta2, feat_idxs) in enumerate(zip(lr_pop, beta1_pop, beta2_pop, feat_idx_pop)):
            result: dict[str, float] = ANN_Classifier_Embedded(X=X_train[:, feat_idxs],
                                                               y=y_train,
                                                               n_splits=n_splits,
                                                               n_units=n_units,
                                                               n_classes=n_classes,
                                                               n_layers=n_layers,
                                                               n_epochs=n_epochs,
                                                               lr=lr,
                                                               beta1=beta1,
                                                               beta2=beta2,
                                                               earlyStopper=earlyStopper, device='cuda:0')
            result = {'Feature_Indices': feat_idxs} | result
            g0_results.append(result)
            if (model_idx + 1) % 10 == 0:
                print(f'{log_head}Model {model_idx + 1}/{n_models} fitted.', flush=True)

        ################################################################################################################
        # Clear memory
        ################################################################################################################
        gc.collect()                # Clear cpu cache
        torch.cuda.empty_cache()    # Clear gpu cache

        ################################################################################################################
        # [Generation 0] Store modeling results
        ################################################################################################################
        df_g0: pd.DataFrame = pd.DataFrame.from_records(g0_results)
        df_g0.insert(0, 'Model_Index', range(n_models))
        df_g0.insert(0, 'Generation', 0)
        df_g0.insert(2, metric_rank_str,
                     df_g0[f'Val_{metric}'].rank(method='first', ascending=not maximize).astype(int) - 1)
        top_score: float = f"{df_g0[df_g0[metric_rank_str] == 0][f'Val_{metric}'].values[0]:.4f}"
        top_mean_score: float = f"{df_g0[df_g0[metric_rank_str] < n_top][f'Val_{metric}'].mean():.4f}"
        top_n_feats: int = df_g0[df_g0[metric_rank_str] == 0]['#Features'].values[0]
        self.g_all_result = df_g0.copy()
        g0_t1: float = time()
        print(f'{log_head}Training completed in {g0_t1 - g0_t0:.1f}s.', flush=True)
        print(f"{log_head}Validation {metric} of the best model: {top_score}", flush=True)
        print(f"{log_head}Average validation {metric} of the best-{n_top} model: {top_mean_score}", flush=True)
        print(f"{log_head}Number of features used to fit the best model: {top_n_feats}", flush=True)
        print('*'*120, flush=True)

        ################################################################################################################
        # [Generation g > 0] Model evolution
        ################################################################################################################
        cur_g: int = 1
        stopping = False
        while cur_g < self.n_max_gen and not stopping:
            gi_t0: float = time()
            prior_g: int = cur_g - 1                                                                # Last generation
            prior_df: pd.DataFrame = self.g_all_result[self.g_all_result['Generation'] == prior_g]  # Last result

            # Initialize the populations of learning rates, beta1, and beta2 for the AdamW optimizer, also the indices
            # of the features being sampled
            lr_pop: list[float] = []
            beta1_pop: list[float] = []
            beta2_pop: list[float] = []
            feat_idx_pop: list[list[int]] = []

            ############################################################################################################
            # [Generation g] Set up children for crossover (aka mating) from the parents in generation (g-1)
            ############################################################################################################
            mate_range: list[int] = list(range(self.n_top_mate))
            prior_mate_df: pd.DataFrame = prior_df[prior_df[metric_rank_str].isin(mate_range)]
            mating_parent_idxs: list[int] = prior_mate_df['Model_Index'].to_list()

            # Sequential mating strategy
            dad_idxs: list[int] = [mating_parent_idxs[i] for i in range(len(mating_parent_idxs)) if i % 2 == 0]
            mum_idxs: list[int] = [mating_parent_idxs[i] for i in range(len(mating_parent_idxs)) if i % 2 == 1]
            for dad_idx, mum_idx in zip(dad_idxs, mum_idxs):
                # dad_lr, dad_beta1, dad_beta2, mum_lr, mum_beta1, mum_beta2: floats
                dad_lr, dad_beta1, dad_beta2 \
                    = prior_df.loc[prior_df['Model_Index'] == dad_idx, ['lr', 'beta1', 'beta2']].values[0]
                mum_lr, mum_beta1, mum_beta2 \
                    = prior_df.loc[prior_df['Model_Index'] == mum_idx, ['lr', 'beta1', 'beta2']].values[0]
                # Perform midpoint crossover
                lr_pop.append((dad_lr + mum_lr) / 2)
                beta1_pop.append((dad_beta1 + mum_beta1) / 2)
                beta2_pop.append((dad_beta2 + mum_beta2) / 2)

            ############################################################################################################
            # [Generation g] Set up children for mutation from the parents in generation (g-1)
            ############################################################################################################
            mutate_range: list[int] = list(range(self.n_top_mate, self.n_top_mate + self.n_top_mutate))
            prior_mutate_df: pd.DataFrame = prior_df[prior_df[metric_rank_str].isin(mutate_range)]
            lr_pop += mutate_hps(prior_mutate_df['lr'].to_list(), hpc['lr'][2])
            beta1_pop += mutate_hps(prior_mutate_df['beta1'].to_list(), hpc['beta1'][2])
            beta2_pop += mutate_hps(prior_mutate_df['beta2'].to_list(), hpc['beta2'][2])

            ############################################################################################################
            # [Generation g] Set up children for the feature sets from the parents in generation g-1
            ############################################################################################################
            top_range: list[int] = list(range(len(lr_pop)))
            feat_idx_pop += prior_df[prior_df[metric_rank_str].isin(top_range)]['Feature_Indices'].to_list()

            ############################################################################################################
            # [Generation g] Set up children for RANDOMIZATION
            ############################################################################################################
            lr_pop += sample_hp(hpc['lr'][0], hpc['lr'][1], n_models - len(lr_pop))
            beta1_pop += sample_hp(hpc['beta1'][0], hpc['beta1'][1], n_models - len(beta1_pop))
            beta2_pop += sample_hp(hpc['beta2'][0], hpc['beta2'][1], n_models - len(beta2_pop))
            feat_idx_pop += sample_feat_idx(n_feat, hpc['n_feat'][0], hpc['n_feat'][1], n_models - len(feat_idx_pop))
            assert len(feat_idx_pop) == n_models, ('Sanity check failed. The number of feature subsets is not the same '
                                                   'as the number of models generated.')

            ############################################################################################################
            # [Generation g] Clip to the valid range
            ############################################################################################################
            lr_pop = np.clip(lr_pop, hpc['lr'][0], hpc['lr'][1])
            beta1_pop = np.clip(beta1_pop, hpc['beta1'][0], hpc['beta1'][1])
            beta2_pop = np.clip(beta2_pop, hpc['beta2'][0], hpc['beta2'][1])

            ############################################################################################################
            # [Generation g] Model fitting
            ############################################################################################################
            log_head = f'[Gen. {cur_g}] '
            gi_results = []
            for model_idx, (lr, beta1, beta2, feat_idxs) in enumerate(zip(lr_pop, beta1_pop, beta2_pop, feat_idx_pop)):
                result: dict[str, float] = ANN_Classifier_Embedded(X=X_train[:, feat_idxs],
                                                                   y=y_train,
                                                                   n_splits=n_splits,
                                                                   n_units=n_units,
                                                                   n_classes=n_classes,
                                                                   n_layers=n_layers,
                                                                   n_epochs=n_epochs,
                                                                   lr=lr,
                                                                   beta1=beta1,
                                                                   beta2=beta2,
                                                                   earlyStopper=earlyStopper, device='cuda:0')
                result = {'Feature_Indices': feat_idxs} | result
                gi_results.append(result)
                if (model_idx + 1) % 10 == 0:
                    print(f'{log_head}Model {model_idx + 1}/{n_models} fitted.', flush=True)

            ############################################################################################################
            # Clear memory
            ############################################################################################################
            gc.collect()                # Clear cpu cache
            torch.cuda.empty_cache()    # Clear gpu cache

            ############################################################################################################
            # [Generation 0] Store modeling results
            ############################################################################################################
            df_gi: pd.DataFrame = pd.DataFrame.from_records(gi_results)
            df_gi.insert(0, 'Model_Index', range(n_models))
            df_gi.insert(0, 'Generation', cur_g)
            df_gi.insert(2, metric_rank_str,
                         df_gi[f'Val_{metric}'].rank(method='first', ascending=not maximize).astype(int) - 1)
            top_score: float = f"{df_gi[df_gi[metric_rank_str] == 0][f'Val_{metric}'].values[0]:.4f}"
            top_mean_score: float = f"{df_gi[df_gi[metric_rank_str] < n_top][f'Val_{metric}'].mean():.4f}"
            top_n_feats: int = df_gi[df_gi[metric_rank_str] == 0]['#Features'].values[0]
            self.g_all_result = pd.concat([self.g_all_result, df_gi], ignore_index=True)
            gi_t1: float = time()
            print(f'{log_head}Training completed in {gi_t1 - gi_t0:.1f}s.', flush=True)
            print(f"{log_head}Validation {metric} of the best model: {top_score}", flush=True)
            print(f"{log_head}Average validation {metric} of the best-{n_top} model: {top_mean_score}", flush=True)
            print(f"{log_head}Number of features used to fit the best model: {top_n_feats}", flush=True)

            ############################################################################################################
            # Perform stopping check
            ############################################################################################################
            if stop_cond is not None:

                min_gen: int = stop_cond[0]
                patience: int = stop_cond[1]
                avg_gens: int = stop_cond[2]
                delta: float = stop_cond[3]

                if cur_g >= min_gen:

                    # Identify the best averaged validation statistic so far
                    df_cur_avg: pd.DataFrame = self.g_all_result[
                        self.g_all_result[metric_rank_str].isin(range(avg_gens))]
                    df_cur_avg = (df_cur_avg[['Generation', f'Val_{metric}']].groupby('Generation')[f'Val_{metric}']
                                  .mean().reset_index())
                    best_val_score: float = df_cur_avg[f'Val_{metric}'].max() \
                        if maximize else df_cur_avg[f'Val_{metric}'].min()
                    best_gen: int = df_cur_avg.loc[df_cur_avg[f'Val_{metric}'] == best_val_score, 'Generation'].values[0]
                    cur_val_score: float = df_cur_avg.loc[df_cur_avg['Generation'] == cur_g, f'Val_{metric}'].values[0]
                    print(f'Best average validation {metric}: {best_val_score:.4f} (at Gen. {best_gen})', flush=True)
                    print(f'Current average validation {metric}: {cur_val_score:.4f} (at Gen. {cur_g})', flush=True)

                    # Determine if patience has run out
                    patience_counter: int = 0
                    for gen, val_score in zip(df_cur_avg[f'Generation'].to_list(),
                                              df_cur_avg[f'Val_{metric}'].to_list()):
                        if gen <= best_gen:
                            continue
                        if ((val_score < best_val_score + delta and maximize)
                                or (val_score > best_val_score - delta and not maximize)):
                            patience_counter += 1
                    if patience_counter >= patience:
                        stopping = True
                    if stopping:
                        print(f'Early stopping condition met at Generation {cur_g}.', flush=True)
                    else:
                        print(f'Early stopping condition not met at Generation {cur_g}. '
                              f'{patience_counter}/{patience} patience used.', flush=True)

            print('*'*120, flush=True)
            cur_g += 1

        ################################################################################################################
        # Report the end of the IEL process
        ################################################################################################################
        print(f'IEL fitting process completed in {time() - g0_t0:.1f}s.', flush=True)
        print('*' * 120, flush=True)

########################################################################################################################
# End of script.
########################################################################################################################
