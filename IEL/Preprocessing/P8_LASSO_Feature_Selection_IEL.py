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
# Overview: This script uses IEL to run hyperparameter optimization of the L1-regularization term used in LASSO
# logistic regression.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import gc
import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from time import time
from typing import Literal, Union
if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Preprocessing.metrics import classify_discrimination_metrics, classify_AIC_BIC, classify_AUROC
else:
    from .metrics import classify_discrimination_metrics, classify_AIC_BIC, classify_AUROC
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
    return list(10 ** np.random.uniform(low=np.log10(low), high=np.log10(hi), size=size))

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
# Define a helper function to run LASSO with cross-validation
########################################################################################################################


def LASSO_cv(X: np.ndarray,
             y: np.ndarray,
             alpha: float,
             n_splits: int = 5,
             random_state: int = 42):
    """
    :param X: A 2-dimensional numpy.ndarray.
           The training feature dataset with dimension (training sample size, number of features).
    :param y: A 1-dimensional numpy.ndarray.
           The training target dataset.
    :param alpha: A positive float.
           The L1-regularization term.
    :param n_splits: Integer.
           Number of folds (k) used in cross-validation.
    :param random_state: Integer.
           Random seed used for stratification.
    :return: A dictionary storing the performance statistics averaged across folds.
    """
    # We leave all the type and value checks to the function in sklearn.
    skf: StratifiedKFold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splitter = skf.split(X, y)
    fit_results: list[dict] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter, 1):
        M = LogisticRegression(penalty='l1', C=alpha, random_state=random_state, solver='saga', n_jobs=-1)
        X_train: np.ndarray = np.take(X, train_idx, axis=0)
        X_val: np.ndarray = np.take(X, val_idx, axis=0)
        y_train: np.ndarray = np.take(y, train_idx, axis=0)
        y_val: np.ndarray = np.take(y, val_idx, axis=0)
        M.fit(X_train, y_train)
        y_pred_label: np.ndarray = M.predict(X_val)
        y_pred_prob: np.ndarray = M.predict_proba(X_val)[:, 1]
        fit_result = (classify_discrimination_metrics(y_true=y_val, y_pred=y_pred_label, prefix='Val_')
                      | classify_AUROC(y_true=y_val, y_score=y_pred_prob, prefix='Val_')
                      | classify_AIC_BIC(y_true=y_val, y_score=y_pred_prob, n_params=X_val.shape[1], prefix='Val_'))
        fit_results.append({k: v for k, v in fit_result.items() if not isinstance(v, list)})
    df_stat: pd.DataFrame = pd.DataFrame.from_records(fit_results)
    mean_row: dict[str, float] = df_stat.mean().to_dict()
    mean_row = {'alpha': alpha} | mean_row
    return mean_row

########################################################################################################################
# Define the LASSO (optimized by IEL) as a class
########################################################################################################################


class LASSO_IEL_Classifier:
    def __init__(self,
                 n_max_gen: int = 100,
                 n_models_per_gen: int = 100,
                 n_top_mate: int = 40,
                 n_top_mutate: int = 20,
                 alpha_config: tuple[float, float, float] = (0.0001, 10000, 0.0001)):
        """
        :param n_max_gen: A positive integer.
               The maximum number of generations to be run.
        :param n_models_per_gen: A positive integer.
               The number of models to be sampled in each generation.
        :param n_top_mate: A positive integer.
               The number of top models to be used for crossover in the genetic procedure.
        :param n_top_mutate: A positive integer.
               The number of top models (following n_top_mate) to be used for mutation in the genetic procedure.
        :param alpha_config: A tuple of integers/floats.
               The first (second) element represents the lower (upper) bound of the regularization term to be sampled
               from when running Logistic Regression. The third element represents the mutation offset. The default
               values of the first and second terms are referenced from sklearn.linear_model.LogisticRegressionCV.
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
        assert isinstance(alpha_config, tuple) and len(alpha_config) == 3, f'alpha_config must be a tuple of length 3.'
        assert all(alpha_config[i] > 0 for i in range(3)), f'alpha_config must be a tuple of positive elements.'
        assert alpha_config[0] < alpha_config[1], \
            f'alpha_config: 1st element must be less than the 2nd'
        assert alpha_config[2] < alpha_config[1] - alpha_config[0], \
            f'alpha_config: 3rd element must be less than the 2nd - 1st'

        # Define properties of the class
        self.n_max_gen: int = n_max_gen
        self.n_models_per_gen: int = n_models_per_gen
        self.n_top_mate: int = n_top_mate
        self.n_top_mutate: int = n_top_mutate
        self.hp_config_dict: dict[str, tuple] = {'alpha': list(alpha_config)}
        self.g_all_result: pd.DataFrame = pd.DataFrame(None)    # Initialize the storage for performance statistics
        print(f'A LASSO-IEL model is created and ready to be fitted.', flush=True)
        print('*'*120, flush=True)

    ####################################################################################################################
    # Define a simple method to get the performance statistics
    ####################################################################################################################
    def get_performance(self):
        if self.g_all_result.shape[0] == 0:
            print(f'The LASSO-IEL model has not been fitted yet.', flush=True)
        return self.g_all_result

    ####################################################################################################################
    # Define the model fitting process of IEL as a method.
    ####################################################################################################################
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            n_splits: int = 5,
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
        # Type and value check --- most left to sklearn.linear_model.LogisticRegression
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
        hpc: dict = self.hp_config_dict               # The initialized configurations of hyperparameters
        n_models: int = self.n_models_per_gen         # The initialized number of models to be run per generation
        metric_rank_str: str = f'Val_{metric}_Rank'   # The name of the metric used for ranking models per generation

        ################################################################################################################
        # Clear memory
        ################################################################################################################
        gc.collect()                # Clear cpu cache

        ################################################################################################################
        # [Generation 0] Model fitting (Chromosome initialization)
        ################################################################################################################
        log_head: str = f'[Gen. 0] '
        g0_t0: float = time()
        g0_results: list[dict[str, float]] = []     # Store the performance statistics for each model
        n_top: int = min(3, n_models)               # Number of top-models to be logged

        # Initialize the population of alphas
        alpha_pop: list[float] = sample_hp(hpc['alpha'][0], hpc['alpha'][1], n_models)

        # Train the models with my defined LASSO_cv
        for model_idx, alpha in enumerate(alpha_pop):
            result: dict[str, float] = LASSO_cv(X=X_train,
                                                y=y_train,
                                                alpha=alpha,
                                                n_splits=n_splits)
            g0_results.append(result)
            if (model_idx + 1) % 10 == 0:
                print(f'{log_head}Model {model_idx + 1}/{n_models} fitted.', flush=True)

        ################################################################################################################
        # Clear memory
        ################################################################################################################
        gc.collect()                # Clear cpu cache

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
        self.g_all_result = df_g0.copy()
        g0_t1: float = time()
        print(f'{log_head}Training completed in {g0_t1 - g0_t0:.1f}s.', flush=True)
        print(f"{log_head}Validation {metric} of the best model: {top_score}", flush=True)
        print(f"{log_head}Average validation {metric} of the best-{n_top} model: {top_mean_score}", flush=True)
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

            # Initialize the population of alpha
            alpha_pop: list[float] = []

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
                dad_alpha = prior_df.loc[prior_df['Model_Index'] == dad_idx, ['alpha']].values[0]
                mum_alpha = prior_df.loc[prior_df['Model_Index'] == mum_idx, ['alpha']].values[0]
                alpha_pop.append(((dad_alpha + mum_alpha) / 2)[0])            # Perform midpoint crossover

            ############################################################################################################
            # [Generation g] Set up children for mutation from the parents in generation (g-1)
            ############################################################################################################
            mutate_range: list[int] = list(range(self.n_top_mate, self.n_top_mate + self.n_top_mutate))
            prior_mutate_df: pd.DataFrame = prior_df[prior_df[metric_rank_str].isin(mutate_range)]
            alpha_pop += mutate_hps(prior_mutate_df['alpha'].to_list(), hpc['alpha'][2])

            ############################################################################################################
            # [Generation g] Set up children for RANDOMIZATION
            ############################################################################################################
            alpha_pop += sample_hp(hpc['alpha'][0], hpc['alpha'][1], n_models - len(alpha_pop))

            ############################################################################################################
            # [Generation g] Clip to the valid range
            ############################################################################################################
            alpha_pop = np.clip(alpha_pop, hpc['alpha'][0], hpc['alpha'][1])

            ############################################################################################################
            # [Generation g] Model fitting
            ############################################################################################################
            log_head = f'[Gen. {cur_g}] '
            gi_results = []
            for model_idx, alpha in enumerate(alpha_pop):
                result: dict[str, float] = LASSO_cv(X=X_train,
                                                    y=y_train,
                                                    alpha=alpha,
                                                    n_splits=n_splits)
                gi_results.append(result)
                if (model_idx + 1) % 10 == 0:
                    print(f'{log_head}Model {model_idx + 1}/{n_models} fitted.', flush=True)

            ############################################################################################################
            # Clear memory
            ############################################################################################################
            gc.collect()                # Clear cpu cache

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
            self.g_all_result = pd.concat([self.g_all_result, df_gi], ignore_index=True)
            gi_t1: float = time()
            print(f'{log_head}Training completed in {gi_t1 - gi_t0:.1f}s.', flush=True)
            print(f"{log_head}Validation {metric} of the best model: {top_score}", flush=True)
            print(f"{log_head}Average validation {metric} of the best-{n_top} model: {top_mean_score}", flush=True)

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
                    best_gen: int = df_cur_avg.loc[df_cur_avg[f'Val_{metric}'] == best_val_score,
                                                   'Generation'].values[0]
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
        print(f'LASSO-IEL fitting process completed in {time() - g0_t0:.1f}s.', flush=True)
        print('*' * 120, flush=True)


########################################################################################################################
# Test Run
########################################################################################################################
if __name__ == '__main__':

    from sklearn.datasets import make_classification

    # Step 1. Generate a synthetic dataset
    X_, y_ = make_classification(n_samples=100,
                                 n_features=50,
                                 n_informative=10,
                                 n_redundant=40,
                                 n_clusters_per_class=2,
                                 random_state=42)

    # Step 2. Create the LASSO-IEL model
    M_ = LASSO_IEL_Classifier(n_max_gen=20,
                              n_models_per_gen=20,
                              n_top_mate=4,
                              n_top_mutate=2,
                              alpha_config=(0.0001, 10000, 0.0001))

    # Step 3. Fit the LASSO-IEL model
    M_.fit(X_, y_, n_splits=3, metric='AUROC', maximize=True, stop_cond=(10, 5, 3, 0.01))

    # Step 4. Get the performance statistics and identify the best alpha term
    df_ = M_.get_performance()
    df_.to_csv('Temp.csv', index=False)
    best_alpha = df_.iloc[df_['Val_AUROC'].idxmax()]['alpha']
    print(f'The best alpha obtained is {best_alpha:.4f}', flush=True)

    # Step 5. Lastly, you can refit a Logistic Regression model with the identified best_alpha to perform feature
    # selection directly.
    M_opt = LogisticRegression(penalty='l1', C=best_alpha, solver='saga', n_jobs=-1)
    M_opt.fit(X_, y_)
    linear_coef = M_opt.coef_[0]
    fs_result = pd.DataFrame({'Feature': [f'X{i}' for i in range(X_.shape[1])],
                              'Linear_Coefficient': linear_coef,
                              'Absolute_Linear_Coefficient': np.abs(linear_coef)})
    threshold = 0
    fs_result['Removal'] = (fs_result['Absolute_Linear_Coefficient'] > threshold).astype('Int32')
    print(fs_result)
