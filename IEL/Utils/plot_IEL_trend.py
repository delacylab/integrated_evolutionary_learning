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
# Overview: Define a function to visualize the learning trend of IEL in terms of a user-defined metric.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import matplotlib.pyplot as plt
import pandas as pd
from typing import Literal

########################################################################################################################
# Define the plotting function
########################################################################################################################


def plot_stat_trend(df: pd.DataFrame,
                    n_top_models: int = 5,
                    smooth_window: int = 10,
                    metric: Literal['Accuracy', 'AUROC', 'F1', 'NPV', 'Precision', 'Recall', 'Specificity',
                                    'AIC', 'BIC', 'NLL'] = 'AUROC',
                    maximize: bool = True,
                    filename: str = 'Temp.png'):
    """
    :param df: A pandas.DataFrame.
           The result obtained from IEL.get_performance().
    :param n_top_models: A positive integer.
           The number of top-models in each generation used.
           Default setting: n_top_models=5
    :param smooth_window: A positive integer.
           The width of the sliding window.
           Default setting: smooth_window=10
    :param metric: A string.
           The metric used to rank the models in each generation. See the options above.
           Default setting: metric='AUROC'
    :param maximize: A boolean.
           True if the metric specified is larger the better, otherwise False (e.g., AIC, BIC, NLL).
           Default setting: maximize=True
    :param filename: A string.
           The file name of the PNG image file to be saved in the working directory.
           Default setting: filename='Temp.png'
    :return: None
    """
    # Type and value check
    assert isinstance(df, pd.DataFrame), 'Make sure you use the output of IEL.get_performance() as the input of df.'
    assert isinstance(n_top_models, int) and n_top_models >= 1, 'n_top_models must be a positive integer.'
    assert df['Model_Index'].max() + 1 > n_top_models, ('n_top_models must be in the range '
                                                        '[1, df["Model_Index"].max() + 1].')
    assert isinstance(smooth_window, int), 'smooth_window must be an integer.'
    assert df['Generation'].max() > smooth_window > 1, 'smooth window must be in the range [1, df["Generation"].max()].'
    assert metric in ['Accuracy', 'AUROC', 'F1', 'NPV', 'Precision', 'Recall', 'Specificity',
                      'AIC', 'BIC', 'NLL'], f'See the docstring for the possible values of metric.'
    assert isinstance(maximize, bool), 'maximize must be a boolean.'

    ####################################################################################################################
    # Step 1. Identify the records with the n_top_models models in each generation.
    ####################################################################################################################
    metric_str: str = f'Val_{metric}'
    metric_rank_str: str = f'Val_{metric}_Rank'
    df[df[metric_rank_str]] = df[metric_str].rank(method='first', ascending=not maximize).astype(int) - 1
    df_sub: pd.DataFrame = df[df[metric_rank_str].isin(range(n_top_models))]

    ####################################################################################################################
    # Step 2. Compute the mean statistics across the n_top_models in each generation.
    ####################################################################################################################
    df_sub = df_sub[['Generation', metric_str]].groupby('Generation')[metric_str].mean().reset_index()

    ####################################################################################################################
    # Step 3. Compute the smoothened mean statistics using a sliding window approach.
    ####################################################################################################################
    smooth_stat = df_sub[metric_str].rolling(window=smooth_window, center=True, min_periods=1).mean()

    ####################################################################################################################
    # Step 4. Set up a plot for the results.
    ####################################################################################################################
    plt.plot(df_sub['Generation'], df_sub[metric_str], linestyle='-', lw=2, color='blue',
             label=f'{metric} averaged across top-{n_top_models} models')
    plt.plot(df_sub['Generation'], smooth_stat, linestyle='-', lw=3, color='orange',
             label=f'Smoothened {metric} (window size={smooth_window})')
    plt.xlabel(f'Validation {metric}', fontsize=14)
    plt.ylabel('Generation', fontsize=14)
    plt.grid(alpha=0.2)
    plt.legend(loc='best')

    ####################################################################################################################
    # Step 5. Display and save the plot.
    ####################################################################################################################
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

########################################################################################################################
# End of script.
########################################################################################################################
