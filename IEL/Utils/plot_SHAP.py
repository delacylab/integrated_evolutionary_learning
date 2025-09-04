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
# Overview: Define a function to visualize the SHAP value distribution relative to the scaled feature values.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Union

####################################################################################################################
# Define a simple function to separate the feature names/descriptions by lines
####################################################################################################################


def sep_desc(desc: str,
             max_char: int):
    """
    :param desc: A string. The name/description of a feature to be visualized.
    :param max_char: A positive integer. The maximum length in characters to be displayed in a line.
    :return: A string. The desc input separated by '\n' with each line containing not more than max_char characters.
    """
    desc_new = ''
    if len(desc) < max_char:
        return desc
    else:
        desc_cur = desc
        while len(desc_cur) > 0:
            # Identify the last space before the max_char
            last_space = desc_cur.rfind(' ', 0, max_char)
            if last_space > max_char:
                desc_new += desc_cur[:last_space].strip() + '\n'
                desc_cur = desc_cur[last_space + 1:]
            elif len(desc_cur.strip()) < max_char:
                desc_new += desc_cur.strip()
                break
            elif last_space != -1 and last_space <= max_char:
                desc_new += desc_cur[:last_space].strip() + '\n'
                desc_cur = desc_cur[last_space + 1:]
            elif last_space == -1:
                desc_new += desc_cur[:last_space].strip()
                break
        return desc_new

####################################################################################################################
# Define a function to visualize the SHAP values relative to the feature values.
####################################################################################################################


def plot_shap_beeswarm(shap: np.ndarray,
                       X: np.ndarray,
                       feat_names: Optional[list[str]] = None,
                       max_char: int = 30,
                       right_shift: float = 0.1,
                       filename: str = 'Temp.png',
                       random_state: Optional[Union[int, np.random.Generator]] = 42):
    """
    :param shap: A 2-dimensional numpy array.
           A data matrix (returned by ann_refit.refit_and_eval) storing SHAP values with dimension of (test sample size,
           number of features).
    :param X: A 2-dimensional numpy array.
           The test feature dataset with dimension of (test sample size, number of features).
    :param feat_names: A list of strings.
           The names/descriptions of the features to be displayed. If None, feature names will be generated
           automatically as [X1, ..., Xn].
           Default setting: feat_names=None
    :param max_char: A positive integer.
           The maximum number of characters for feat_names to be displayed in a line.
           Default setting: max_char=30
    :param right_shift: A float.
           The fraction of the figure width that the left edge is shifted to the right, allowing more space for
           displaying feat_names.
           Default setting: filename=0.1
    :param filename: A string.
           The file name of the PNG image file to be saved in the working directory.
           Default setting: filename='Temp.png'
    :param random_state: An integer or a numpy RNG object.
           Controls the sampling process when sample size exceed 300.
           Default setting: random_state=42
    :return:
    """
    # Type and value check
    assert isinstance(shap, np.ndarray), 'shap must be a np.ndarray.'
    assert len(shap.shape) == 2, 'shap must be 2-dimensional.'
    assert isinstance(X, np.ndarray), 'X must be a np.ndarray.'
    assert len(X.shape) == 2, 'X must be 2-dimensional.'
    assert X.shape == shap.shape, 'shap and X must have the same dimension.'
    if feat_names is not None:
        assert isinstance(feat_names, list), 'feat_names, if not None, must be a list.'
        assert all(isinstance(feat, str) for feat in feat_names), 'Each element in feat_names must be a string.'
        assert len(feat_names) == shap.shape[1], 'feat_names, if not None, has have a length equal to shap.shape[1].'
    assert isinstance(max_char, int) and max_char >= 1, 'max_char must be a positive integer.'
    assert isinstance(right_shift, float) and right_shift > 0, 'right_shift must be a positive float.'
    assert isinstance(filename, str), 'filename must be a string.'

    ####################################################################################################################
    # Step 1. Separate the features names into lines.
    ####################################################################################################################
    feat_names: list[str] = sep_desc(deepcopy(feat_names), max_char=max_char)

    ####################################################################################################################
    # Step 2. Compute the mean absolute SHAP values.
    ####################################################################################################################
    mashap = pd.DataFrame({'Feature': feat_names, 'MASHAP': np.mean(np.abs(shap), axis=0)})
    mashap = mashap.sort_values(by='MASHAP', ascending=False, ignore_index=True)
    sorted_feat: list[str] = mashap['Feature'].to_list()
    sorted_feat_idx: list[int] = [feat_names.index(feat) for feat in sorted_feat]

    ####################################################################################################################
    # Step 3. Subset to the top-20 features to be displayed.
    ####################################################################################################################
    if len(sorted_feat) > 20:
        warnings.warn('Due to readability, showing only the top 20 features (in mean-absolute SHAP across samples).')
        sorted_feat_idx = sorted_feat_idx[:20]
        sorted_feat = sorted_feat[:20]
        shap = shap[:, sorted_feat_idx]
        X = deepcopy(X[:, sorted_feat_idx])

    ####################################################################################################################
    # Step 4. Subset to 300 samples at a maximum for readability.
    ####################################################################################################################
    if X.shape[0] > 300:
        warnings.warn('Due to readability, showing only 300 random samples.')
        rng = np.random.default_rng(random_state)
        samples = rng.integers(low=0, high=X.shape[0], size=300)
        shap = shap[samples, :]
        X = X[samples, :]

    ####################################################################################################################
    # Step 5. Scale X to the unit interval using min-max normalization.
    ####################################################################################################################
    X: np.ndarray = MinMaxScaler().fit_transform(deepcopy(X))

    ####################################################################################################################
    # Step 6. Create dataframes for plotting.
    ####################################################################################################################
    shap_df: pd.DataFrame = pd.DataFrame(shap, columns=sorted_feat)
    feat_df: pd.DataFrame = pd.DataFrame(X, columns=sorted_feat)
    shap_long: pd.DataFrame = shap_df.melt(var_name='Feature', value_name='SHAP value')
    feat_long: pd.DataFrame = feat_df.melt(var_name='Feature', value_name='Feature value')
    combined_df: pd.DataFrame = (pd.concat([shap_long, feat_long['Feature value']], axis=1)
                                 .dropna(subset=['Feature value']))

    ####################################################################################################################
    # Step 7. Create the plot.
    ####################################################################################################################
    plt.figure(figsize=(10, 6))
    norm = plt.Normalize(0, 1)
    ax = sns.stripplot(data=combined_df, y='Feature', x='SHAP value', hue='Feature value', palette='coolwarm',
                       hue_norm=norm, jitter=0.3, alpha=0.7, linewidth=0.5, legend=False, dodge=False)
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Feature values (min-max normalized)', fontsize=14)
    cbar.ax.text(2.5, 1.0, 'More\nrisk', transform=cbar.ax.transAxes,
                 va='center', ha='left', fontsize=12, color='black')
    cbar.ax.text(2.5, 0.0, 'More\nprotective', transform=cbar.ax.transAxes,
                 va='center', ha='left', fontsize=12, color='black')

    ####################################################################################################################
    # Step 8. Add annotation.
    ####################################################################################################################
    ax.set_xlim(combined_df['SHAP value'].min()*1.03, combined_df['SHAP value'].max()*1.03)
    xmin, xmax = ax.get_xlim()
    bar_y = 1.03
    ax.plot([0, xmax], [bar_y, bar_y], transform=ax.get_xaxis_transform(),
            color='black', linewidth=1.5, clip_on=False)
    ax.plot([0, 0], [bar_y+0.01, bar_y-0.01], transform=ax.get_xaxis_transform(),
            color='black', linewidth=1.5, clip_on=False)
    ax.plot([xmax, xmax], [bar_y+0.01, bar_y-0.01], transform=ax.get_xaxis_transform(),
            color='black', linewidth=1.5, clip_on=False)
    ax.text((xmax + 0) / 2, 1.04, 'Feature impact', ha='center', va='bottom', fontsize=12,
            transform=ax.get_xaxis_transform())
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2)

    ####################################################################################################################
    # Step 9. Configure the plot.
    ####################################################################################################################
    plt.xlabel('SHAP values', fontsize=14)
    plt.ylabel('')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=12.5)
    ax.tick_params(axis='y', pad=10)
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.subplots_adjust(left=right_shift)

    ####################################################################################################################
    # Step 10. Return and display the plot.
    ####################################################################################################################
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

########################################################################################################################
# End of script.
########################################################################################################################
