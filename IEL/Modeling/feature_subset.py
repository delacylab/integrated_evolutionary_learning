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
# Overview: Using the results of the IEL procedure, identify a subset of features to be used to warm-start IEL again.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import kneed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from ast import literal_eval
from collections import defaultdict
from scipy.interpolate import UnivariateSpline
from typing import Union
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

########################################################################################################################
# Define a function to identify a feature subset for subsequent warm-start IEL modeling.
########################################################################################################################


def feature_subsetter(df: pd.DataFrame,
                      n_top_models: int = 5,
                      sd_multiplier: float = 0,
                      show_knee: bool = False):
    """
    :param df: A pandas.DataFrame.
           The result obtained from IEL.get_performance().
    :param n_top_models: A positive integer.
           The number of top-models in each generation used to extract the eli5 permutation-based feature importances.
           Default setting: n_top_models=5
    :param sd_multiplier: A float.
           The multiplier of the standard deviations of the eli5 importances of the features up to a certain
           automatically computed knee point.
           Default setting: sd_multiplier=0
    :param show_knee: A boolean.
           Whether to display the knee point projected in a geometrical plane with X-axis as the metric and Y-axis as
           the number of features. A PNG image will also be saved in the working directory as 'Knee.png' when show_knee
           is set as True.
           Default setting: show_knee = False
    :return: A list of integers, representing the feature indices recommended to use for warm-start IEL modeling.
    """
    # Type and value check
    assert isinstance(df, pd.DataFrame), 'Make sure you use the output of IEL.get_performance() as the input of df.'
    assert isinstance(n_top_models, int) and n_top_models >= 1, 'n_top_models must be a positive integer.'
    assert df['Model_Index'].max() + 1 > n_top_models, ('n_top_models must be in the range '
                                                        '[1, df["Model_Index"].max() + 1].')
    assert isinstance(sd_multiplier, (int, float)) and sd_multiplier >= 0, ('sd_multiplier must be a non-negative '
                                                                            'float or integer.')
    assert isinstance(show_knee, bool), 'show_elbow must be a boolean.'

    ####################################################################################################################
    # Step 1. Identify the records with the n_top_models models in each generation
    ####################################################################################################################
    metric_rank_str: str = [col for col in df.columns if isinstance(col, str)]
    metric_rank_str = [col for col in metric_rank_str if col.startswith('Val_') and col.endswith('Rank')][0]
    metric_str: str = metric_rank_str.replace('_Rank', '')
    assert not any(metric_str.endswith(y) for y in ['AIC', 'BIC', 'NLL']), ('Non-maximizing metric is not supported '
                                                                            'in the current release.')
    df_sub: pd.DataFrame = df[df[metric_rank_str].isin(range(n_top_models))]

    ####################################################################################################################
    # Step 2. Prepare the dataset with X as number of features and Y as the validation AUROC scores
    ####################################################################################################################
    X: np.ndarray = df_sub['#Features'].to_numpy()
    Y: np.ndarray = df_sub[metric_str].to_numpy()

    ####################################################################################################################
    # Step 3. Smooth the Y's values to reduce outlier effect
    ####################################################################################################################
    df_smooth: pd.DataFrame = pd.DataFrame({"X": X, "Y": Y}).groupby('X')['Y'].median().reset_index()
    Xs: np.ndarray = df_smooth['X'].values
    Ys: np.ndarray = df_smooth['Y'].values

    ####################################################################################################################
    # Step 4. Fit a spline to Xs and Ys
    ####################################################################################################################
    sorted_idx: np.ndarray = np.argsort(Xs)
    X_sorted: np.ndarray = Xs[sorted_idx]
    Y_sorted: np.ndarray = Ys[sorted_idx]
    k: int = min(3, len(X_sorted) - 1)      # Polynomial type
    spline: UnivariateSpline = UnivariateSpline(X_sorted, Y_sorted, k=k,
                                                s=np.var(Y_sorted) * len(X_sorted))     # Heuristic smoothing
    X_dense: np.ndarray = np.linspace(X_sorted.min(), X_sorted.max(), 500)         # 500 points to fit the spline
    Y_dense: np.ndarray = spline(X_dense)

    ####################################################################################################################
    # Step 5. Compute the derivatives of the spline and identify its increasing segments
    ####################################################################################################################
    dy: np.ndarray = spline.derivative(n=1)(X_dense)
    d2y: np.ndarray = spline.derivative(n=2)(X_dense)
    increasing_segments: list[tuple[int, int]] = []
    in_segment: bool = False
    start_idx: int = None
    for i in range(len(dy)):
        if dy[i] > 0:
            if not in_segment:
                start_idx, in_segment = i, True
        else:
            if in_segment:
                if i - start_idx >= 50:
                    increasing_segments.append((start_idx, i))
                in_segment = False
                start_idx = None
    if in_segment and start_idx is not None:
        if len(dy) - start_idx >= 50:
            increasing_segments.append((start_idx, len(dy)))

    ####################################################################################################################
    # Step 6. Identify the knee of the spline, if any
    ####################################################################################################################
    knee_X: Union[int, None] = None
    knee_Y: Union[float, None] = None

    # Identify the knee of the curve
    if increasing_segments:
        s_start, s_end = increasing_segments[0]
        x_seg = X_dense[s_start:s_end]
        y_seg = Y_dense[s_start:s_end]
        net_curvature: float = np.sum(d2y[s_start:s_end])
        shape: str = 'concave' if net_curvature < 0 else 'convex'
        knee: kneed.KneeLocator = kneed.KneeLocator(x_seg, y_seg, curve=shape, direction='increasing')
        knee_X = int(round(knee.knee)) if knee.knee is not None else None
        knee_Y = knee.knee_y

    if knee_X is None or knee_Y is None:
        print('No knee identified. Please attempt a different (or larger) value of n_top_models.', flush=True)
        print('*'*120, flush=True)
        exit()

    ####################################################################################################################
    # Step 7. (Optional) Visualize the knee, if any
    ####################################################################################################################
    if show_knee and knee_X is not None and knee_Y is not None:
        metric_full_str = metric_str.replace('Val_', 'Validation ')
        plt.plot(X_dense, Y_dense, label=metric_full_str + ' (Averaged)')
        plt.plot(knee_X, knee_Y, 'ro', markersize=10, label=f'Knee (with {knee_X} features)')
        plt.grid(alpha=0.2)
        plt.xlabel('Number of features', fontsize=12)
        plt.ylabel(metric_full_str, fontsize=12)
        plt.legend(loc='best')
        plt.savefig('knee.png', dpi=300, bbox_inches='tight')
        plt.show()

    ####################################################################################################################
    # Step 8. Extract the eli5 importance scores from df
    ####################################################################################################################
    feat_idx_records: list[list[int]] = df_sub['Feature_Indices'].apply(lambda x: literal_eval(x))
    eli5_records: list[list[int]] = df_sub['eli5_Importance'].apply(lambda x: literal_eval(x))
    feat_imp_dict: dict[int, list[float]] = defaultdict(list)
    for feat_idx_list, eli5_list in zip(feat_idx_records, eli5_records):
        assert len(feat_idx_list) == len(eli5_list)
        eli5_list = list(np.abs(eli5_list))
        for feat_idx, eli5_imp in zip(feat_idx_list, eli5_list):
            feat_imp_dict[feat_idx].append(eli5_imp)
    feat_imp_dict = dict(feat_imp_dict)

    ####################################################################################################################
    # Step 9. Identify the features with the highest maximum eli5 importance score
    ####################################################################################################################
    feat_imp_max_dict: dict = {k: max(v) for k, v in feat_imp_dict.items()}
    feat_imp_max_dict = dict(sorted(feat_imp_max_dict.items(), key=lambda x: x[1], reverse=True)[:knee_X])
    feat_imp_max_dict_out: dict = {k: round(v, 4) for k, v in feat_imp_max_dict.items()}

    ####################################################################################################################
    # Step 10. Further thresholding the importance scores
    ####################################################################################################################
    sd: float = np.std([item for k, v in feat_imp_dict.items() for item in v])
    threshold: float = min(feat_imp_max_dict.values()) + (sd_multiplier * sd)
    warm_start_feats: list[int] = [feat for feat, imp in feat_imp_max_dict.items() if imp >= threshold]

    ####################################################################################################################
    # Step 11. Return the feature indices for subsequent warm-start IEL modeling
    ####################################################################################################################
    print(f'Unthresholded feature indices (length={len(feat_imp_max_dict)}): {feat_imp_max_dict_out}', flush=True)
    print(f'Thresholded feature indices (length={len(warm_start_feats)}): {warm_start_feats}', flush=True)
    return warm_start_feats

########################################################################################################################
# End of script.
########################################################################################################################
