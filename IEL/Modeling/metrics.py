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
# Overview: This script defines a list of performance metrics to evaluate a predictive model for classification tasks.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import torch
import warnings
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
                             log_loss)
from sklearn.preprocessing import label_binarize, KBinsDiscretizer      # For classification
from typing import Literal, Optional
warnings.simplefilter(action='ignore', category=FutureWarning)

########################################################################################################################
# Standard classification metrics (Accuracy, F1, Precision, Recall, Specificity, NPV).
########################################################################################################################


def classify_discrimination_metrics(y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    prefix: str = "",
                                    average: Optional[Literal['binary', 'micro', 'macro', 'weighted']] = 'binary'):
    """
    Compute 6 metrics for binary/multi-class classification tasks: Accuracy, F1, Precision, Recall, Specificity, NPV
    See https://www.kaggle.com/code/nkitgupta/evaluation-metrics-for-multi-class-classification

    :param y_true: A one-dimensional numpy array.
           Values of the true label.
    :param y_pred: A one-dimensional numpy array.
           Values of the predicted label.
    :param prefix: A string.
           The prefix (e.g. 'Train_') of the metric name shown in the keys of the output dictionary.
           Default setting: prefix=""
    :param average: A string in ['binary', 'micro', 'macro', 'weighted'] or None.
           Configuration for averaging techniques.
           'binary': use only for binary classification
           'micro': compute metric globally by counting the total true positives, false negatives, and false positives
           'macro': compute the averaged metric for each label in an unweighted manner
           'weighted': similar to 'macro' but weighted by the number of true instances for each label
           None: compute metric for each label without averaging
           Default setting: average='weighted'
    :return:
    A dictionary with keys as {prefix}{metric name} and values as results.
    """

    # Type and value check
    try:
        y_true = np.array(y_true)
    except TypeError:
        raise TypeError(f"y_true must be (convertible to) a numpy array. Now its type is {type(y_true)}.")
    try:
        y_pred = np.array(y_pred)
    except TypeError:
        raise TypeError(f"y_pred must be (convertible to) a numpy array. Now its type is {type(y_pred)}.")
    assert len(y_true.shape) == 1, \
        f"The true labels y_true must be one-dimensional. Now it is {len(y_true.shape)}-dimensional."
    assert len(y_pred.shape) == 1, \
        f"The predicted labels y_pred must be one-dimensional. Now it is {len(y_pred.shape)}-dimensional."
    assert isinstance(prefix, str), \
        f"prefix must be a string. Now its type is {type(prefix)}."
    if average is not None:
        assert average in ['binary', 'micro', 'macro', 'weighted'], \
            (f"average (if not None) must be a string in ['binary', 'micro', 'macro', 'weighted']. "
             f" Now it is {average}.")
    n_classes: int = len(np.unique(y_true))
    if n_classes != 2:
        warnings.warn("Currently supporting only binary labels. Returning numpy.nan.")
        return {f'{prefix}TP': np.nan,
                f'{prefix}FP': np.nan,
                f'{prefix}TN': np.nan,
                f'{prefix}FN': np.nan,
                f'{prefix}Accuracy': np.nan,
                f'{prefix}F1': np.nan,
                f'{prefix}NPV': np.nan,
                f'{prefix}Precision': np.nan,
                f'{prefix}Recall': np.nan,
                f'{prefix}Specificity': np.nan}

    cm: np.ndarray = confusion_matrix(y_true, y_pred)
    tn: int = cm[0, 0]
    fn: int = cm[1, 0]
    tp: int = cm[1, 1]
    fp: int = cm[0, 1]
    result: dict[str, float] = {}     # Storing the computed statistics

    # Metric 1: Accuracy
    # - Accuracy works the same for binary and multiclass comparison.
    # - No zero division handling needed because of the assertion that the number of classes in the true label > 1.
    # - Returning a float object.
    acc: float = accuracy_score(y_true, y_pred)
    result[f'{prefix}Accuracy'] = acc

    # Metric 2: Specificity
    # - No zero division handling needed because of the assertion that the number of classes in the true label > 1.
    # - Returning a float object if average in ['macro', 'micro', 'weighted'], or a list if average == None.
    spec: float = tn / (tn + fp) if tn + fp > 0 else np.nan
    result[f'{prefix}Specificity'] = spec

    # Metric 3, 4, 5: F1, Precision, Recall
    # - Cases of zero division is represented by numpy.nan with a warning.
    # - Returning a float object if average in ['macro', 'micro', 'weighted'], or a list if average == None.
    for (metric_str, metric_func) in [('Precision', precision_score),
                                      ('Recall', recall_score),
                                      ('F1', f1_score)]:
        score: float = metric_func(y_true, y_pred, average=average, zero_division=np.nan)
        warn_msg: str = f"Division by zero error occurred when computing {metric_str.lower()}. Returning numpy.nan."
        if isinstance(score, np.ndarray):
            score: list[float] = list(score)
            if np.any(np.isnan(score)):
                warnings.warn(warn_msg)
        elif np.isnan(score):
            warnings.warn(warn_msg)
        result[f'{prefix}{metric_str}'] = score

    # Metric 6: NPV
    if tn + fn > 0:
        result[f'{prefix}NPV'] = tn / (tn + fn)
    else:
        warnings.warn("Division by zero error occurred when computing NPV. Returning numpy.nan.")
        result[f'{prefix}NPV'] = np.nan

    return {f'{prefix}TP': tp,
            f'{prefix}FP': fp,
            f'{prefix}TN': tn,
            f'{prefix}FN': fn} | {k: result[k] for k in sorted(result.keys())}

########################################################################################################################
# Classification metric (used with predicted probability measures): AUROC.
########################################################################################################################


def classify_AUROC(y_true: np.ndarray,
                   y_score: np.ndarray,
                   prefix: str = "",
                   average: Optional[Literal['micro', 'macro']] = 'micro'):
    """
    Compute AUROC (Area Under the Receiver Operating Characteristic curve), TPR (True Positive Rates), and FPR (False
    Positive Rates) by comparing the true labels and the predicted probability measures (NOT the predicted labels). In
    the multi-class classification case, this function adopts the one-versus-rest strategy (comparing each class with
    the rest of the classes) using either of the following averaging approaches:
    (i) micro-averaging: focus on overall performance by treating each sample equally;
    (ii) macro-averaging: focus on per-class performance by treating each class equally important.

    :param y_true: A one-dimensional numpy array.
           Values of the true label.
    :param y_score: A one-dimensional (binary) or two-dimensional (multi-class) numpy array.
           Probability measures of each class.
    :param prefix: A string.
           The prefix (e.g. 'Train_') of the metric name shown in the keys of the output dictionary.
           Default setting: prefix=""
    :param average: A string in ['micro', 'macro'] or None.
           Averaging approach for multi-class cases.
           Default setting: average='micro'
    :return:
    (a) A dictionary with key as {prefix}AUROC and value as the AUROC value.
    (b) A list of TPR values.
    (c) A list of FPR values.
    """
    try:
        y_true = np.array(y_true)
    except TypeError:
        raise TypeError(f"y_true must be (convertible to) a numpy array. Now its type is {type(y_true)}.")
    try:
        y_score = np.array(y_score)
    except TypeError:
        raise TypeError(f"y_score must be (convertible to) a numpy array. Now its type is {type(y_score)}.")
    assert len(y_true.shape) == 1, \
        f"y_true must be one-dimensional. Now it is {len(y_true.shape)}-dimensional."
    assert len(y_score.shape) in [1, 2], \
        f"y_score must be one-/two-dimensional. Now it is {len(y_score.shape)}-dimensional."
    assert isinstance(prefix, str), \
        f"prefix must be a string. Now its type is {type(prefix)}."
    if average is not None:
        assert average in ['micro', 'macro'], \
            f"average (if not None) must be a string in ['micro', 'macro']. Now it is {average}."
    n_classes = len(np.unique(y_true))
    if n_classes == 1:
        warnings.warn("Only one class found in the true label. Returning numpy.nan.")
        return {f'{prefix}AUROC': np.nan,
                f'{prefix}TPR': np.nan,
                f'{prefix}FPR': np.nan}

    # AUROC in the binary case
    if n_classes == 2:
        assert y_score.shape == y_true.shape, \
            f"The probability measures y_score should be in the same shape as the true labels y_true"
        fpr, tpr, _ = roc_curve(y_true, y_score)    # fpr: np.ndarray; tpr: np.ndarray
        auroc: float = roc_auc_score(y_true, y_score, average=average)

    # AUROC in the multi-class case
    else:
        assert len(y_score.shape) == 2, \
            (f"The probability measures y_score must be two-dimensional in the non-binary case. "
             f"Now it is {len(y_score.shape)}-dimensional.")

        y_true_binarized: np.ndarray = label_binarize(y_true, classes=np.arange(n_classes))

        if average == 'micro':
            fpr, tpr, _ = roc_curve(y_true_binarized.ravel(), y_score.ravel())    # fpr: np.ndarray; tpr: np.ndarray
            auroc: float = roc_auc_score(y_true, y_score, average='micro', multi_class='ovr')
        else:
            auroc: float = roc_auc_score(y_true, y_score, average='macro', multi_class='ovr')
            fpr_list: list[float] = []
            tpr_list: list[float] = []
            for i in range(n_classes):  # fpr_i: np.ndarray; tpr_i: np.ndarray
                fpr_i, tpr_i, _ = roc_curve(y_true_binarized[:, i], y_score[:, i])
                fpr_list.append(fpr_i)
                tpr_list.append(tpr_i)
            fpr: np.ndarray = np.linspace(0, 1, 1000)        # A relatively fine grid for interpolation
            tpr: np.ndarray = np.zeros_like(fpr)
            for i in range(n_classes):
                tpr += np.interp(fpr, fpr_list[i], tpr_list[i])
            tpr /= n_classes

    return {f'{prefix}AUROC': auroc,
            f'{prefix}AUPRC': average_precision_score(y_true, y_score),
            f'{prefix}TPR': list(tpr),
            f'{prefix}FPR': list(fpr)}

########################################################################################################################
# Classification metric (used with predicted probability measures): AIC, BIC, and negative log-likelihood (NLL).
########################################################################################################################


def classify_AIC_BIC(y_true: np.ndarray,
                     y_score: np.ndarray,
                     n_params: int,
                     bic_complexity_penalty: float = 1,
                     prefix: str = ""):
    """
    Compute AIC and BIC scores (and NLL) by comparing the true labels and the predicted probability measures (NOT the
    predicted labels). Notice that the number of free model parameters (n_params) is required for the calculation.

    :param y_true: A one-dimensional numpy array.
           Values of the true label.
    :param y_score: A one-dimensional (binary) or two-dimensional (multi-class) numpy array.
           Probability measures of each class.
    :param n_params: A positive integer.
           Number of free model parameters.
    :param prefix: A string.
           The prefix (e.g. 'Train_') of the metric name shown in the keys of the output dictionary.
           Default setting: prefix=""
    :return:
    A dictionary with keys as {prefix}{metric name} and values as results.
    """

    # Type and value check
    try:
        y_true = np.array(y_true)
    except TypeError:
        raise TypeError(f"y_true must be (convertible to) a numpy array. Now its type is {type(y_true)}.")
    try:
        y_score = np.array(y_score)
    except TypeError:
        raise TypeError(f"y_score must be (convertible to) a numpy array. Now its type is {type(y_score)}.")
    assert len(y_true.shape) == 1, \
        f"y_true must be one-dimensional. Now it is {len(y_true.shape)}-dimensional."
    assert len(y_score.shape) in [1, 2], \
        f"y_score must be one-/two-dimensional. Now it is {len(y_score.shape)}-dimensional."
    assert isinstance(prefix, str), \
        f"prefix must be a string. Now its type is {type(prefix)}."
    assert isinstance(n_params, int), \
        f'n_params must be an integer. Now its type is {type(n_params)}.'
    assert n_params >= 1, \
        f'n_params must be a positive integer. Now it is {n_params}.'
    n_classes = len(np.unique(y_true))
    if n_classes == 1:
        warnings.warn("Only one class found in the true label. Returning numpy.nan.")
        return {f'{prefix}AIC': np.nan,
                f'{prefix}BIC': np.nan,
                f'{prefix}NLL': np.nan}

    # Define sample size
    n_samples = len(y_true)

    # Remarks for calculation
    # AIC = -2 * LL + 2 * n_params, see https://en.wikipedia.org/wiki/Akaike_information_criterion
    # BIC = -2 * LL + ln(n_samples) * n_params, see https://en.wikipedia.org/wiki/Bayesian_information_criterion
    # where LL denotes the log-likelihood. In other words, we can compute AIC and BIC as follows:
    # AIC = 2 * NLL + 2 * n_params
    # BIC = 2 * NLL + ln(n_samples) * n_params
    # where NLL denotes the Negative Log-Likelihood.
    # Reference: https://towardsdatascience.com/cross-entropy-negative-log-likelihood-and-all-that-jazz-47a95bd2e81
    NLL: float = log_loss(y_true, y_score, normalize=False)
    AIC: float = (2 * NLL) + (n_params * 2)
    BIC: float = (2 * NLL) + (n_params * np.log(n_samples) * bic_complexity_penalty)
    return {f'{prefix}AIC': AIC,
            f'{prefix}BIC': BIC,
            f'{prefix}NLL': NLL}

########################################################################################################################
# End of script.
########################################################################################################################
