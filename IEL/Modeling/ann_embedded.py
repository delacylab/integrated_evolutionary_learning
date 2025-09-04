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
# Overview: This script defines a function to create, train, and evaluate an ANN model through cross-validation.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.nn as nn
from .ann_base import ANN_Classifier, train_model, test_model, EarlyStopping
from .metrics import classify_discrimination_metrics, classify_AIC_BIC, classify_AUROC
from eli5.permutation_importance import get_score_importances
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

########################################################################################################################
# Define a function to perform k-fold cross-validation.
########################################################################################################################


def kfold_splitter(X_: np.ndarray,
                   y_: np.ndarray,
                   n_split: int = 5,
                   random_state: int = 42):
    """
    :param X_: A two-dimensional numpy array.
               Feature dataset.
    :param y_: A one-dimensional numpy array.
               Target dataset.
    :param n_split: Integer.
           Number of folds (k) used in cross-validation.
    :param random_state: Integer.
           Random seed used for stratification.
    :return: An generator object of the row indices in X_ and y_
    """
    # Type and value check are left to the sklearn implementation
    skf: StratifiedKFold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=random_state)
    return skf.split(X_, y_)

########################################################################################################################


def ANN_Classifier_Embedded(X: np.ndarray,
                            y: np.ndarray,
                            n_splits: int,
                            n_units: int,
                            n_classes: int,
                            n_layers: int,
                            n_epochs: int,
                            lr: float,
                            beta1: float,
                            beta2: float,
                            earlyStopper: EarlyStopping,
                            device: str = 'cuda:0'):
    """
    :param X: A 2-dimensional numpy array (or Torch.Tensor).
           The feature data with dimension of (sample size, number of features).
    :param y:  A 1-dimensional numpy array (or Torch.Tensor).
           The target 1-dimensional data with length as the sample size.
    :param n_split: Integer.
           Number of folds (k) used in cross-validation.
    :param n_units: A positive integer.
           The number of hidden units in each hidden layer.
    :param n_classes: A positive integer greater than 1.
           The number of classes in the target.
    :param n_layers: A positive integer.
           The number of hidden layers in the model.
    :param n_epochs: A positive integer.
           The number of maximum epochs to be run in training the model.
    :param lr: A float.
           The learning rate of the AdamW optimizer.
    :param beta1: A float.
           The beta1 value used in the AdamW optimizer.
    :param beta2: A float.
           The beta2 value used in the AdamW optimizer.
    :param earlyStopper: An EarlyStopping object or None.
    :param device: A string.
           The name of the device used for model training (e.g., 'cuda:0' as GPU and 'cpu' as CPU)
           Default setting: device='cuda:0'
    :return: A dictionary storing the performance statistics averaged across folds.
    """
    # Type and value check are left to the execution of ann_base.py

    ####################################################################################################################
    # Step 1: Create the splitter for k-fold cross-validation
    ####################################################################################################################
    splitter = kfold_splitter(X, y, n_splits)

    ####################################################################################################################
    # Step 2: In each fold, create the ANN_Classifier object
    ####################################################################################################################
    # Initializes variables to store results
    n_samples: int = X.shape[0]
    n_feat: int = X.shape[1]
    fit_results: list[dict] = []
    eli_importance_list: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter, 1):

        # Create the model
        M = ANN_Classifier(n_feat=n_feat, n_units=n_units, n_classes=n_classes, n_layers=n_layers)
        M.set_device(device)

        # Stratified the dataset using the defined splitter
        X_train: np.ndarray = np.take(X, train_idx, axis=0)
        X_val: np.ndarray = np.take(X, val_idx, axis=0)
        y_train: np.ndarray = np.take(y, train_idx, axis=0)
        y_val: np.ndarray = np.take(y, val_idx, axis=0)

        ################################################################################################################
        # Step 3: Fit the in-fold model
        ################################################################################################################
        fit_result: dict[str, float] = train_model(model=M, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                                                   n_epochs=n_epochs, criterion=nn.BCEWithLogitsLoss(),
                                                   optimizer=torch.optim.AdamW, earlyStopper=earlyStopper,
                                                   verbose_epoch=None, lr=lr, betas=(beta1, beta2))

        # Restart early stopper
        earlyStopper.reset() if earlyStopper is not None else None

        # Obtain training and validation performance statistics
        for (X_, y_, prefix) in [(X_train, y_train, 'Train_'), (X_val, y_val, 'Val_')]:
            y_pred: torch.Tensor = test_model(model=M, X=X_, y=y_, criterion=nn.BCEWithLogitsLoss(), prefix=prefix,
                                              return_pred=True)[1]                      # Predicted logits
            y_pred_prob: np.ndarray = torch.sigmoid(y_pred).cpu().detach().numpy()      # Predicted probabilities
            y_pred_label: np.ndarray = np.where(y_pred_prob >= 0.5, 1, 0)               # Predicted labels
            fit_result |= (classify_discrimination_metrics(y_true=y_, y_pred=y_pred_label, prefix=prefix)
                           | classify_AUROC(y_true=y_, y_score=y_pred_prob, prefix=prefix)
                           | classify_AIC_BIC(y_true=y_, y_score=y_pred_prob, n_params=X_.shape[1], prefix=prefix))

            # Compute eli5 permutation importance score
            if prefix == 'Val_':
                def acc_score(X_dummy, y_dummy):
                    y_pred_dummy: torch.Tensor = test_model(model=M, X=X_dummy, y=y_dummy,
                                                            criterion=nn.BCEWithLogitsLoss(),
                                                            return_pred=True)[1]
                    y_pred_prob_dummy: torch.Tensor = torch.sigmoid(y_pred_dummy).cpu().detach().numpy()
                    y_pred_label_dummy: torch.Tensor = np.where(y_pred_prob_dummy >= 0.5, 1, 0)
                    return accuracy_score(y_true=y_dummy, y_pred=y_pred_label_dummy)
                score_decreases: np.ndarray = get_score_importances(acc_score, X_, y_, random_state=42)[1]
                permutation_importance: list[float] = list(np.mean(score_decreases, axis=0))
                eli_importance_list.append(permutation_importance)

        # Concatenate performance statistics
        fit_result: dict[str, float] = {k: v for k, v in fit_result.items()
                                        if not isinstance(v, list) and 'loss' not in k}
        fit_results.append(fit_result)

    ####################################################################################################################
    # Step 4: Organize the performance statistics across folds
    ####################################################################################################################
    df_stat: pd.DataFrame = pd.DataFrame.from_records(fit_results)
    mean_row: dict[str, float] = df_stat.mean().to_dict()
    eli_importance_avg: list[float] = list(np.mean(np.array(eli_importance_list), axis=0))
    mean_row = {'#Features': n_feat, '#Samples': n_samples, 'lr': lr, 'beta1': beta1, 'beta2': beta2} | mean_row
    mean_row |= {'eli5_Importance': eli_importance_avg}
    mean_row['Elapsed_train_time'] = df_stat['Elapsed_train_time'].sum()
    return mean_row

########################################################################################################################
# End of script.
########################################################################################################################
