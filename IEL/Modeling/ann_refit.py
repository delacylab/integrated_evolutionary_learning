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
# Overview: This script refits an ANN model with the hyperparameters optimized by the IEL algorithm.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import pandas as pd
import torch
from .ann_base import ANN_Classifier, train_model, test_model, EarlyStopping
from .metrics import classify_discrimination_metrics, classify_AIC_BIC, classify_AUROC
from ast import literal_eval
from captum.attr import GradientShap
from typing import Literal, Optional

########################################################################################################################
# Define a wrapper class for sigmoid computation.
########################################################################################################################


class SigmoidWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.sigmoid(self.model(x))  # returns probability

########################################################################################################################
# Define a function to refit an ANN model with the hyperparameters optimized by the IEL algorithm and evaluate its
# performance at the test partition.
########################################################################################################################


def refit_and_eval(X_train: np.ndarray,
                   X_test: np.ndarray,
                   y_train: np.ndarray,
                   y_test: np.ndarray,
                   df_result: pd.DataFrame,
                   n_units: int = 300,
                   n_classes: int = 2,
                   n_layers: int = 3,
                   n_epochs: int = 100,
                   earlyStopper: Optional[EarlyStopping] = EarlyStopping(patience=3),
                   metric: Literal['Accuracy', 'AUROC', 'F1', 'NPV', 'Precision', 'Recall', 'Specificity',
                                   'AIC', 'BIC', 'NLL'] = 'AUROC',
                   maximize: bool = True):
    """
    :param X_train: A 2-dimensional numpy array.
           The training feature data with dimension of (training sample size, number of features).
    :param X_test: A 2-dimensional numpy array.
           The test feature data with dimension of (test sample size, number of features).
    :param y_train: A 1-dimensional numpy array.
           The training target 1-dimensional data with length as the training sample size.
    :param y_test: A 1-dimensional numpy array.
           The test target 1-dimensional data with length as the test sample size.
    :param df_result: A pandas.DataFrame.
           The result obtained from IEL.get_performance().
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
    :return:
    (a) df_fit_result: pandas.DataFrame. A single-row pandas.DataFrame storing the performance statistics of the
        refitted model.
    (b) shap_matrix: numpy.ndarray. A data matrix storing the SHAP values of the model (computed through
        captum.GradientSHAP) with the same dimension as X_test subset to the best-performing set of features identified
        in df_result using metric.
    """
    ####################################################################################################################
    # Step 1. Identify the best performing model according to the user-defined metric and its hyperparameter settings.
    ####################################################################################################################
    df_best: pd.DataFrame = df_result.iloc[df_result[f'Val_{metric}'].idxmax()] if maximize \
        else df_result.iloc[df_result[f'Val_{metric}'].idxmin()]
    feat_idxs: list[int] = literal_eval(df_best['Feature_Indices'])
    lr: float = df_best['lr']
    beta1: float = df_best['beta1']
    beta2: float = df_best['beta2']

    ####################################################################################################################
    # Step 2. Prepare the feature datasets.
    ####################################################################################################################
    X_train_sub: np.ndarray = X_train[:, feat_idxs]
    X_test_sub: np.ndarray = X_test[:, feat_idxs]

    ####################################################################################################################
    # Step 3. Create an ANN model.
    ####################################################################################################################
    M = ANN_Classifier(n_feat=X_train_sub.shape[1], n_units=n_units, n_classes=n_classes, n_layers=n_layers)
    M.set_device('cuda:0')

    ####################################################################################################################
    # Step 4. Train the ANN model.
    ####################################################################################################################
    train_result: dict[str, float] = train_model(model=M, X_train=X_train_sub, y_train=y_train,
                                                 X_val=X_train_sub, y_val=y_train,
                                                 n_epochs=n_epochs, criterion=torch.nn.BCEWithLogitsLoss(),
                                                 optimizer=torch.optim.AdamW,
                                                 earlyStopper=earlyStopper, verbose_epoch=5,
                                                 lr=lr, betas=(beta1, beta2))

    ####################################################################################################################
    # Step 5. Obtain training and test statistics.
    ####################################################################################################################
    fit_result: dict[str, float] = {'Feature_Indices': feat_idxs, '#Features': X_train_sub.shape[0],
                                    'lr': lr, 'beta1': beta1, 'beta2': beta2,
                                    'Elapsed_train_time': train_result['Elapsed_train_time'],
                                    'Elapsed_train_epochs': train_result['Elapsed_train_epochs'],
                                    'Train_sample_size': X_train_sub.shape[1],
                                    'Test_sample_size': X_test_sub.shape[1]}
    for (X_, y_, prefix) in [(X_train_sub, y_train, 'Train_'), (X_test_sub, y_test, 'Test_')]:
        y_pred: torch.Tensor = test_model(M, X_, y_, criterion=torch.nn.BCEWithLogitsLoss(), prefix=prefix,
                                          return_pred=True)[1]                      # Predicted logits
        y_pred_prob: np.ndarray = torch.sigmoid(y_pred).cpu().detach().numpy()      # Predicted probabilities
        y_pred_label: np.ndarray = np.where(y_pred_prob >= 0.5, 1, 0)               # Predicted labels
        fit_result |= (classify_discrimination_metrics(y_true=y_, y_pred=y_pred_label, prefix=prefix)
                       | classify_AUROC(y_true=y_, y_score=y_pred_prob, prefix=prefix)
                       | classify_AIC_BIC(y_true=y_, y_score=y_pred_prob, n_params=X_.shape[1], prefix=prefix))
    df_fit_result: pd.DataFrame = pd.DataFrame([fit_result])

    ####################################################################################################################
    # Step 6. Compute SHAP values.
    ####################################################################################################################
    W = SigmoidWrapper(M)
    shap_matrix: np.ndarray = (GradientShap(W).attribute(inputs=torch.tensor(X_test_sub, dtype=torch.float32),
                                                         baselines=torch.tensor(X_train_sub, dtype=torch.float32))
                               .cpu().detach().numpy())

    ####################################################################################################################
    # Step 7. Return df_fit_result and shap_matrix.
    ####################################################################################################################
    return df_fit_result, shap_matrix

########################################################################################################################
# End of script.
########################################################################################################################
