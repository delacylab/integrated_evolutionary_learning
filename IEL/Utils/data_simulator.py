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
# Overview: This script defines a function to generate a toy dataset for binary classification.
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

########################################################################################################################
# Define a function to generate a toy dataset for binary classification.
########################################################################################################################


def simulate(n_samples: int = 1000,
             n_features: int = 100,
             n_informative: int = 20,
             n_redundant: int = 10,
             train_ratio: float = 0.7):
    np.random.seed(42)
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=n_informative, n_redundant=n_redundant,
                               n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=42,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

########################################################################################################################
# End of script.
########################################################################################################################
