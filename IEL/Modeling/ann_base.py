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
# Overview: This script defines the building blocks of creating, training, and evaluating a feedforward Artificial
# Neural Network (ANN).
########################################################################################################################

########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import torch
from time import time
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from typing import Callable, Optional, Union

########################################################################################################################
# Define a callable class for loss functions in PyTorch.
########################################################################################################################
LossFunction: Callable = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

########################################################################################################################
# Define the early stopping technique used in model training as a class.
########################################################################################################################


class EarlyStopping:
    """
    Early stopping technique used during model training to speed up runtime and prevent over-fitting.

    A. Runtime parameters
    ---------------------
    A1. patience: A non-negative integer.
        Number of times that a worse result (depending on A2) can be tolerated.
        Default setting: patience=1
    A2. min_delta: A non-negative float.
        The threshold where loss_old + min_delta < loss_new is considered as worse.
        Default setting: min_delta=1e-8

    B. Attributes
    -------------
    B1. min_loss: A float, recording the minimum loss encountered in the training process.
    B2. counter: A non-negative integer, recording the number of times a worse result is observed. The counter restarts
                 when a better result (i.e., smaller loss) is observed.
    B3. best_state_dict: A dictionary, recording the model parameters of the best model observed so far.
    (A1-A2 are initialized as instance attributes.)

    C. Methods
    ---------
    C1. refresh()
        Reset the attributes B1 and B2 to their original values.
    C2. early_stop(loss)
    :param loss: An integer or float. The loss obtained in a given training epoch.
    :return: A boolean indicating whether the training process should be stopped.
    """
    def __init__(self,
                 patience: int = 1,
                 min_delta: float = 1e-8):

        # Type and value check
        assert isinstance(patience, int), \
            f"patience must be an integer. Now its type is {type(patience)}."
        assert patience >= 0, \
            f"patience must be a non-negative integer. Now it is {patience}."
        self.patience: int = patience
        try:
            min_delta = float(min_delta)
        except TypeError:
            raise TypeError(f"min_delta must be a float. Now its type is {type(min_delta)}.")
        assert min_delta >= 0, \
            f"min_delta must be a non-negative float. Now it is {min_delta}."
        self.min_delta: float = min_delta
        self.min_loss: float = float('inf')
        self.counter: int = 0
        self.best_state_dict: dict = None

    def reset(self):
        self.min_loss = float('inf')
        self.counter = 0
        self.best_state_dict = None

    def early_stop(self, loss: float, model=None):
        try:
            loss: float = float(loss)
        except TypeError:
            raise TypeError(f"loss must be a float. Now its type is {type(loss)}.")
        if loss < self.min_loss:
            self.min_loss, self.counter = loss, 0
            if model is not None:
                self.best_state_dict = copy.deepcopy(model.state_dict())
        elif loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

########################################################################################################################
# Define a superclass (inherited from torch.nn.Module) for ANNs for model specification.
########################################################################################################################


class DL_Class(nn.Module):
    """
    A superclass for various model classes that will be used in this module.

    A. Runtime parameters
    ---------------------
    (None)

    B. Attributes
    -------------
    B1. dummy_param: A torch.nn.Parameter object.
        An identifier of the physical location of the model.

    C. Methods
    ----------
    C1. set_device(device_str)
        :param device_str: A string or torch.device object. The physical location of the model to be set.
    C2. get_device()
        :return: A string. The current physical location of the model.
    C3. get_n_params()
        :return: An integer. The number of parameters of the model.
    """
    def __init__(self):
        super().__init__()
        self.dummy_param: nn.Parameter = nn.Parameter(torch.empty(0))

    def set_device(self, device_str: Union[str, torch.device]):
        assert type(device_str) in [str, torch.device], \
            f'device_str must be a torch.device object or a string. Now its type is {type(device_str)}.'
        self.to(device_str)

    def get_device(self):
        return self.dummy_param.device

    def get_n_params(self):
        return sum(p.numel() for p in self.parameters())        # For AIC & BIC calculation if needed

########################################################################################################################
# Define an ANN model class for classification tasks.
########################################################################################################################


class ANN_Classifier(DL_Class):
    """
    A PyTorch ANN class for classification.

    A. Runtime parameters
    ---------------------
    A1. n_feat: A positive integer.
        The number of features to be fitted to the model.
    A2. n_units: A positive integer.
        The number of hidden units in each hidden layer.
    A3. n_classes: A positive integer greater than 1.
        The number of classes in the target.
    A4. n_layers: A positive integer.
        The number of hidden layers in the model.
        Default setting: n_layers=2

    B. Attributes
    -------------
    B1. model: A torch.nn.modules.container.Sequential object.
        The actual model.
    (A1-A4 are initialized as instance attributes.)

    C. Methods
    ----------
    C1. forward(X)
        :param X: A 2-dimensional numpy array (or Torch.Tensor) with rows as samples and columns as features.
        :return: A torch.Tensor of dimension (samples size, ) [for binary case] or (sample size, n_classes) [for
                 multiclass case] as the probability measure returned by the model.
    C2. init_Xavier_weights()
        Assign weights with Xavier uniform distribution and bias as 0.
        See https://paperswithcode.com/method/xavier-initialization
    """
    def __init__(self,
                 n_feat: int,
                 n_units: int,
                 n_classes: int,
                 n_layers: int = 2):
        super().__init__()

        # Type and value check
        assert isinstance(n_feat, int), \
            f"n_feat must be a positive integer. Now its type is {type(n_feat)}."
        assert n_feat >= 1, \
            f"n_feat must be a positive integer. Now its value is {n_feat}."
        self.n_feat = n_feat
        assert isinstance(n_units, int), \
            f"n_units must be a positive integer. Now its type is {type(n_units)}."
        assert n_units >= 1, \
            f"n_units must be a positive integer. Now its value is {n_units}."
        self.n_units = n_units
        assert isinstance(n_classes, int), \
            f"n_classes must be a positive integer. Now its type is {type(n_classes)}."
        assert n_classes >= 2, \
            f"n_classes must be a positive integer not less than 2. Now its value is {n_classes}."
        self.n_classes = n_classes
        assert isinstance(n_layers, int), \
            f"n_layers must be a positive integer. Now its type is {type(n_layers)}."
        assert n_layers >= 1, \
            f"n_layers must be a positive integer. Now its value is {n_layers}."
        self.n_layers: int = n_layers
        self.__init_Xavier_weights()

        # Create model
        modules: list[nn.Module] = [nn.Linear(n_feat, n_units), nn.ReLU()]
        for _ in range(n_layers-1):
            modules += [nn.Linear(n_units, n_units), nn.ReLU()]
        modules += [nn.Linear(n_units, 1 if n_classes == 2 else n_classes)]
        self.model: nn.Sequential = nn.Sequential(*modules)

    def __init_Xavier_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, X: Union[np.ndarray, torch.Tensor]):
        try:
            X: torch.Tensor = torch.Tensor(X)
        except TypeError:
            raise TypeError(f'X must be (convertible to) a torch.Tensor. Now its type is {type(X)}.')
        assert len(X.shape) == 2, \
            f'X must be 2-dimensional. Now it is {len(X.shape)}-dimensional.'
        device: str = self.get_device()
        X = X.to(device)
        output: torch.Tensor = self.model(X)
        return torch.reshape(output, shape=(-1,)) if self.n_classes == 2 else output

########################################################################################################################
# Define the function to train an ANN model.
########################################################################################################################


def train_model(model: ANN_Classifier,
                X_train: torch.Tensor,
                y_train: torch.Tensor,
                X_val: torch.Tensor,
                y_val: torch.Tensor,
                n_epochs: int,
                criterion: LossFunction,
                optimizer: torch.optim.Optimizer,
                earlyStopper: Optional[EarlyStopping] = None,
                verbose_epoch: Optional[int] = None,
                **kwargs):
    """
    :param model: An ANN_Classifier object.
    :param X_train: A 2-dimensional numpy array (or Torch.Tensor).
           The training feature data with dimension of (sample size, number of features).
    :param y_train: A 1-dimensional numpy array (or Torch.Tensor).
           The training target 1-dimensional data with length as the sample size.
    :param X_val: A 2-dimensional numpy array (or Torch.Tensor).
           The validation feature data with dimension of (sample size, number of features).
    :param y_val: A 1-dimensional numpy array (or Torch.Tensor).
           The validation target 1-dimensional data with length as the sample size.
    :param n_epochs: A positive integer.
           The number of maximum epochs to be run in training the model.
    :param criterion: A loss function from torch.nn.
           See https://pytorch.org/docs/main/nn.html#loss-functions. Make sure you specify it WITH brackets.
           Example: torch.nn.CrossEntropyLoss()
    :param optimizer: An torch.optim.Optimizer object.
           See https://pytorch.org/docs/stable/optim.html. Make sure you specify it WITHOUT brackets.
           Example: torch.nn.AdamW
    :param earlyStopper: An EarlyStopping object or None.
           It aims to speed up training time and prevent over-fitting.
           Default setting: earlyStopper=None
    :param verbose_epoch: A positive integer or None.
           If integer, it controls the frequency of printing the training and validation losses with a rate of every
           {verbose_epoch} epochs. No logging will be printed if None.
           Default setting: verbose_epoch=None
    :param kwargs: (Any extra runtime parameters of optimizer)
           Example: lr=0.001 for the learning rate parameter of the optimizer.
    :return:
    A dictionary of the following four pairs of items:
    - 'Elapsed_train_time': the elapsed time of training the model
    - 'Elapsed_train_epochs': the elapsed epochs of training the model
    - 'Train_loss': the training loss
    - 'Val_loss': the validation loss
    """

    # Type and value check
    assert isinstance(model, ANN_Classifier), \
        f'model must be an ANN_Classifier object. Now its type is {type(model)}.'
    device: str = model.get_device()
    try:
        X_train: torch.Tensor = torch.Tensor(X_train).to(device)
    except TypeError:
        raise TypeError(f'X_train must be (convertible to) a torch.tensor. Now its type is {type(X_train)}.')
    try:
        y_train: torch.Tensor = torch.Tensor(y_train).to(device)
    except TypeError:
        raise TypeError(f'y_train must be (convertible to) a torch.tensor. Now its type is {type(y_train)}.')
    try:
        X_val: torch.Tensor = torch.Tensor(X_val).to(device)
    except TypeError:
        raise TypeError(f'X_val must be (convertible to) a torch.tensor. Now its type is {type(X_val)}.')
    try:
        y_val: torch.Tensor = torch.Tensor(y_val).to(device)
    except TypeError:
        raise TypeError(f'y_val must be (convertible to) a torch.tensor. Now its type is {type(y_val)}.')
    assert len(X_train.shape) == 2, \
        f'X_train must be 2-dimensional. Now its dimension is {X_train.shape}'
    assert len(X_val.shape) == 2, \
        f'X_val must be 2-dimensional. Now its dimension is {X_val.shape}'
    assert len(y_train.shape) == 1, \
        f'y_train must be one-dimensional. Now its dimension is {y_train.shape}.'
    assert len(y_val.shape) == 1, \
        f'y_val must be one-dimensional. Now its dimension is {y_val.shape}.'
    assert isinstance(n_epochs, int), \
        f"n_epochs must be a positive integer. Now its type is {type(n_epochs)}."
    assert n_epochs >= 1, \
        f"n_epochs must be a positive integer. Now it is {n_epochs}."
    # No type check for criterion and optimizer because PyTorch did not define the associated class.
    if earlyStopper is not None:
        assert isinstance(earlyStopper, EarlyStopping), \
            f"earlyStopper (if not None) must be a EarlyStopping object. Now its type is {type(earlyStopper)}."
    if verbose_epoch is not None:
        assert isinstance(verbose_epoch, int), \
            f"verbose_epoch (if not None) must be a positive integer. Now its type is {type(verbose_epoch)}."
        assert n_epochs >= verbose_epoch >= 1, \
            f"verbose_epoch must be in the range [1, n_epochs]. Now its value is {n_epochs}."

    # Specify the optimizer using keyword arguments
    opt: torch.optim.Optimizer = optimizer(model.parameters(), **kwargs)

    # Initialize loss values and epoch counter
    train_loss: float = None
    val_loss: float = None
    elapsed_epochs: int = 0

    # Training starts
    train_start: float = time()
    for epoch_idx in range(n_epochs):
        model.train()
        opt.zero_grad()
        y_train_pred: torch.Tensor = model(X_train)
        try:
            train_loss: float = criterion(y_train_pred, y_train)
        except RuntimeError:  # CrossEntropyLoss and NLLLoss require LongTensor
            train_loss: float = criterion(y_train_pred, y_train.long())

        # Backward propagation
        train_loss.backward()
        opt.step()
        model.eval()
        elapsed_epochs += 1

        # Validation
        with torch.no_grad():
            y_val_pred: torch.Tensor = model(X_val)
            try:
                val_loss: float = criterion(y_val_pred, y_val)
            except RuntimeError:  # CrossEntropyLoss and NLLLoss require LongTensor
                val_loss = criterion(y_val_pred, y_val.long())
            if verbose_epoch is not None and (epoch_idx+1) % verbose_epoch == 0:
                print(f"Epochs: {epoch_idx+1:4}/{n_epochs:>4}; Training loss = {train_loss:.4f}; "
                      f"Validation loss = {val_loss:.4f}", flush=True)

        # Early stopping
        if earlyStopper is not None:
            if earlyStopper.early_stop(val_loss):
                break

    # Training ends
    train_end: float = time()

    # Return results
    return {'Elapsed_train_time': train_end - train_start, 'Elapsed_train_epochs': elapsed_epochs,
            'Train_loss': train_loss.item(), 'Val_loss': val_loss.item()}

########################################################################################################################
# Define the function to evaluate an ANN model.
########################################################################################################################


def test_model(model: ANN_Classifier,
               X: torch.Tensor,
               y: torch.Tensor,
               criterion: LossFunction,
               prefix: str = 'Test_',
               return_pred: bool = False):
    """
    :param model: An ANN_Classifier object.
    :param X: A 2-dimensional numpy array (or Torch.Tensor).
           The test feature data with dimension of (sample size, number of features).
    :param y:  A 1-dimensional numpy array (or Torch.Tensor).
           The test target 1-dimensional data with length as the sample size.
    :param criterion: A loss function from torch.nn.
           See https://pytorch.org/docs/main/nn.html#loss-functions. Make sure you specify it WITH brackets.
           Example: torch.nn.CrossEntropyLoss()
    :param prefix: A string.
           The prefix (e.g. 'Test_') of the metric name shown in the keys of the output dictionary.
           Default setting: prefix='Test_'
    :param return_pred: A boolean.
           Return the predicted labels as well if True, or only the loss (as a dictionary) otherwise.
           Default setting: return_pred=False
    :return:
    (a) A dictionary with key as '{prefix}loss' and value as the loss value.
    If return_pred=True:
    (b) The predicted labels or probability measures as a torch.tensor.
    """

    # Type and value check
    assert isinstance(model, ANN_Classifier), \
        f'model must be an ANN_Classifier object. Now its type is {type(model)}.'
    model.eval()
    device = model.get_device()
    try:
        X = torch.Tensor(X).to(device)
    except TypeError:
        raise TypeError(f'X must be (convertible to) a torch.tensor. Now its type is {type(X)}.')
    try:
        y = torch.Tensor(y).to(device)
    except TypeError:
        raise TypeError(f'y must be (convertible to) a torch.tensor. Now its type is {type(y)}.')
    assert len(X.shape) == 2, \
        f'X must be 2-dimensional. Now its dimension is {X.shape}'
    assert len(y.shape) == 1, \
        f'y must be one-dimensional. Now its dimension is {y.shape}.'
    assert isinstance(prefix, str), \
        f"prefix must be a string. Now its type is {type(prefix)}."
    assert isinstance(return_pred, bool), \
        f"return_pred must be a boolean. Now its type is {type(return_pred)}."

    # Evaluation starts
    start: float = time()

    with torch.no_grad():
        y_pred: torch.Tensor = model(X)
        try:
            test_loss: float = criterion(y_pred, y)
        except RuntimeError:  # CrossEntropyLoss and NLLLoss require LongTensor
            test_loss: float = criterion(y_pred, y.long())

    # Evaluation ends
    end = time()

    # Return results
    if return_pred:
        return {f'{prefix}loss': test_loss.item(), f'{prefix}time': end-start}, y_pred
    else:
        return {f'{prefix}loss': test_loss.item(), f'{prefix}time': end-start}

########################################################################################################################
# End of script.
########################################################################################################################
