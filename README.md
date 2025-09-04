<div align="right">
  Last update: 2025 September 4, 15:58 MT (by Wayne Lam)
</div>
<hr>

# :crystal_ball: Integrated Evolutionary Learning #
Neural networks have been criticized for being a "black box" of sorts, where the features that may be critical in the network's learning and evaluating functions are not clear to the user and are hidden in the network's weights. Our novel approach Integrated Evolutionary Learning (IEL) provides an automated method for simultaneously accomplishing principled feature selection and hyperparameter tuning while furnishing interpretable models where the original features used to make predictions may be obtained and ranked in order of importance, even in deep learning with artificial neural networks. In this approach, the machine learning algorithm of choice is nested inside an evolutionary algorithm which selects features and hyperparameters over generations on the basis of an information function to converge on an optimal solution.

# :paperclip: Characteristics # 
* Adopt the [AdamW optimizer](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) for stable back-propagation in deep-learning 
* Hyperparameter optimization through an evolutionary mechanism built within IEL
* Built-in feature selection for improvements in time and predictive performance
* Automated modeling pipelines for classification tasks
* Explainable AI methods empowered by SHAP value computation
* Strong performance on risk stratification in tested datasets

# :computer: Installation #

We provide a beta version of IEL for testing to ensure usability. 

**Using git and pip** 
```
pip install git+https://github.com/delacylab/integrated_evolutionary_learning.git
```

# :page_with_curl: Sample Modeling Script #

```python
from IEL.Modeling.feature_subset import feature_subsetter
from IEL.Modeling.ielclassifier import IELClassifier
from IEL.Modeling.ann_refit import refit_and_eval
from IEL.Utils.data_simulator import simulate
from IEL.Utils.plot_IEL_trend import plot_stat_trend
from IEL.Utils.plot_SHAP import plot_shap_beeswarm

# Step 1. Create a partitioned synthetic dataset
X_train, X_test, y_train, y_test = simulate()

# Step 2. IEL modeling
M1 = IELClassifier()
M1.fit(X_train, y_train, metric='AUROC')
result_1 = M1.get_performance()

# Step 3. Visualize IEL learning progress across generations
plot_stat_trend(result_1) 

# Step 4. Perform additional feature selection
feature_subset = feature_subsetter(result_1)
X_train_sub, X_test_sub = X_train[: feature_subset], X_test[: feature_subset]

# Step 5. Another round of IEL modeling with the warm-started feature subset
M2 = IELClassifier()
M2.fit(X_train_sub, y_train_sub, metric='AUROC')
result_2 = M2.get_performance()

# Step 6. Refit a feedforward neural network with the best-fitting configuration
result_final, shap = refit_and_eval(X_train_sub, X_test_sub, y_train, y_test, result_2)

# Step 7. Visualize the feature importance using SHAP values
final_feature_idxs = result_final['Feature_Indices'].to_list()[0]
X_test_final = X_test_sub[:, final_feature_idxs]
plot_shap_beeswarm(shap=shap, X=X_test_final)
```
- `Step 1` generates a synthetic partitioned dataset for binary classification. 
- `Step 2` is the core IEL modeling step to identify the best-fitting hyperparameters (i.e., learning rate, β<sub>1</sub>, and β<sub>2</sub>) and feature subset with a user-defined performance metric. 
- `Step 3` visualizes the learning trend of IEL throughout the executed generations (see `Image 1` below), allowing users to analyze whether a longer evolutionary process is required.
- `Step 4-5` are optional steps to retrain another IEL model with a warm-started feature subset. To identify the subset, we first compute the optimal number of features (_knee_, see `Image 2` below) and then identify the top-knee features ordered by their permutation importance scores.
- `Step 6` utilizes the best-fitting hyperparameters and feature subset identified by IEL, and retrains a new feedforward neural network with the optimized configuration.
- `Step 7` uses the re-fitted model in the last step to visualize feature importance scores represented by SHAP values to capture both group-level and individual-level feature contribution.

| `Image 1` | `Image 2` | `Image 3`|
|---------|---------|---------|
| <img width="300" height="200" alt="trend" src="https://github.com/user-attachments/assets/681c7cb2-b071-43fa-8c67-2ec066a8c10c" /> | <img width="300" height="200" alt="knee" src="https://github.com/user-attachments/assets/4323badf-4738-454a-9c2f-e992f4dec9fd" /> | <img width="300" height="200" alt="shap" src="https://github.com/user-attachments/assets/1bb43a22-9186-422e-b18f-6faf59e9c68c" /> |

Check the __Google Colab Notebook__ for an example of the modeling pipeline with finer controls over the runtime parameters.

[![IEL Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18scyVDA3VtMxgG9wDV5cxV39OO8gRYIP?usp=sharing)

# ⚙️ Data Preprocessing Pipelines (Optional) #

In the manuscript "Predicting the onset of internalizing disorders in early adolescence using deep learning optimized with AI" (under review), we proposed a data preprocessing pipeline before performing IEL model fitting. This pipeline includes methods that remove variables with too many missing values, partition data for training/test purposes, winsorize and scale data to avoid potential graident explosion/vanishing problems, impute data to ensure feasibility of modeling, and select feature subsets to enhance model runtime performance and parismony.  

This section delineates the preprocessing pipeline with executable Python scripts (stored in `IEL/Preprocessing/`). While this README only provides high-level descriptions, users are recommended to consult each script's _docstring_ for comprehensive explanations and the _Test run_ section for usage examples.  

| | Script | Class/Function | Description | Used on |
|---------|---------|---------|---------|---------|
|1.|`P1_NaN_Thresholding.py`|`remove_null`|Remove variables with a missing value percentage above a user-defined threshold.| Full |
|2.|`P2_Partitioning.py`|`partitioning`|Split a dataset into training and test sets for model fitting and evaluation.| Full |
|3.|`P3_Sample_Balancing.py`|`sample_balancing`|Balance a binary target variable, optionally matching samples by a secondary variable (e.g., gender).| Training & test, independently| 
|4.|`P4_Winsorizing.py`|`winsorize`|Clip continuous/ordinal variables to a user-defined range of values.| Training & test, independently| 
|5.|`P5_Scaling.py`|`minMaxScale`|Scale variables to a specified range (e.g., [0, 1]) using min-max normalization.| Training & test, independently| 
|6.|`P6_Imputation.py`|`impute_nnmf` & `impute_mice`|Perform imputation via Non-Negative Matrix Factorization (NNMF), with both sklearn and a 5x faster custom PyTorch version, and via Multiple Imputation by Chained Equations (MICE).| Training & test, independently| 
|7.|`P7_Feature_Filtering.py`|`feat_filter`|Filter out features with a zero test statistic with the target from three different statistical tests: (a) mutual information (for all features), (b) chi-squared (for binary features), and (c) ANOVA (for continuous and ordinal features).| Training only |
|8.|`P8_LASSO_Feature_Selection_IEL`|`LASSO_IEL_Classifier`|Use L1-regularized logistic regression to perform feature selection. Since the regression problem is sensitive to the L1-regularization term α, we adopt IEL to optimize the value of α.| Training only |
|9.|`P9_Boruta_Feature_Selection.py`|`BorutaClass`|Implement Boruta, an ensemble-based feature selection method using random forests. This version supports multiple hyperparameter configurations and is optimized over the original [BorutaPy implementation](https://github.com/scikit-learn-contrib/boruta_py).| Training only |

The __Used on__ column above indicates when to apply the class/function. Starting with the removal of features with too many missing values in step 1 on the full feature dataset, we partition it into a training set and a test set in step 2, then apply steps 3-6 to the training and test sets separately. Subsequently, we apply step 7 to subset the training feature dataset X<sub>train</sub> in step 7, and apply steps 8 and 9 independently on X<sub>train</sub>. In the manuscript, we identify the union of the features selected by steps 8 and 9 to form the preprocessed feature dataset to reduce the risk of neglecting purely linear/nonlinear features and multicollinearity. Finally, the test feature dataset is filtered with the same feature subset in the preprocessed feature dataset for subseqeunt model evaluation. 

<!-- # :book: References #
Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. In _Proceedings of the International Conference on Learning Representations (ICLR)_. [https://arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101).-->


# :globe_with_meridians: License #
This project is licensed under the Apache License, Version 2.0. See the [LICENSE](./LICENSE) file for details.

