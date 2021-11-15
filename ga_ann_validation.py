#Copyright 2021 Nina de Lacy

   #Licensed under the Apache License, Version 2.0 (the "License");
   #you may not use this file except in compliance with the License.
   #You may obtain a copy of the License at

     #http://www.apache.org/licenses/LICENSE-2.0

   #Unless required by applicable law or agreed to in writing, software
   #distributed under the License is distributed on an "AS IS" BASIS,
   #WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   #See the License for the specific language governing permissions and
   #limitations under the License.


#eli5 permutation routine is released under the MIT license and may be found at https://eli5.readthedocs.io/en/latest/

#import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from eli5.permutation_importance import get_score_importances
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
import os
import time


start_time = time.time()

#select GPU if desired
with tf.device('device:GPU:0'):

    #read features    
    DLFS = pd.read_csv('yourpath')
     
    #read targets
    class_targets  = pd.read_csv('yourpath')    
    class_target_list = ['yourtarget']
    
    learn_test = DLFS
    
    #drop subjectID
    learn_test = learn_test.drop('Identifiers',axis=1)
    class_targets = class_targets.drop('Identifiers',axis=1)
    
    #set number of models you wish to evaluate
    accuracy_threshold = 100
    
    # define eli5 score importances
    def score(X,y):
      y_predict_classes = model.predict_classes(X)
      return accuracy_score(y, y_predict_classes)
  
    
    #test with bool threshold correl
    for target in class_target_list:
        
        #read in parameters and features
        parameters_basepath = 'yourpath'
        parameters_filename = parameters_basepath + target + '.csv'
        parameters = pd.read_csv(parameters_filename)
        parameters = parameters.drop(parameters.columns[0], axis=1)
        
        features_basepath = 'yourpath'
        features_filename = features_basepath + target + '.csv'
        features = pd.read_csv(features_filename)
        features = features.drop(features.columns[0], axis=1)
        
        #determine hyperparameters for top accuracy models
        accuracy_idx = np.argsort(parameters['accuracy'])
        top_accuracy_idx = accuracy_idx[:accuracy_threshold].tolist()
        learning_list = parameters['learning'].to_list()
        learning_select = [learning_list[i] for i in top_accuracy_idx]
        beta_1_list = parameters['beta_1'].to_list()
        beta_1_select = [beta_1_list[i] for i in top_accuracy_idx]
        beta_2_list = parameters['beta_2'].to_list()
        beta_2_select = [beta_2_list[i] for i in top_accuracy_idx]
        
        
        #determine features for top accuracy models
        names_list = []
        for col in features:
            ser = features[col].dropna()
            names = ser.unique()
            names = names.tolist()
            names_list.append(names)
        features_select = [names_list[i] for i in top_accuracy_idx]
          
        #create X,y
        X_df = learn_test
        y = class_targets[target]
        y = y.astype('int64')
        y = y.to_numpy()
        y = np.where(y==1,0,y)
        y = np.where(y==2,1,y)
        y_true = y
        y = tf.keras.utils.to_categorical(y, num_classes=2)
    
        #create lists for performance measures
        accuracy_list = []
        precision_list = []
        recall_list = []
        feature_importances_scores = []
        num_features_list = []
       
        # perform validation
        for feature, learning, beta_1, beta_2 in zip(features_select, learning_select, beta_1_select, beta_2_select):
                                                     
            
            X = X_df[feature]
            num_features = len(feature)
            num_features_list.append(num_features)
            X = np.asarray(X).astype(np.float32)
            
        
            model = Sequential()
            model.add(Dense(300, activation = 'relu', input_shape=(len(feature),)))
            model.add(Dense(300, activation = 'relu'))
            model.add(Dense(300, activation = 'relu'))
            model.add(Dense(2, activation = 'softmax'))
            opt = Adam(lr = learning, beta_1=beta_1, beta_2=beta_2)
            model.compile(loss="categorical_crossentropy",
                     optimizer = opt,
                     metrics=['accuracy'])
            early_stopping_monitor = EarlyStopping(monitor='loss',patience=3)
            model.fit(X,y, epochs=20, callbacks=[early_stopping_monitor])
            print("GPU 0 running validation testing for", target)
        
            y_predict_classes = model.predict_classes(X)
            base_score, score_decreases = get_score_importances(score, X,y_true)
            feature_importances = np.mean(score_decreases, axis=0)
            feature_importances = feature_importances.tolist()
            feature_importances_scores.append(feature_importances)
            accuracy = accuracy_score(y_true, y_predict_classes)
            accuracy_list.append(accuracy)
            precision = average_precision_score(y_true, y_predict_classes)
            precision_list.append(precision)
            recall = recall_score(y_true,y_predict_classes)
            recall_list.append(recall)
       
    
        learning_array = np.array(learning_select)
        beta_1_array = np.array(beta_1_select)
        beta_2_array = np.array(beta_2_select)
        recall_array = np.array(recall_list)
        precision_array = np.array(precision_list)
        accuracy_array = np.array(accuracy_list)
        num_features_array = np.array(num_features_list)
        
        print(target, "accuracy max:", accuracy_array.max())
        print(target, "recall max:", recall_array.max())
        print(target, "precision max:", precision_array.max())
        
        #collect performance measures, features and importances
        best_ga_ann_features = pd.DataFrame(data=features_select).T
        best_accuracy_models = np.stack((learning_select, beta_1_select, beta_2_select, accuracy_array, precision_array, recall_array, num_features_list), axis=1)
        best_ga_ann_models = pd.DataFrame(data=best_accuracy_models, columns = ['learning_rate', 'beta_1', 'beta_2', 'accuracy', 'precision', 'recall', 'num_features'])
        best_ga_ann_importances = pd.DataFrame(data=feature_importances_scores).T
        #best_ga_ann_importances = best_ga_ann_importances.T
    
    
        #get base path and name string for pickle
        basepath = "yourpath"
        basename = target
        
        #save features
        features_path = "ga_ann_classification_test_unseen_elbow_2SD_best_features_"
        features_path = os.path.join(basepath,features_path)
        features_filename = features_path + basename
        features_csv_filename = features_path + basename + '.csv'
        best_ga_ann_features.to_pickle(features_filename)
        best_ga_ann_features.to_csv(features_csv_filename)
        
        #save models
        models_path = "ga_ann_classification_test_unseen_elbow_2SD_best_models_"
        models_path = os.path.join(basepath,models_path)
        models_filename = models_path + basename
        models_csv_filename = models_path + basename + '.csv'
        best_ga_ann_models.to_pickle(models_filename)
        best_ga_ann_models.to_csv(models_csv_filename)
        
        #save importances
        importances_path = "ga_ann_classification_test_unseen_elbow_2SD_best_importances_"
        importances_path = os.path.join(basepath,importances_path)
        importances_filename = importances_path + basename
        importances_csv_filename = importances_path + basename + '.csv'
        best_ga_ann_importances.to_pickle(importances_filename)
        best_ga_ann_importances.to_csv(importances_csv_filename)
        
       
        
    
