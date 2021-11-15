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
from imblearn.combine import SMOTEENN
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from eli5.permutation_importance import get_score_importances
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
import os
import time

tf.keras.backend.clear_session()

#test gpu env if desired
#assert tf.test.is_gpu_available()
#assert tf.test.is_built_with_cuda()

#limit GPU memory growth if desired
#physical_devices = tf.config.list_physical_devices('GPU')
#try:
#    tf.config.experimental.set_memory_growth(physical_devices[0], True)
#    tf.config.experimental.set_memory_growth(physical_devices[1], True)
#except:
#    pass

#select GPU if desired
with tf.device('device:GPU:0'):
    

    start_time = time.time()
      
    #read in features
    DLFS = pd.read_csv('yourpath')
    
    #read in targets and remove extra column names
    class_targets = pd.read_csv('yourpath')
    
    learn_train = DLFS
    
    #set ga parameters
    sol_per_pop = 100
    num_generations = 300
    FIFO_len = 30
    # num parents mating and num parents mut should be even numbers
    num_parents_mating = 40
    num_parents_mut = 20
    num_parents_rand = int(sol_per_pop - (num_parents_mating/2 + num_parents_mut))
    queue_len = num_parents_mating + num_parents_mut
    count_subjects = learn_train['yoursubjectID'].count()
    
    #drop subject ID col
    learn_train = learn_train.drop('yoursubjectID',axis=1)
    class_targets = class_targets.drop('yoursubjectID',axis=1)
    class_target_list = class_targets
    
    # define custom BIC function
    def BIC_pop_fitness(test, predict):
        resid = test - predict
        sse = sum(resid**2)
        sample_size = len(X_sample)
        num_params = len(features)
        BIC_fitness = (sample_size * np.log(sse/sample_size)) + (num_params * np.log(sample_size))
        #BIC_fitness = int(BIC_fitness)
        return BIC_fitness
    
    # define eli5 score importances
    def score(X,y):
      y_predict_classes = model.predict_classes(X)
      return accuracy_score(y, y_predict_classes)
    
    for target in class_target_list:
    # define x and y, format for ann initialization 
      X = learn_train
      y = class_targets[target]
    
      sm = SMOTEENN(sampling_strategy = 1, random_state=42)
      X,y = sm.fit_resample(X, y)
      X=X.fillna(0)
      y = y.astype('int64')
      y = y.to_numpy()
      y = np.where(y==1,0,y)
      y = np.where(y==2,1,y)
    
      #set arrays of # features, learning rate, beta_1, beta_2 for ann initialization
      # you may change hyperparameters or their ranges
      rand_pop = np.random.randint(1,high=50, size=sol_per_pop).tolist()
      learning_pop = np.random.uniform(low=0.00001, high=0.01, size=sol_per_pop).tolist()
      beta_1_pop = np.random.uniform(low=0.9, high=0.999, size=sol_per_pop).tolist()
      beta_2_pop = np.random.uniform(low=0.9, high=0.999, size=sol_per_pop).tolist()
    
      #initialize ann
      feature_list = []
      fitness_list = []
      feature_importances_list = []
      for rand, learning, beta_1, beta_2 in zip(rand_pop, learning_pop, beta_1_pop, beta_2_pop):
        X_sample = X.sample(n=rand, axis=1)
        features = X_sample.columns
        feature_list.append(features)
        X_sample = np.asarray(X_sample).astype(np.float32)
        
        k = np.floor(len(X_sample)/len(features)).astype('int64')
        k = np.where(k>10, 10, k).astype('int64').item()
        k = np.where(k > (np.sum(y == 1)/k), np.floor(np.sum(y == 1)/k), k).astype('int64').item()
        k = np.where(k > (np.sum(y == 0)/k), np.floor(np.sum(y == 0)/k), k).astype('int64').item()
        k = np.where(k<2, 2, k).astype('int64').item()
        skf = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)
    
        fitness_scores = []
        feature_importances_scores = []
        for train,test in skf.split(X_sample,y):
          
          X_train = X_sample[train]
          X_test = X_sample[test]
          y_train = tf.keras.utils.to_categorical(y[train], num_classes=2)
          y_test = tf.keras.utils.to_categorical(y[test], num_classes=2)
          model = Sequential()
          model.add(Dense(300, activation = 'relu', input_shape=(len(features),)))
          model.add(Dense(300, activation = 'relu'))
          model.add(Dense(300, activation = 'relu'))
          model.add(Dense(2, activation = 'softmax'))
          opt = Adam(lr = learning, beta_1=beta_1, beta_2=beta_2)
          model.compile(loss="categorical_crossentropy",
                   optimizer = opt,
                   metrics=['accuracy'])
          early_stopping_monitor = EarlyStopping(monitor='val_loss',patience=3)
          model.fit(X_train,y_train, epochs=20, validation_data = (X_test, y_test), callbacks=[early_stopping_monitor])
          print("GPU 0 initializing ann")
    
          y_predict_classes = model.predict_classes(X_sample[test])
          fitness = BIC_pop_fitness(y[test], y_predict_classes)
          fitness_scores.append(fitness)
          base_score, score_decreases = get_score_importances(score, X_sample[test],y[test])
          feature_importances = np.mean(score_decreases, axis=0)
          feature_importances_scores.append(feature_importances)
    
        #compute mean fitness from k folds
        mean_fitness = np.mean(fitness_scores)
        fitness_list.append(mean_fitness)
        feature_importances_scores_folds=pd.DataFrame(data=feature_importances_scores) 
        feature_importances_scores_mean = []
        for column in feature_importances_scores_folds:
            col_mean = feature_importances_scores_folds[column].mean()
            feature_importances_scores_mean.append(col_mean)
        feature_importances_list.append(feature_importances_scores_mean)
      #determine fitness and feature arrays for all models in initialization
      ann_features = np.asarray(feature_list)
      ann_fitness = np.asarray(fitness_list)
    
      print("first classification initialized for:", target)
      tf.keras.backend.clear_session()
    
      #initialize queue
      init_fitness_sorted = np.sort(ann_fitness)
      #reduces to length of queue
      init_top_fitness = init_fitness_sorted[:queue_len]
      init_top_fitness_round = np.round(init_top_fitness)
      queue = init_top_fitness_round.reshape(queue_len,1)
    
      #initialize ga while loop and collection lists
      generation=1
      is_converged = False
    
      #empty lists to collect best model in each generation based on fitness function
      best_fitness_list = []
      best_feature_list = []
      best_num_feature_list = []
      best_learning_list = []
      best_beta_1_list = []
      best_beta_2_list = []
      best_accuracy_list = []
      best_precision_list = []
      best_recall_list = []
      best_importances_list = []
    
      #initialize for loop for genetic algorithm
      while generation in range(num_generations) and is_converged == False:
        #generate parameter arrays for mating parents
        fitness_idx = np.argsort(ann_fitness)
        fitness_mating_idx = fitness_idx[:num_parents_mating].tolist()
        fitness_mating = [ann_fitness[i] for i in fitness_mating_idx]
        learning_mating = [learning_pop[i] for i in fitness_mating_idx]
        beta_1_mating = [beta_1_pop[i] for i in fitness_mating_idx]
        beta_2_mating = [beta_2_pop[i] for i in fitness_mating_idx]
    
        bisect = np.array(num_parents_mating/2, dtype=int)
    
        # generate children by crossover at pivot point
        # learning rate children
        learning_first = learning_mating[:bisect]
        learning_second = learning_mating[bisect:]
        learning_mate_child = np.add(learning_first, learning_second)/2
        learning_mate_child = np.where(learning_mate_child<0.00001, 0.00001, learning_mate_child)
        learning_mate_child = learning_mate_child.tolist()
        #beta_1 children
        beta_1_first = beta_1_mating[:bisect]
        beta_1_second = beta_1_mating[bisect:]
        beta_1_mate_child = np.add(beta_1_first, beta_1_second)/2
        beta_1_mate_child = np.where(beta_1_mate_child<0.9, 0.9, beta_1_mate_child)
        beta_1_mate_child = np.where(beta_1_mate_child>0.999, 0.999, beta_1_mate_child)
        beta_1_mate_child = beta_1_mate_child.tolist()
        #beta_2 children
        beta_2_first = beta_2_mating[:bisect]
        beta_2_second = beta_2_mating[bisect:]
        beta_2_mate_child = np.add(beta_2_first, beta_2_second)/2
        beta_2_mate_child = np.where(beta_2_mate_child<0.9, 0.9, beta_1_mate_child)
        beta_2_mate_child = np.where(beta_2_mate_child>0.999, 0.999, beta_1_mate_child)
        beta_2_mate_child = beta_2_mate_child.tolist()
    
        #determine mating features: top half of parent models to pivot point
        fitness_for_features_mating_idx = fitness_idx[:bisect]
        features_mate_child = [ann_features[i] for i in fitness_for_features_mating_idx]
    
        #determine remaining population after num_parents_mating removed from each array
        #note that features remainder is larger than parameters remainder since only half the number of parents mating are removed
        fitness_remainder = np.delete(ann_fitness, fitness_mating_idx)
        fitness_remainder_for_features = np.delete(ann_fitness,fitness_for_features_mating_idx)
        features_remainder = np.delete(ann_features, fitness_for_features_mating_idx)
        learning_remainder = np.delete(learning_pop, fitness_mating_idx)
        beta_1_remainder = np.delete(beta_1_pop, fitness_mating_idx)
        beta_2_remainder = np.delete(beta_2_pop, fitness_mating_idx)
    
        #determine arrays for parameters and features for mutation from remainders
        # note that features mutating are drawn from higher in the remainder queue (which is longer)
        fitness_idx = np.argsort(fitness_remainder)
        fitness_mut_idx = fitness_idx[:num_parents_mut]
        fitness_mut = fitness_remainder[fitness_mut_idx]
        learning_mut = learning_remainder[fitness_mut_idx]
        beta_1_mut = beta_1_remainder[fitness_mut_idx]
        beta_2_mut = beta_2_remainder[fitness_mut_idx]
    
        #add mutations for learning rate to num_parents_mut by splitting and shifting half 0.01 to right (+) and half 0.01 to left (-)
        #set pivot point
        bisect = np.array(num_parents_mut/2, dtype=int)
        #add mutations to learning rate by splitting and shifting half 0.0001 to right (+) and half 0.0001 to left (-)
        learning_left = learning_mut[:bisect,]
        learning_left_child = learning_left + 0.0001
        learning_right = learning_mut[bisect:,]
        learning_right_child = learning_right - 0.0001
        learning_mut_child = np.append(learning_right_child, learning_left_child)
        learning_mut_child = np.where(learning_mut_child<0.00001, 0.00001, learning_mut_child)
        learning_mut_child = learning_mut_child.tolist()
    
        #add mutations for beta_1 to num_parents_mut by splitting and shifting half 0.001 to right (+) and half 0.001 to left (-)
        beta_1_left = beta_1_mut[:bisect,]
        beta_1_left_child = beta_1_left + 0.001
        beta_1_right = beta_1_mut[bisect:,]
        beta_1_right_child = beta_1_right - 0.001
        beta_1_mut_child = np.append(beta_1_right_child, beta_1_left_child)
        #add floor/ceiling to learning_mut_child
        beta_1_mut_child = np.where(beta_1_mut_child<0.9, 0.9, beta_1_mut_child)
        beta_1_mut_child = np.where(beta_1_mut_child>0.999, 0.999, beta_1_mut_child)
        beta_1_mut_child = beta_1_mut_child.tolist()
    
        #add mutations for beta_2 to num_parents_mut by splitting and shifting half 0.001 to right (+) and half 0.001 to left (-)
        beta_2_left = beta_2_mut[:bisect,]
        beta_2_left_child = beta_2_left + 0.001
        beta_2_right = beta_2_mut[bisect:,]
        beta_2_right_child = beta_2_right - 0.001
        beta_2_mut_child = np.append(beta_2_right_child, beta_2_left_child)
        beta_2_mut_child = np.where(beta_2_mut_child<0.9, 0.9, beta_2_mut_child)
        beta_2_mut_child = np.where(beta_2_mut_child>0.999, 0.999, beta_2_mut_child)
        beta_2_mut_child = beta_2_mut_child.tolist()
    
        #determine features mut child
        features_idx = np.argsort(fitness_remainder_for_features)
        features_mut_idx = features_idx[:num_parents_mut].tolist()
        features_mut_child = [features_remainder[i] for i in features_mut_idx]
    
        #collect mated and mutated children
        new_learning_child = learning_mate_child + learning_mut_child
        new_beta_1_child = beta_1_mate_child + beta_1_mut_child
        new_beta_2_child = beta_2_mate_child + beta_2_mut_child
        new_features_child = features_mate_child + features_mut_child
    
        #add new random parents
        learning_rand_child = np.random.uniform(low=0.00001, high=0.01, size = num_parents_rand).tolist()
        beta_1_rand_child = np.random.uniform(low=0.9, high=0.999, size=num_parents_rand).tolist()
        beta_2_rand_child = np.random.uniform(low=0.9, high=0.999, size=num_parents_rand).tolist()
        #rand_array for features selection now sized to num_parents_rand
        new_rand_array = np.random.randint(2,high=len(X.columns), size=num_parents_rand)
        features_for_X_array = []
        for rand in new_rand_array:
            feature_names = X.sample(n=rand,axis=1).columns
            features_for_X_array.append(feature_names)
        features_rand_child = features_for_X_array
    
        #create new population of mated, mutated and random children
        learning_pop = new_learning_child + learning_rand_child
        beta_1_pop = new_beta_1_child + beta_1_rand_child
        beta_2_pop = new_beta_2_child + beta_2_rand_child
        features_pop = new_features_child + features_rand_child 
    
        #set up lists to collect metrics and features from CV 
        fitness_list = []
        accuracy_list = []
        precision_list = []
        recall_list = []
        feature_list = []
        feature_importances_list = []
        num_features_list = []
    
        # run ANN and generate n+1 fitness array of shape (sol_per_pop,) 
        for feature, learning, beta_1, beta_2 in zip(features_pop, learning_pop, beta_1_pop, beta_2_pop):
          tf.keras.backend.clear_session()
          X_sample = X[feature]
          features = X_sample.columns
          feature_list.append(features)
          num_features = len(features)
          num_features_list.append(num_features)
          X_sample = np.asarray(X_sample).astype(np.float32)
          
          k = np.floor(len(X_sample)/len(features)).astype('int64')
          k = np.where(k>10, 10, k).astype('int64').item()
          k = np.where(k > (np.sum(y == 1)/k), np.floor(np.sum(y == 1)/k), k).astype('int64').item()
          k = np.where(k > (np.sum(y == 0)/k), np.floor(np.sum(y == 0)/k), k).astype('int64').item()
          k = np.where(k<2, 2, k).astype('int64').item()
          skf = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)
          
          accuracy_scores = []
          precision_scores = []
          recall_scores = []
          fitness_scores = []
          feature_importances_scores = []
          for train,test in skf.split(X_sample,y):
            X_train = X_sample[train]
            X_test = X_sample[test]
            y_train = tf.keras.utils.to_categorical(y[train], num_classes=2)
            y_test = tf.keras.utils.to_categorical(y[test], num_classes=2)
            model = Sequential()
            model.add(Dense(300, activation = 'relu', input_shape=(len(features),)))
            model.add(Dense(300, activation = 'relu'))
            model.add(Dense(300, activation = 'relu'))
            model.add(Dense(2, activation = 'softmax'))
            opt = Adam(lr = learning, beta_1=beta_1, beta_2=beta_2)
            model.compile(loss="categorical_crossentropy",
                     optimizer = opt,
                     metrics=['accuracy'])
            early_stopping_monitor = EarlyStopping(monitor='val_loss',patience=3)
            model.fit(X_train,y_train, epochs=20, validation_data = (X_test, y_test), callbacks=[early_stopping_monitor])
            print("GPU 0 computing resample ann generation:", generation, "for", target)
            
            y_predict_classes = model.predict_classes(X_sample[test])
            accuracy = accuracy_score(y[test], y_predict_classes)
            accuracy_scores.append(accuracy)
            precision = average_precision_score(y[test], y_predict_classes)
            precision_scores.append(precision)
            recall = recall_score(y[test],y_predict_classes)
            recall_scores.append(recall)
            fitness = BIC_pop_fitness(y[test], y_predict_classes)
            fitness_scores.append(fitness)
            base_score, score_decreases = get_score_importances(score, X_sample[test],y[test])
            feature_importances = np.mean(score_decreases, axis=0)
            feature_importances_scores.append(feature_importances)
          #compute mean metrics from k folds and append metric collection lists 
          mean_accuracy = np.mean(accuracy_scores)
          accuracy_list.append(mean_accuracy)
          mean_precision = np.mean(precision_scores)
          precision_list.append(mean_precision)
          mean_recall = np.mean(recall_scores)
          recall_list.append(mean_recall)
          mean_fitness = np.mean(fitness_scores)
          fitness_list.append(mean_fitness)
          feature_importances_scores_folds=pd.DataFrame(data=feature_importances_scores) 
          feature_importances_scores_mean = []
          for column in feature_importances_scores_folds:
            col_mean = feature_importances_scores_folds[column].mean()
            feature_importances_scores_mean.append(col_mean)
          feature_importances_list.append(feature_importances_scores_mean)
    
        ann_fitness = fitness_list
        ann_features = feature_list
        
        
    
        # pull out top models based on fitness and its parameters and join to repository of best fitness models in every generation
        best_fitness_idx = np.argsort(ann_fitness)
        best_fitness_idx = best_fitness_idx[:3].tolist()
        best_fitness = [ann_fitness[i] for i in best_fitness_idx]
        best_fitness_list = best_fitness_list + best_fitness
        best_learning = [learning_pop[i] for i in best_fitness_idx]
        best_learning_list= best_learning_list + best_learning
        best_beta_1 = [beta_1_pop[i] for i in best_fitness_idx]
        best_beta_1_list= best_beta_1_list + best_beta_1
        best_beta_2 = [beta_2_pop[i] for i in best_fitness_idx]
        best_beta_2_list= best_beta_2_list + best_beta_2
        best_accuracy = [accuracy_list[i] for i in best_fitness_idx]
        best_accuracy_list = best_accuracy_list + best_accuracy
        best_precision = [precision_list[i] for i in best_fitness_idx]
        best_precision_list = best_precision_list + best_precision
        best_recall = [recall_list[i] for i in best_fitness_idx]
        best_recall_list = best_recall_list + best_recall
        best_features = [ann_features[i] for i in best_fitness_idx]
        best_feature_list = best_feature_list + best_features
        best_num_features = [num_features_list[i] for i in best_fitness_idx]
        best_num_feature_list = best_num_feature_list + best_num_features
        best_importances = [feature_importances_list[i] for i in best_fitness_idx]
        best_importances_list = best_importances_list + best_importances
    
    
        # queue
        # pull out top fitness models and join to queue
        top_fitness_idx = np.argsort(ann_fitness) 
        top_fitness_idx = top_fitness_idx[:queue_len]
        fitness_round = np.round(ann_fitness)
        top_fitness = fitness_round[top_fitness_idx]
        top_fitness = top_fitness.reshape(queue_len,1)
        queue = np.hstack((queue, top_fitness))
    
        #set condition for convergence
        if queue.shape < (queue_len, FIFO_len):
          print("working")
          generation += 1
        else:
          roll_std_sum = np.sum(np.diff(np.std(queue,axis=0)))
          roll_min_sum = np.sum(np.diff(np.min(queue,axis=0)))
          if roll_std_sum not in np.arange(-0.01,0.01) or roll_min_sum not in np.arange(-0.01,0.01):
              queue = np.delete(queue,1,1)
              #print(queue)
              print("deleting")
              generation += 1
          else:
              is_converged = True
              print("converged")
    
      #collect best models from each generation based on fitness
    
      best_ann_ga_features = pd.DataFrame(data=best_feature_list).T
      best_ann_ga_importances = pd.DataFrame(data=best_importances_list).T
      best_fitness_models = np.stack((best_fitness_list, best_learning_list, best_beta_1_list, best_beta_2_list, best_accuracy_list, best_precision_list, best_recall_list, best_num_feature_list), axis=1)
      best_ann_ga_models = pd.DataFrame(data=best_fitness_models, columns = ['fitness', 'learning', 'beta_1', 'beta_2', 'accuracy', 'precision', 'recall', 'num_features'])
    
      #create filepaths
      basepath = 'your_output_path'
      basename = target
      features_path = 'classification_features_'
      full_features_path = os.path.join(basepath,features_path)
      features_filename = full_features_path + basename + '.csv'
      models_path = 'classification_models_'
      full_models_path = os.path.join(basepath,models_path)
      models_filename = full_models_path + basename + '.csv'
      importances_path = 'classification_importances_'
      full_importances_path = os.path.join(basepath,importances_path)
      importances_filename = full_importances_path + basename + '.csv'
      
      #save output
      
      best_ann_ga_features.to_csv(features_filename)
      best_ann_ga_models.to_csv(models_filename)
      best_ann_ga_importances.to_csv(importances_filename)
      
        
    
    
      print("end routine for", target)
      print("time elapsed is:", np.round((time.time() - start_time)/60), "minutes")
