from __future__ import print_function

from functools import partial

# hyper-opt
from hyperopt import hp, Trials, STATUS_OK, tpe, fmin
from hyperopt import space_eval
from hyperopt.pyll import scope

# from tensorflow.keras.layers import Dense, Dropout, Activation
# from tensorflow.keras.models import Sequential

# utils
import pickle
import argparse
import os
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
# import xgboost
from sklearn.metrics import accuracy_score, precision_score, recall_score


dataset_dict = {
    1 : 'adult_income'
}

data_dict = {
    'adult_income'      : ('adult_income', 'income')   
}  


space_RF = {
    'max_depth'         : scope.int(hp.uniform('max_depth', 1, 11)),
    'max_features'      : hp.choice('max_features', ['sqrt', 'log2', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    'n_estimators'      : scope.int(hp.qloguniform('n_estimators', np.log(9.5), np.log(300.5), 1)),
    'criterion'         : hp.choice('criterion', ["gini", "entropy"]),
    'min_samples_split' : hp.choice('min_samples_split', [2, 5, 10]),
    'min_samples_leaf'  : hp.choice('min_samples_leaf', [1, 2, 4]),
    'bootstrap'         : hp.choice('bootstrap', [True, False]),
}



def prepare_data(data, rseed, partition):
    sensitive_attribute = 'education_Doctorate'
    dataset, decision = data_dict[data]
    datadir = '{}/'.format(dataset)    

    #filenames
    suffix = 'OneHot'
    train_file      = '{}{}_train{}_{}.csv'.format(datadir, dataset, suffix, rseed)
    test_file       = '{}{}_test{}_{}.csv'.format(datadir, dataset, suffix, rseed)
    sg_file         = '{}{}_attack{}_{}.csv'.format(datadir, dataset, suffix, rseed)

    # load dataframe
    df_train    = pd.read_csv(train_file)
    df_train = np.array_split(df_train, 250)[partition]
    df_test  = pd.read_csv(test_file)
    df_test1 = df_test[df_test[sensitive_attribute] == 0]
    df_test2 = df_test[df_test[sensitive_attribute] == 1]
    
    
    df_sg = pd.read_csv(sg_file)

    # prepare the data
    scaler = StandardScaler()
    ## training set
    y_train = df_train[decision]
    X_train = df_train.drop(labels=[decision], axis = 1)
    X_train = X_train.drop(labels=[sensitive_attribute], axis=1)
    X_train = X_train.drop(labels=['education_HS-grad'], axis=1)
    X_train = X_train.drop(labels=['education_Masters'], axis=1)
    X_train = X_train.drop(labels=['education_Prof-school'], axis=1)
    X_train = X_train.drop(labels=['education_School'], axis=1)
    X_train = X_train.drop(labels=['education_Some-college'], axis=1)
    X_train = X_train.drop(labels=['education_Assoc'], axis=1)
    X_train = X_train.drop(labels=['education_Bachelors'], axis=1)
    X_train = scaler.fit_transform(X_train)
    ### cast
    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    
    
    
    ## test set
    y_test = df_test[decision]
    X_test = df_test.drop(labels=[decision], axis = 1)
    X_test = X_test.drop(labels=[sensitive_attribute], axis = 1)
    X_test = X_test.drop(labels=['education_HS-grad'], axis = 1)
    X_test = X_test.drop(labels=['education_Masters'], axis = 1)
    X_test = X_test.drop(labels=['education_Prof-school'], axis = 1)
    X_test = X_test.drop(labels=['education_School'], axis = 1)
    X_test = X_test.drop(labels=['education_Some-college'], axis = 1)
    X_test = X_test.drop(labels=['education_Assoc'], axis = 1)
    X_test = X_test.drop(labels=['education_Bachelors'], axis = 1)
    X_test = scaler.fit_transform(X_test)
    ### cast
    X_test = np.asarray(X_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)

    ## test set
    y_test1 = df_test1[decision]
    X_test1 = df_test1.drop(labels=[decision], axis = 1)
    X_test1 = X_test1.drop(labels=[sensitive_attribute], axis = 1)
    
    
    X_test1 = X_test1.drop(labels=['education_HS-grad'], axis = 1)
    X_test1 = X_test1.drop(labels=['education_Masters'], axis = 1)
    X_test1 = X_test1.drop(labels=['education_Prof-school'], axis = 1)
    X_test1 = X_test1.drop(labels=['education_School'], axis = 1)
    X_test1 = X_test1.drop(labels=['education_Some-college'], axis = 1)
    X_test1 = X_test1.drop(labels=['education_Assoc'], axis = 1)
    X_test1 = X_test1.drop(labels=['education_Bachelors'], axis = 1)
    X_test1 = scaler.fit_transform(X_test1)
    ### cast
    X_test1 = np.asarray(X_test1).astype(np.float32)
    y_test1 = np.asarray(y_test1).astype(np.float32)
    
    ## test set
    y_test2 = df_test2[decision]
    X_test2 = df_test2.drop(labels=[decision], axis = 1)
    X_test2 = X_test2.drop(labels=[sensitive_attribute], axis = 1)
    X_test2 = X_test2.drop(labels=['education_HS-grad'], axis = 1)
    X_test2 = X_test2.drop(labels=['education_Masters'], axis = 1)
    X_test2 = X_test2.drop(labels=['education_Prof-school'], axis = 1)
    X_test2 = X_test2.drop(labels=['education_School'], axis = 1)
    X_test2 = X_test2.drop(labels=['education_Some-college'], axis = 1)
    X_test2 = X_test2.drop(labels=['education_Assoc'], axis = 1)
    X_test2 = X_test2.drop(labels=['education_Bachelors'], axis = 1)
    X_test2 = scaler.fit_transform(X_test2)
    ### cast
    X_test2 = np.asarray(X_test2).astype(np.float32)
    y_test2 = np.asarray(y_test2).astype(np.float32)



    return X_train, y_train, X_test1, y_test1, X_test2, y_test2, X_test, y_test



best = 0
# Random Forest
def obj_func__RF(params, data, rseed, partition):
    

    def acc_model(params, data, rseed, partition):
        X_train, y_train, _, _,_,_,_,_ = prepare_data(data, rseed, partition)
        model = RandomForestClassifier(**params)

        result = cross_validate(model, X_train, y_train, return_estimator=True, n_jobs=5, verbose=2)
        
        idx_best_model = np.argmax(result['test_score'])

        score = np.mean(result['test_score'])
        best_model = result['estimator'][idx_best_model]

        return score, best_model

    global best
    acc, model = acc_model(params, data, rseed, partition)

    if acc > best:
        best = acc

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}




def binary_one_hot(i):
    if i == 0:
        return np.array([1, 0])
    return np.array([0, 1])
    


if __name__ == '__main__':
    np.random.seed(0)

    # parser initialization
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', type=int, default=0, help='which partition')

    # get input
    args = parser.parse_args()
    dataset = 'adult_income'
    rseed = 0
    model_class = 'RF'
    nbr_evals = 25
    partition = args.partition


    
    # Initialize an empty trials database
    trials = Trials()

    # Perform the evaluations on the search space
    obj_func__RF = partial(obj_func__RF, data=dataset, rseed=rseed, partition=partition)
    best = fmin(obj_func__RF, space_RF, algo=tpe.suggest, trials=trials, max_evals=nbr_evals)

    # get params of the best model
    best_params = space_eval(space_RF, best)
    print(best_params)

    # get the best model
    best_model = trials.best_trial['result']['model']

    # accuracy of the best model
    X_train, y_train, X_test1, y_test1, X_test2, y_test2, X_test, y_test = prepare_data(dataset, rseed, partition)
    acc_train   = accuracy_score(y_train, best_model.predict(X_train))
    print('Train Acc: ', acc_train)
    acc_test1    = accuracy_score(y_test1, best_model.predict(X_test1))
    hists = np.array([binary_one_hot(i) for i in best_model.predict(X_test1)])
    np.save('not_phd_'+str(partition), hists)
    print('Test1 Acc: ', acc_test1)
    acc_test2    = accuracy_score(y_test2, best_model.predict(X_test2))
    hists = np.array([binary_one_hot(i) for i in best_model.predict(X_test2)])
    np.save('phd_'+str(partition), hists)
    print('Test2 Acc: ', acc_test2)
    acc_test    = accuracy_score(y_test, best_model.predict(X_test))
    print('Test Acc: ', acc_test)
    print(best_model.predict(X_test))