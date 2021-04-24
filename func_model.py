from func_data import data_preprocessor

import numpy as np
import pandas as pd
from scipy.stats import loguniform

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import BayesianRidge, RidgeClassifier, SGDClassifier, LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.svm import SVC, NuSVC, SVR
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, DecisionTreeRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, accuracy_score, log_loss, classification_report, f1_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, GroupShuffleSplit
from sklearn.inspection import permutation_importance
from eli5.sklearn import PermutationImportance

from xgboost import XGBClassifier, XGBRegressor
from fbprophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

from env_stocktrading import StockTradingEnv

tscv = TimeSeriesSplit(n_splits=4)

# Define classifiers and parameters
classifiers = {}
classifiers.update({"LR": LogisticRegression(solver='liblinear')})
classifiers.update({"LDA": LinearDiscriminantAnalysis()})
classifiers.update({"QDA": QuadraticDiscriminantAnalysis()})
classifiers.update({"AdaBoost": AdaBoostClassifier()})
classifiers.update({"Bagging": BaggingClassifier()})
classifiers.update({"ETE": ExtraTreesClassifier()})
classifiers.update({"GB": GradientBoostingClassifier()})
classifiers.update({"RF": RandomForestClassifier()})
classifiers.update({"RidgeC": RidgeClassifier()})
classifiers.update({"SGD": SGDClassifier()})
classifiers.update({"BNB": BernoulliNB()})
classifiers.update({"GNB": GaussianNB()})
classifiers.update({"KNN": KNeighborsClassifier()})
classifiers.update({"MLP": MLPClassifier()})
classifiers.update({"NuSVC": NuSVC(probability=True,kernel='rbf',nu=0.01)})
classifiers.update({"SVC": SVC(C=0.025, probability=True)})
classifiers.update({"DTC": DecisionTreeClassifier()})
classifiers.update({"ETC": ExtraTreeClassifier()})
classifiers.update({"XGB": XGBClassifier()})

parameters = {}
# Must connect each parameter to the named step in your pipeline with a double underscore __.
parameters.update({"LR": {"classifier__C": [0.1, 0.5, 1, 5, 10, 50, 80, 100],
                         }})
parameters.update({"LDA": {"classifier__solver": ["svd"],
                          }})    
parameters.update({"QDA": {"classifier__reg_param":[0.01*ii for ii in range(0, 101)],
                          }})
parameters.update({"AdaBoost": { 
                                "classifier__base_estimator": [DecisionTreeClassifier(max_depth = ii) for ii in range(1,6)],
                                "classifier__n_estimators": [200],
                                "classifier__learning_rate": [0.001, 0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 1.0],
                                 }})
parameters.update({"Bagging": { 
                                "classifier__base_estimator": [DecisionTreeClassifier(max_depth = ii) for ii in range(1,6)],
                                "classifier__n_estimators": [200],
                                "classifier__max_features": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                "classifier__n_jobs": [-1],
                                }})
parameters.update({"GB": { 
                                        "classifier__learning_rate":[0.15,0.1,0.05,0.01,0.005,0.001], 
                                        "classifier__n_estimators": [100,200,500,1000],
                                        "classifier__max_depth": [2,3,4,5,6,7,8],
                                        "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                        "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                        "classifier__max_features": ["auto", "sqrt", "log2"],
                                        "classifier__subsample": [0.8, 0.9, 1],
                                         }})
parameters.update({"ETE": { 
                                            "classifier__n_estimators": [100,200,500,1000],
                                            "classifier__class_weight": [None, "balanced"],
                                            "classifier__max_features": ["auto", "sqrt", "log2"],
                                            "classifier__max_depth" : [3, 5, 7, 10, 15, 18, 20, 30, 50],
                                            "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                            "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                            "classifier__criterion" :["gini", "entropy"],
                                            "classifier__n_jobs": [-1],
                                             }})
parameters.update({"RF": { 
                                    "classifier__n_estimators": [100,200,500,1000],
                                    "classifier__class_weight": [None, "balanced"],
                                    "classifier__max_features": ["auto", "sqrt", "log2"],
                                    "classifier__max_depth" : [3, 5, 7, 10, 15, 20, 30],
                                    "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                    "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                    "classifier__criterion" :["gini", "entropy"],
                                    "classifier__n_jobs": [-1],
                                     }})
parameters.update({"RidgeC": { 
                            "classifier__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0],
                             }})
parameters.update({"SGD": { 
                            "classifier__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0],
                            "classifier__penalty": ["l1", "l2"],
                            "classifier__n_jobs": [-1],
                            "classifier__loss": ['log'],
                             }})
parameters.update({"BNB": { 
                            "classifier__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0],
                             }})
parameters.update({"GNB": { 
                            "classifier__var_smoothing": [1e-9, 1e-8,1e-7, 1e-6, 1e-5],
                             }})
parameters.update({"KNN": { 
                            "classifier__n_neighbors": list(range(1,31)),
                            "classifier__p": [1, 2, 3, 4, 5],
                            "classifier__leaf_size": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                            "classifier__n_jobs": [-1],
                            "classifier__weights": ['uniform', 'distance'],
                            "classifier__metric": ['euclidean', 'manhattan'],
                             }})
parameters.update({"MLP": { 
                            "classifier__hidden_layer_sizes": [(5), (10), (5,5), (10,10), (5,5,5), (10,10,10), (15,15,15)],
                            "classifier__activation": ["identity", "logistic", "tanh", "relu"],
                            "classifier__learning_rate": ["constant", "invscaling", "adaptive"],
                            "classifier__max_iter": [100, 200, 300, 500, 1000, 2000],
                            "classifier__alpha": list(10.0 ** -np.arange(1, 10)),
                             }})
parameters.update({"NuSVC": { 
                            "classifier__nu": [0.25, 0.50, 0.75],
                            "classifier__kernel": ["linear", "rbf", "poly"],
                            "classifier__degree": [1,2,3,4,5,6],
                            "classifier__probability": [True],
                             }})
parameters.update({"SVC": { 
                            "classifier__kernel": ["linear", "rbf", "poly"],
                            "classifier__gamma": ["auto",0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                            "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                            "classifier__degree": [1, 2, 3, 4, 5, 6],
                             }})
parameters.update({"DTC": { 
                            "classifier__criterion" :["gini", "entropy"],
                            "classifier__splitter": ["best", "random"],
                            "classifier__class_weight": [None, "balanced"],
                            "classifier__max_features": ["auto", "sqrt", "log2"],
                            "classifier__max_depth" : [1,2,3, 4, 5, 6, 7, 8],
                            "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                            "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                             }})
parameters.update({"ETC": { 
                            "classifier__criterion" :["gini", "entropy"],
                            "classifier__splitter": ["best", "random"],
                            "classifier__class_weight": [None, "balanced"],
                            "classifier__max_features": ["auto", "sqrt", "log2"],
                            "classifier__max_depth" : [1,2,3, 4, 5, 6, 7, 8],
                            "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                            "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                             }})
parameters.update({'XGB': {
                            "classifier__min_child_weight": [1, 5, 10],
                            "classifier__gamma": [0.5, 1, 1.5, 2, 5],
                            "classifier__subsample": [0.6, 0.8, 1.0],
                            "classifier__colsample_bytree": [0.6, 0.8, 1.0],
                            "classifier__max_depth": [3, 4, 5, 6, 7, 8],
                            }})


# Define regressors and parameters
regressors = {}
regressors.update({"GBR": GradientBoostingRegressor()})
regressors.update({"XGBR": XGBRegressor()})
regressors.update({"Linear": LinearRegression()})
regressors.update({"Ridge": Ridge()})
regressors.update({"KNNR": KNeighborsRegressor()})
regressors.update({"DTR": DecisionTreeRegressor()})
#regressors.update({"ElasticNet": ElasticNet()})
#regressors.update({"SVR": SVR()})

reg_parameters = {}
# Must connect each parameter to the named step in your pipeline with a double underscore __.
reg_parameters.update({'GBR': {'regressor__learning_rate': [0.01,0.02,0.03,0.04],
                              'regressor__subsample'    : [0.9, 0.5, 0.2, 0.1],
                              'regressor__n_estimators' : [100,500,1000, 1500],
                              'regressor__max_depth'    : [4,6,8,10]
                              }})
reg_parameters.update({'XGBR': {'regressor__min_child_weight': [4,5],
                              'regressor__gamma': [i/10.0 for i in range(3,6)],
                              'regressor__subsample' : [i/10.0 for i in range(6,11)],
                              'regressor__colsample_bytree': [i/10.0 for i in range(6,11)],
                              'regressor__max_depth': [2,3,4]
                              }})
reg_parameters.update({'Linear': {'regressor__fit_intercept': [True, False],
                              'regressor__normalize': [True, False]
                              }})
reg_parameters.update({'Ridge': {'regressor__solver': ['svd', 'cholesky', 'lsqr', 'sag'],
                              'regressor__alpha': loguniform(1e-5, 100),
                              'regressor__fit_intercept': [True, False],
                              'regressor__normalize': [True, False]
                              }})
reg_parameters.update({'ElasticNet': {'regressor__alpha': loguniform(1e-5, 100),
                                      'l1_ratio': [0, 0.25, 0.5, 0.75, 1]
                                      }})
reg_parameters.update({"KNNR": { 
                            "regressor__n_neighbors": list(range(1,31)),
                            "regressor__p": [1, 2, 3, 4, 5],
                            "regressor__leaf_size": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                            "regressor__n_jobs": [-1],
                            "regressor__weights": ['uniform', 'distance'],
                            "regressor__metric": ['euclidean', 'manhattan'],
                             }})
reg_parameters.update({"DTR": { 
                            "regressor__criterion" :["mse", "friedman_mse", "mae", "poisson"],
                            "regressor__splitter": ["best", "random"],
                            "regressor__max_features": ["auto", "sqrt", "log2"],
                            "regressor__max_depth" : [1,2,3, 4, 5, 6, 7, 8],
                            "regressor__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                            "regressor__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                             }})
reg_parameters.update({"SVR": { 
                            "regressor__kernel": ["linear", "rbf", "poly"],
                            "regressor__gamma": ["auto",0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                            "regressor__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                            "regressor__degree": [1, 2, 3, 4, 5, 6],
                             }})

# Preliminary Model Selection
def clas_model_selection(X_train, y_train, X_test, y_test):
    table = pd.DataFrame(columns=['Model','Test Score', 'Test F1', 'Train Score', 'Train F1'])
    preprocessor = data_preprocessor(X_train)
    for classifier in classifiers.keys():
        pipe = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', classifiers[classifier])])
        pipe.fit(X_train, y_train)
        y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)
        
        table = table.append({'Model': classifier,
                              'Test Score': balanced_accuracy_score(y_test, y_test_pred),
                              'Test F1': f1_score(y_test, y_test_pred, average='weighted'), 
                              'Train Score': balanced_accuracy_score(y_train, y_train_pred),
                              'Train F1': f1_score(y_train, y_train_pred, average='weighted'),}, ignore_index=True) 
    
    train_scores = pd.pivot_table(table, index='Model', values='Train Score')
    train_f1s = pd.pivot_table(table, index='Model', values='Train F1')
    test_scores = pd.pivot_table(table, index='Model', values='Test Score')
    test_f1s = pd.pivot_table(table, index='Model', values='Test F1')
    
    result = pd.merge(test_scores, test_f1s, left_index=True, right_index=True)
    result = pd.merge(result, train_scores, left_index=True, right_index=True)
    result = pd.merge(result, train_f1s, left_index=True, right_index=True)
    result.sort_values(['Test F1','Test Score','Train F1','Train Score'],inplace=True,ascending=False)
    print(result)
    return result.index[:10]


def reg_model_selection(X_train, y_train, X_test, y_test):
    table = pd.DataFrame(columns=['Model', 'Test Error', 'Train Error'])
    preprocessor = data_preprocessor(X_train)
    for regressor in regressors.keys():
        pipe = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', regressors[regressor])])
        pipe.fit(X_train, y_train)
        y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)
        
        table = table.append({'Model': regressor,
                              'Test Error': mean_squared_error(y_test, y_test_pred), 
                              'Train Error': mean_squared_error(y_train, y_train_pred),}, ignore_index=True) 
    
    train_scores = pd.pivot_table(table, index='Model', values='Train Error')
    test_scores = pd.pivot_table(table, index='Model', values='Test Error')
    
    result = pd.merge(test_scores, train_scores, left_index=True, right_index=True)
    result.sort_values(['Test Error','Train Error'],inplace=True,ascending=True)
    print(result)
    return result.index[:5]


# Randomized Search
def clas_rs(X_train, y_train, X_test, y_test, chosen):    
    table = pd.DataFrame(columns=['Model','Test Score', 'CV Score'])
    preprocessor = data_preprocessor(X_train)
    for classifier in chosen:
        print('=============',classifier, 'Randomized Search', '=============')
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', classifiers[classifier])])
        param_grid = parameters[classifier]
        RS = RandomizedSearchCV(model, param_grid, cv=tscv, random_state=0,
                                n_iter=50, scoring='balanced_accuracy', n_jobs=-1)
        RS.fit(X_train, y_train)  
        y_test_pred = RS.predict(X_test)
        print('Best Parameters:',RS.best_params_)
        
        table = table.append({'Model': classifier,
                              'Test Score': balanced_accuracy_score(y_test, y_test_pred), 
                              'CV Score': RS.best_score_}, ignore_index=True) 
    
    CV_scores = pd.pivot_table(table, index='Model', values='CV Score')
    test_scores = pd.pivot_table(table, index='Model', values='Test Score')
    
    result = pd.merge(test_scores, CV_scores, left_index=True, right_index=True)
    result.sort_values(['Test Score','CV Score'],inplace=True,ascending=False)
    print(result[:5])


def reg_rs(X_train, y_train, X_test, y_test, chosen):    
    table = pd.DataFrame(columns=['Model','Test Error', 'Train Error'])
    preprocessor = data_preprocessor(X_train)
    for regressor in chosen:
        print('=============',regressor, 'Randomized Search', '=============')
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', regressors[regressor])])
        param_grid = reg_parameters[regressor]
        RS = RandomizedSearchCV(model, param_grid, cv=tscv, random_state=0,
                                n_iter=50, scoring='neg_mean_squared_error', n_jobs=-1)
        RS.fit(X_train, y_train)  
        y_test_pred = RS.predict(X_test)
        y_train_pred = RS.predict(X_train)
        print('Best Parameters:',RS.best_params_)
        
        table = table.append({'Model': regressor,
                              'Test Error': mean_squared_error(y_test, y_test_pred), 
                              'Train Error': mean_squared_error(y_train, y_train_pred)}, ignore_index=True) 
    
    train_scores = pd.pivot_table(table, index='Model', values='Train Error')
    test_scores = pd.pivot_table(table, index='Model', values='Test Error')
    
    result = pd.merge(test_scores, train_scores, left_index=True, right_index=True)
    result.sort_values(['Test Error','Train Error'], inplace=True, ascending=True)
    print(result[:5])


# Grid Search
def clas_gs(X_train, y_train, X_test, y_test, chosen):    
    X_pretrade = X_train.append(X_test)
    y_pretrade = y_train.append(y_test)
    preprocessor = data_preprocessor(X_pretrade)
    print('=============',chosen, 'Grid Search', '=============')
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', classifiers[chosen])])
    param_grid = parameters[chosen]
#    GS = GridSearchCV(model, param_grid, cv=tscv, scoring='balanced_accuracy', n_jobs=-1)
    GS = RandomizedSearchCV(model, param_grid, cv=tscv, random_state=0,
                            n_iter=200, scoring='balanced_accuracy', n_jobs=-1)
    GS.fit(X_pretrade, y_pretrade)  
    print('Best Parameters:',GS.best_params_)
    print('Best Score:', GS.best_score_)
    return GS


def reg_gs(X_train, y_train, X_test, y_test, chosen):    
    X_pretrade = X_train.append(X_test)
    y_pretrade = y_train.append(y_test)
    preprocessor = data_preprocessor(X_pretrade)
    print('=============',chosen, 'Grid Search', '=============')
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', regressors[chosen])])
    param_grid = reg_parameters[chosen]
#    GS = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    GS = RandomizedSearchCV(model, param_grid, cv=tscv, random_state=0,
                            n_iter=200, scoring='neg_mean_squared_error', n_jobs=-1)
    GS.fit(X_pretrade, y_pretrade)  
    print('Best Parameters:',GS.best_params_)
    return GS



# Model Evaluation
def clas_evaluation(X_train, y_train, X_test, y_test, X_trade, y_trade, model):
    X_pretrade = X_train.append(X_test)
    y_pretrade = y_train.append(y_test)
    X = X_pretrade.append(X_trade)
    y = y_pretrade.append(y_trade)
    
    # classification report
#    print('=============','Train Data', '=============')
#    y_train_pred = model.predict(X_train)
#    print(classification_report(y_train,y_train_pred))
#    print('Balanced Accuracy Score', balanced_accuracy_score(y_train,y_train_pred))
#    
#    print('=============','Test Data', '=============')
#    y_test_pred = model.predict(X_test)
#    print(classification_report(y_test,y_test_pred))
#    print('Balanced Accuracy Score', balanced_accuracy_score(y_test,y_test_pred))
    
    print('=============','Trade Data', '=============')
    y_trade_pred = model.predict(X_trade)
    print(classification_report(y_trade,y_trade_pred))
    print('Balanced Accuracy Score', balanced_accuracy_score(y_trade,y_trade_pred))
    
#    print('=============','All Data', '=============')
#    y_pred = model.predict(X)
#    print(classification_report(y,y_pred))
#    print('Balanced Accuracy Score', balanced_accuracy_score(y,y_pred))
    
    # feature importance
    perm = PermutationImportance(model, random_state=0).fit(X_trade, y_trade)
    
    return y_trade_pred, perm


def reg_evaluation(X_train, y_train, X_test, y_test, X_trade, y_trade, model):
    X_pretrade = X_train.append(X_test)
    y_pretrade = y_train.append(y_test)
    X = X_pretrade.append(X_trade)
    y = y_pretrade.append(y_trade)
    
    # classification report
#    print('=============','Train Data', '=============')
#    y_train_pred = model.predict(X_train)
#    print(classification_report(y_train,y_train_pred))
#    print('Balanced Accuracy Score', balanced_accuracy_score(y_train,y_train_pred))
#    
#    print('=============','Test Data', '=============')
#    y_test_pred = model.predict(X_test)
#    print(classification_report(y_test,y_test_pred))
#    print('Balanced Accuracy Score', balanced_accuracy_score(y_test,y_test_pred))
    
    print('=============','Trade Data', '=============')
    y_trade_pred = model.predict(X_trade)
    print('Mean Squared Error:', mean_squared_error(y_trade,y_trade_pred))
    
#    print('=============','All Data', '=============')
#    y_pred = model.predict(X)
#    print(classification_report(y,y_pred))
#    print('Balanced Accuracy Score', balanced_accuracy_score(y,y_pred))
    
    # feature importance
    perm = PermutationImportance(model, random_state=0).fit(X_trade, y_trade)
    
    return y_trade_pred, perm


# reinforcement learning environment
def create_env(df, selected_cols):
    stock_dimension = 1
    # [Current Balance]+[Close prices * dim]+[Owned shares * dim] +[Other * dim]
    state_space = 1 + 2*stock_dimension + len(selected_cols) * stock_dimension
    indicators = selected_cols
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    env_kwargs = {
        "hmax": 1, 
        "initial_amount": 10000, 
        "buy_cost_pct": 0,
        "sell_cost_pct": 0,
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": indicators, 
        "action_space": stock_dimension, 
        "reward_scaling": 1e-4
    }
    e_train_gym = StockTradingEnv(df, **env_kwargs)
    return e_train_gym








