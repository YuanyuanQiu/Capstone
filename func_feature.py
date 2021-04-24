import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
import scipy.stats
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif, chi2, f_regression
from collections import defaultdict
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


# Handling Multicollinear Features
def feature_correlation(X,y):
    df = pd.merge(X, y, left_index=True, right_index=True)
    print(df.corr()['Label'].sort_values().drop('Label'))
    
    plt.figure(figsize=(20, 12))
    sns.heatmap(round(df.corr(),2), annot=False, cmap="coolwarm");


def cal_vif(X):
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return vif


def fs_hierarchical_clustering(X, threshold=0.9):
    # Perform hierarchical clustering on the Spearman rank-order correlations
    corr = spearmanr(X).correlation
    corr_linkage = hierarchy.ward(corr) # perform Ward’s linkage on a condensed distance matrix

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    dendro = hierarchy.dendrogram(corr_linkage, labels=X.columns, ax=ax1, leaf_rotation=90)
    dendro_idx = np.arange(0, len(dendro['ivl']))

    ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
    ax2.set_yticklabels(dendro['ivl'])
    fig.tight_layout()
    plt.show()
    
    # Pick a threshold, and keep a single feature from each cluster
    cluster_ids = hierarchy.fcluster(corr_linkage, threshold, criterion='distance') # Form flat clusters from the hierarchical clustering
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_idx = [v[0] for v in cluster_id_to_feature_ids.values()]
    selected_cols = X.iloc[:,selected_idx].columns
    print('Number of Features:', len(selected_idx))
    print('Selected Features:', X.columns[selected_idx])
    return selected_cols


# Select Important Features
def fs_importance(X, y, n_features=20):

    '''Removing features with low variance'''
    # fs = VarianceThreshold(threshold=(.8 * (1 - .8)))
    
    '''Univariate feature selection: selecting the best features based on univariate statistical tests'''
    # Regression: f_regression, mutual_info_regression; Classification: chi2, f_classif, mutual_info_classif
    fs = SelectKBest(chi2, k=n_features)
    
    '''Recursive feature elimination: least important features are pruned'''
    # fs = RFE(estimator=SVC(kernel="linear", C=1), n_features_to_select=n_features, step=1)
    
    '''Feature selection using SelectFromModel'''
    # # L1-based: L1 norm have sparse solutions: many coefficients are zero
    # fs = SelectFromModel(LinearSVC(C=0.01, penalty="l1"))
    # Tree-based: irrelevant features
    # fs = SelectFromModel(ExtraTreesClassifier())
    # fs = SelectFromModel(PermutationImportance(RandomForestClassifier(), cv=5),threshold=0.001,)
        
    fs.fit(X, y)
    # Get columns to keep and create new dataframe with those only
    selected_idx = fs.get_support(indices=True)
    selected_cols = X.iloc[:,selected_idx].columns
    print('Number of Features:', len(selected_cols))
    print('Selected Features:', selected_cols)
    return selected_cols


def reg_fs_importance(X, y, n_features=20):

    '''Removing features with low variance'''
    # fs = VarianceThreshold(threshold=(.8 * (1 - .8)))
    
    '''Univariate feature selection: selecting the best features based on univariate statistical tests'''
    # Regression: f_regression, mutual_info_regression; Classification: chi2, f_classif, mutual_info_classif
    fs = SelectKBest(f_regression, k=n_features)
    
    '''Recursive feature elimination: least important features are pruned'''
    # fs = RFE(estimator=SVC(kernel="linear", C=1), n_features_to_select=n_features, step=1)
    
    '''Feature selection using SelectFromModel'''
    # # L1-based: L1 norm have sparse solutions: many coefficients are zero
#     fs = SelectFromModel(LinearSVC(C=0.01, penalty="l1"))
    # Tree-based: irrelevant features
    # fs = SelectFromModel(ExtraTreesClassifier())
    # fs = SelectFromModel(PermutationImportance(RandomForestClassifier(), cv=5),threshold=0.001,)
        
    fs.fit(X, y)
    # Get columns to keep and create new dataframe with those only
    selected_idx = fs.get_support(indices=True)
    selected_cols = X.iloc[:,selected_idx].columns
    print('Number of Features:', len(selected_cols))
    print('Selected Features:', selected_cols)
    return selected_cols


# Final Features
def fs_final(X_train, X_test, X_trade, selected_cols): 
    X_train_final = X_train.loc[:,selected_cols]
    X_test_final = X_test.loc[:,selected_cols]
    X_trade_final = X_trade.loc[:,selected_cols]
    return X_train_final, X_test_final, X_trade_final


