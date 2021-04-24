import numpy as np
import pandas as pd
import yfinance as yf

# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# Data Preprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, RobustScaler, Normalizer, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA


# Data
def get_stock_data(stock, start="2010-01-01", end="2020-12-31"):
    '''
    get stock data from yfinance
    '''
    df = yf.download(stock,start=start,end=end,auto_adjust=True)
    return df


# EDA
def data_overview(df):
    print(df.head(), '\n')
    print(df.info(), '\n')
    print(df.describe())


# Outliers (numeric only)
def outliers(col):
    std3 = col.std() * 3
    mean = col.mean()
    c = 0
    for row in col:
        if abs(row-mean) > std3:
            c += 1
    return c


def outlier_analysis(num_df):
    n = num_df.shape[0]
    ls_analysis = []
    for col in num_df:
        outlier = outliers(num_df[col])

        ls_analysis.append([col,outlier/n*100])

    df_outlier_analysis = pd.DataFrame(ls_analysis, columns=['Columns','Outliers%'])
    return df_outlier_analysis


# Missing, Imbalance, Unique Data
def analysis(df):
    n = df.shape[0]
    ls_analysis = []
    ls_unique = []
    for col in df:
        # missing data
        missing = sum(df[col].isnull())

        # imbalance data
        counts = df[col].value_counts().values
        counts.sort()
        imblance = counts[-1]

        # unique
        unique = df[col].nunique()
        if unique/n < 0.03:
            ls_unique.append(col)

        ls_analysis.append([col,missing/n*100, imblance/n*100, unique/n*100])

    df_analysis = pd.DataFrame(ls_analysis, columns=['Columns','Missing%','Imbalance%','Unique%'])
    return df_analysis, ls_unique


# Split
def clas_train_test_trade_split(df, train_start, train_end, test_start, test_end, trade_start, trade_end):
    # 10 years of data
    X = df.loc[train_start:,:].drop(['Label'], axis=1)
    y = df.loc[train_start:,:]['Label']
    # 7 years of training
    X_train = X[(X.index >= train_start) & (X.index <= train_end)]
    y_train = y[(y.index >= train_start) & (y.index <= train_end)]
    # 1 year of testing
    X_test = X[(X.index >= test_start) & (X.index <= test_end)]
    y_test = y[(y.index >= test_start) & (y.index <= test_end)]
    # 2 year of trading
    X_trade = X[(X.index >= trade_start) & (X.index <= trade_end)]
    y_trade = y[(y.index >= trade_start) & (y.index <= trade_end)]
    return X_train, y_train, X_test, y_test, X_trade, y_trade


def reg_train_test_trade_split(df, df_raw, train_start, train_end, test_start, test_end, trade_start, trade_end, window):
    trade_start_date = df[(df.index >= trade_start)].index[0]
    trade_start_actual = df.index[df.index.get_loc(trade_start_date) - window*2]
    test_end_actual = df.index[df.index.get_loc(trade_start_date) - window*2 - 1]
    
    # whole data
    X = df.loc[train_start:,:].drop(['Close'], axis=1)
    y = df_raw.loc[train_start:,:]['Close']
    # training
    X_train = X[(X.index >= train_start) & (X.index <= train_end)]
    y_train = y[(y.index >= train_start) & (y.index <= train_end)]
    # testing
    X_test = X[(X.index >= test_start) & (X.index <= test_end_actual)]
    y_test = y[(y.index >= test_start) & (y.index <= test_end_actual)]
    # trading
    X_trade = X[(X.index >= trade_start_actual) & (X.index <= trade_end)]
    y_trade = y[(y.index >= trade_start_actual) & (y.index <= trade_end)]
    return X_train, y_train, X_test, y_test, X_trade, y_trade, trade_start_actual


# Data Preprocessor
def data_preprocessor(X):
    num_features = X.select_dtypes(include=['int64', 'float64']).columns
    cat_features = X.select_dtypes(include=['object']).columns
    
    num_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(random_state=0)),
        ('scaler', MinMaxScaler()),
#         ('scaler', RobustScaler(quantile_range=[5, 95])),
#         ('norm', Normalizer(norm='l2'))
#         ,('pca', PCA(whiten=True, random_state=0))
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
    ])
    
    return preprocessor


# Data Transform for Neural Network
def data_transform(X_train, y_train, X_test, y_test, X_trade, y_trade, learn_type):
    # Transform X
    preprocessor = data_preprocessor(X_train)
    preprocessor.fit(X_train)
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    X_trade_transformed = preprocessor.transform(X_trade)
    
    if learn_type == 'clas':
        # Transform y
        encoder = LabelEncoder()
        encoder.fit(y_train)
        y_train_transformed = encoder.transform(y_train)
        y_test_transformed = encoder.transform(y_test)
        y_trade_transformed = encoder.transform(y_trade)
        
        return X_train_transformed, y_train_transformed, X_test_transformed, y_test_transformed, X_trade_transformed, y_trade_transformed, encoder
    else:
        return X_train_transformed, y_train, X_test_transformed, y_test, X_trade_transformed, y_trade

# split a multivariate sequence into samples
def split_sequences(features, target, n_steps):
    X, y = [], []
    for i in range(len(features)):
        end_ix = i + n_steps
        if end_ix > len(features):
            break
        seq_x, seq_y = features[i:end_ix], target[end_ix-1]
        X.append(seq_x)
        y.append(seq_y)
    X = np.array(X)
    y = np.array(y)
    print(X.shape, y.shape)
    return X, y


# reinforcement learning formatting
def rl_formatting(df, df_raw, tic):
    df_rl = df.copy()
    for i in df_raw.index:
        if i in df_rl.columns:
            df_rl.drop([i], axis=1, inplace=True)
    df_rl = pd.merge(df_rl, df_raw, how='left', left_index=True, right_index=True)
    df_rl["tic"] = tic
    df_rl = df_rl.reset_index()
    df_rl.rename(columns={"Date": "date", 'Open': "open", 'High': "high", 'Low': "low", 'Close': "close", 'Volume': "volume"}, inplace=True)
    df_rl["day"] = df_rl["date"].dt.dayofweek
    df_rl["date"] = df_rl["date"].apply(lambda x: x.strftime("%Y-%m-%d"))
    df_rl = df_rl.dropna()
    df_rl = df_rl.reset_index(drop=True)
    return df_rl






















