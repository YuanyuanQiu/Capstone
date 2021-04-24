from func_trading import get_trading_log, get_trading_report

import numpy as np
import pandas as pd
import scipy


# Baseline Labeling
def baseline_labeling(df):
    df_baseline = df.copy()
    baseline_predictions = np.random.randint(-1, 2, len(df))
    df_baseline['Label'] = baseline_predictions
    return df_baseline


# Fixed Horizon
def fixed_horizon(df, horizon=10, T=0.025):
    df_fixed = df.copy()
    labels = []
    for i in range(len(df)-horizon):
        # WINDOW = df.iloc[i-window:i]
        now = df.Close[i]
        future = df.Close[i + horizon]
        ret = (future - now)/now
        
        if ret > T:
            labels.append(1)
        elif ret < -T:
            labels.append(-1)
        else:
            labels.append(0)
    labels.append(0)
    df_fixed['Label'] = labels
    return df_fixed


# Volatility Horizon
def volatility_horizon(df, horizon=10, lookback=10):
    labels = []
    for i in range(lookback, len(df)-horizon):
        window = df.iloc[i - lookback:i]
        now = df.Close[i]
        future = df.Close[i + horizon]
        ret = (future - now)/now
        
        window_abs_returns = np.abs(window.Close.pct_change())
        Ti = 2*np.std(window_abs_returns) + np.mean(window_abs_returns)
        
        if ret > Ti:
            labels.append(1)
        elif ret < -Ti:
            labels.append(-1)
        else:
            labels.append(0)
    
    return labels


# Triple-barrier
# get volatility (threshold)
def get_vol(prices, span=100, delta=pd.Timedelta(hours=1)):
    # 1. compute returns of the form p[t]/p[t-1] - 1
    
    # find the timestamps of p[t-1]
    df0 = prices.index.searchsorted(prices.index - delta)
    df0 = df0[df0>0]

    # align timestamps of p[t-1] to timestamps of p[t]
    df0 = pd.Series(prices.index[df0-1],index=prices.index[prices.shape[0]-df0.shape[0] : ])

    # get values by timestamps, then compute returns
    df0 = prices.loc[df0.index] / prices.loc[df0.values].values - 1

    # 2. estimate rolling standard deviation
    df0 = df0.ewm(span=span).std()

    return df0

# get timestamps of the horizon barriers (t1)
def get_horizons(prices, delta=pd.Timedelta(days=10)):
    t1 = prices.index.searchsorted(prices.index + delta)
    t1 = t1[t1 < prices.shape[0]]
    t1 = prices.index[t1]
    t1 = pd.Series(t1, index=prices.index[:t1.shape[0]])
    return t1

# set upper and lower barriers based on volatility
def get_touches(prices, events, factors=[1, 1]):
    '''
    events: pd dataframe with columns
    t1: timestamp of the next horizon
    threshold: unit height of top and bottom barriers
    side: the side of each bet
    factors: multipliers of the threshold to set the height of top/bottom barriers
    '''
    out = events[['t1']].copy(deep=True)
    if factors[0] > 0:
        thresh_uppr = factors[0] * events['threshold']
    else:
        thresh_uppr = pd.Series(index=events.index) # no uppr thresh
        
    if factors[1] > 0:
        thresh_lwr = -factors[1] * events['threshold']
    else:
        thresh_lwr = pd.Series(index=events.index)  # no lwr thresh
    
    for loc, t1 in events['t1'].iteritems(): # iterates over the DataFrame columns, returning a tuple with the column name and the content as a Series.
        df0 = prices[loc:t1] # path prices
        df0 = (df0 / prices[loc] - 1) * events.side[loc]  # path returns
        out.loc[loc, 'stop_loss'] = df0[df0 < thresh_lwr[loc]].index.min()  # earliest stop loss
        out.loc[loc, 'take_profit'] = df0[df0 > thresh_uppr[loc]].index.min() # earliest take profit
    
    return out

def get_labels(touches):
    out = touches.copy(deep=True)
    # pandas df.min() ignores NaN values
    first_touch = touches[['stop_loss', 'take_profit']].min(axis=1)
    for loc, t in first_touch.iteritems():
        if pd.isnull(t):
            out.loc[loc, 'label'] = 0
        elif t == touches.loc[loc, 'stop_loss']: 
            out.loc[loc, 'label'] = -1
        else:
            out.loc[loc, 'label'] = 1
    return out


# Meta labeling
def get_meta_barrier(future_window, now, min_ret, tp, sl, vertical_zero = False):
    
    min_ret_situation = [0, 0, 0] # [up,down,vertical]
    
    # return for every bar in future_window
    differences = np.array([(fc - now) / now for fc in future_window])
    
    '''1st label: direction (min_ret_situation)'''
    # get first bar hit the up fluctuation
    min_ret_ups = np.where((differences >= min_ret) == True)[0]
    # get first bar hit the down fluctuation
    min_ret_downs = np.where((differences <= -min_ret) == True)[0]
    
    # if not hit up/down fluctuations
    if (len(min_ret_ups) == 0) and (len(min_ret_downs) == 0):
        if vertical_zero: # with vertical barrier
            min_ret_situation[2] = 1 # [0,0,1]        
        else: # without vertical barrier
            if differences[-1] > 0:
                min_ret_situation[0] = 1 # [1,0,0]
            else:
                min_ret_situation[1] = 1 # [0,1,0]
    
    # if hit up/down fluctuations
    else:
        # if not hit up but hit down
        if len(min_ret_ups) == 0:
            min_ret_ups = [np.inf]
        # if not hit down but hit up
        if len(min_ret_downs) == 0:
            min_ret_downs = [np.inf]
        
        if min_ret_ups[0] < min_ret_downs[0]:
            min_ret_situation[0] = 1 # [1,0,0]
        else:
            min_ret_situation[1] = 1 # [0,1,0]
    
    '''2nd label: certainty of this bet (take_action)'''
    # get first bar hit the up barrier
    take_profit = np.where((differences >= tp) == True)[0]
    # get first bar hit the down barrier
    stop_loss = np.where((differences < sl) == True)[0]
    
    '''Fluctuation directions coincide with take profit/stop loss actions'''
    # if first label says “up” & hit the take profit goal, label as 1
    if min_ret_situation[0] == 1 and len(take_profit) != 0:
        take_action = 1
    # if we have the first label “down” & hit the stop loss goal, label as 1
    elif min_ret_situation[1] == 1 and len(stop_loss) != 0:
        take_action = 1
    else:
        take_action = 0
    
    return min_ret_situation, take_action

def final_meta_labeling(df_labeled):
    if df_labeled['up'] == 1 and df_labeled['action'] == 1:
        return 1
    elif df_labeled['down'] == 1 and df_labeled['action'] == 1:
        return -1
    else:
        return 0

def meta_labeling(df, WINDOW=40, HORIZON=40, TP=0.15, SL=-0.15):   
    Date, X, Y, Y2, TIs = [], [], [], [], []
    
    for i in range(WINDOW, len(df), 1):
        window = df.iloc[i-WINDOW:i]
        now = df.Close[i]
        future_window = df.Close[i:i+HORIZON]
        
        # history
        window_abs_returns = np.abs(window.Close.pct_change())
        Ti = np.std(window_abs_returns) + np.mean(window_abs_returns)
        min_ret_situation, take_action = get_meta_barrier(future_window, now, Ti, TP, SL, True)
        
        Date.append(df.iloc[[i]].index[0])
        X.append(now)
        Y.append(min_ret_situation)
        Y2.append(take_action)
        TIs.append(Ti)
        
    df_labeled = pd.DataFrame(Y,columns = ['up','down','vertical'])
    df_labeled['action'] = Y2
    df_labeled['Close'] = X
    df_labeled['Date'] = Date
    df_labeled.set_index(['Date'],inplace=True)
    df_labeled['Label'] = df_labeled.apply(lambda x: final_meta_labeling(x), axis=1)
    df_labeled = df_labeled.iloc[:,4:]
    
    df_labeled = pd.merge(df_labeled, df, how='left', left_index=True, right_index=True)
    df_labeled.drop(['Close_y'], axis=1, inplace=True)
    df_labeled.rename(columns={'Close_x':'Close'}, inplace=True)
    
    return df_labeled, TIs

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def get_best_window(stock, principle, percent):
    s_Sharpe = 0
    for TP in [0.05,0.1,0.15,0.2]:
        for SL in [0.05,0.1,0.15,0.2]:
            for WINDOW in [10, 20, 40, 60]:
                for HORIZON in [10, 20, 40, 60]:
                    df_labeled, df_TIs = meta_labeling(stock, WINDOW=WINDOW, HORIZON=HORIZON, TP=TP, SL=SL * -1)
                    df_trading_log = get_trading_log(df_labeled, principle=principle, percent=percent)
                    df_report, s_Sharpe_temp = get_trading_report(df_trading_log)
                    if s_Sharpe < s_Sharpe_temp:
                        TP_best = TP
                        SL_best = SL
                        WINDOW_best = WINDOW
                        HORIZON_best = HORIZON
                        s_Sharpe = s_Sharpe_temp
    print(TP_best,SL_best, WINDOW_best, HORIZON_best, s_Sharpe)
    return TP_best,SL_best, WINDOW_best, HORIZON_best




