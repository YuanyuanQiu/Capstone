import numpy as np
import pandas as pd
from prettytable import PrettyTable
from stockstats import StockDataFrame as Sdf
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns


def economic_indicator(df,indicators=['GDP', 'PAYEMS', 'DTB1YR', 'DTB6',
                                      'DTB3', 'DGS3MO', 'DGS1MO', 'CPIAUCSL',
                                      'BOGMBASE', 'TEDRATE']):
    fred = Fred(api_key='cfcd953abacca55f74c93ca364dfce2d')
    df_eco = df.copy()
    for i in indicators:
        # Scrap
        indicator = fred.get_series_all_releases(i)
        indicator.dropna(inplace=True)
        indicator.drop_duplicates(subset=['date'], keep='first', inplace=True)
        indicator.drop_duplicates(subset=['realtime_start'], keep='last', inplace=True)
        indicator.set_index('realtime_start', inplace=True)
        indicator.drop(['date'],axis=1, inplace=True)

        # Merge
        df_eco = pd.merge(df_eco, indicator, how='left', left_index=True, right_index=True)
        df_eco.rename(columns={'value':i}, inplace=True)
        
    df_eco.fillna(method = 'ffill', inplace=True)
    df_eco.fillna(method = 'bfill', inplace=True)

    return df_eco


def technical_indicator(df):
    '''
    calcualte technical indicators
    :param data: (df_tec) pandas dataframe
    :return: (df_tec) pandas dataframe
    '''
    
    df_tec = df.copy()
    
    # Volatility Feature
    df_tec['volatility_-5'] = df_tec.Close.ewm(5).std()
    df_tec['volatility_-10'] = df_tec.Close.ewm(10).std()
    df_tec['volatility_-20'] = df_tec.Close.ewm(20).std()
    df_tec['volatility_-60'] = df_tec.Close.ewm(60).std()
    df_tec['volatility_-120'] = df_tec.Close.ewm(120).std()
    
    # use stockstats package to add additional technical inidactors
    stock = Sdf.retype(df_tec.copy())
    # close price change (in percent)
    df_tec['close_-5_r'] = stock['close_-5_r']
    df_tec['close_-10_r'] = stock['close_-10_r']
    df_tec['close_-20_r'] = stock['close_-20_r']
    df_tec['close_-60_r'] = stock['close_-60_r']
    df_tec['close_-120_r'] = stock['close_-120_r']
    # volume change (in percent)
    df_tec['volume_-5_r'] = stock['volume_-5_r']
    df_tec['volume_-10_r'] = stock['volume_-10_r']
    df_tec['volume_-20_r'] = stock['volume_-20_r']
    df_tec['volume_-60_r'] = stock['volume_-60_r']
    df_tec['volume_-120_r'] = stock['volume_-120_r']
    # volume delta against previous day
    df_tec['volume_delta'] = stock['volume_delta']
    # volume max of three days ago, two days ago and yesterday
    df_tec['volume_-3,-2,-1_max'] = stock['volume_-3,-2,-1_max']
    df_tec['volume_-10_max_r'] = stock['volume_-3,-2,-1_max']/stock['volume_-10_max']
    df_tec['volume_-20_max_r'] = stock['volume_-3,-2,-1_max']/stock['volume_-20_max']
    df_tec['volume_-60_max_r'] = stock['volume_-3,-2,-1_max']/stock['volume_-60_max']
    df_tec['volume_-120_max_r'] = stock['volume_-3,-2,-1_max']/stock['volume_-120_max']
    # volume min of three days ago, two days ago and yesterday
    df_tec['volume_-3,-2,-1_min'] = stock['volume_-3,-2,-1_min']
    df_tec['volume_-10_min_r'] = stock['volume_-3,-2,-1_min']/stock['volume_-10_min']
    df_tec['volume_-20_min_r'] = stock['volume_-3,-2,-1_min']/stock['volume_-20_min']
    df_tec['volume_-60_min_r'] = stock['volume_-3,-2,-1_min']/stock['volume_-60_min']
    df_tec['volume_-120_min_r'] = stock['volume_-3,-2,-1_min']/stock['volume_-120_min']
    # KDJ, default to 9 days
    df_tec['kdjk'] = stock['kdjk']
    df_tec['kdjd'] = stock['kdjd']
    df_tec['kdjj'] = stock['kdjj']
    # simple moving average on close price
    df_tec['close_5_sma'] = stock['close_5_sma']
    df_tec['close_10_sma'] = stock['close_10_sma']
    df_tec['close_20_sma'] = stock['close_20_sma']
    df_tec['close_60_sma'] = stock['close_60_sma']
    df_tec['close_120_sma'] = stock['close_120_sma']
    # exponential moving average on close price
    df_tec['close_5_ema'] = stock['close_5_ema']
    df_tec['close_10_ema'] = stock['close_10_ema']
    df_tec['close_20_ema'] = stock['close_20_ema']
    df_tec['close_60_ema'] = stock['close_60_ema']
    df_tec['close_120_ema'] = stock['close_120_ema']
    # DMA, difference of 10 and 50 moving average
    df_tec['dma'] = stock['dma']
    # MACD
    df_tec['macd'] = stock['macd']
    # MACD signal line
    df_tec['macds'] = stock['macds']
    # bolling, including upper band and lower band
    df_tec['boll'] = stock['boll']
    df_tec['boll_ub'] = stock['boll_ub']
    df_tec['boll_lb'] = stock['boll_lb']
    df_tec['boll_ub_r'] = df_tec['Close']/stock['boll_ub']
    df_tec['boll_lb_r'] = df_tec['Close']/stock['boll_lb']
    # 5 days RSI
    df_tec['rsi_5'] = stock['rsi_5']
    df_tec['rsi_10'] = stock['rsi_10']
    df_tec['rsi_20'] = stock['rsi_20']
    df_tec['rsi_60'] = stock['rsi_60']
    df_tec['rsi_120'] = stock['rsi_120']
    df_tec['rsi_10_r'] = stock['rsi_1']/stock['rsi_10']
    df_tec['rsi_20_r'] = stock['rsi_1']/stock['rsi_20']
    df_tec['rsi_60_r'] = stock['rsi_1']/stock['rsi_60']
    df_tec['rsi_120_r'] = stock['rsi_1']/stock['rsi_120']
    # CCI, default to 14 days
    df_tec['cci_5'] = stock['cci_5']
    df_tec['cci_10'] = stock['cci_10']
    df_tec['cci_20'] = stock['cci_20']
    df_tec['cci_60'] = stock['cci_60']
    df_tec['cci_120'] = stock['cci_120']
    # DX, 30 days of +DI and -DI
    df_tec['dx_5'] = stock['dx_5']
    df_tec['dx_10'] = stock['dx_10']
    df_tec['dx_20'] = stock['dx_20']
    df_tec['dx_60'] = stock['dx_60']
    df_tec['dx_120'] = stock['dx_120']
    # VR, default to 26 days
    df_tec['vr_20'] = stock['vr_20']
    df_tec['vr_60'] = stock['vr_60']
    df_tec['vr_120'] = stock['vr_120']
    
    return df_tec


def get_trading_log(df_labeled, principle=10000, percent=0.7):
    trading_log = df_labeled.copy()
    trading_log['Buy to open'] = 0
    trading_log['Buy to close'] = 0
    trading_log['Sell to open'] = 0
    trading_log['Sell to close'] = 0
    
    trading_log['Position'] = 0
    trading_log['Cost'] = 0
    trading_log['Invest Amt'] = 0
    trading_log['Duration'] = 0
    
    trading_log['Profit'] = 0
    trading_log['Profit %'] = 0
    trading_log['Cum Profit'] = 0
    trading_log['Principle'] = 0
    trading_log['Daily Return'] = 0
    
    
    for i in range(len(trading_log)):
        
        '''1.get latest position info '''
        '''1) Position, Cost, Invest Amt, Duration recall'''
        if i > 0 and trading_log.iloc[i-1,trading_log.columns.get_loc("Buy to close")] != 1 \
                 and trading_log.iloc[i-1,trading_log.columns.get_loc("Sell to close")] != 1:
            trading_log.iloc[i,trading_log.columns.get_loc("Position"):trading_log.columns.get_loc("Profit")] = \
            trading_log.iloc[i-1,trading_log.columns.get_loc("Position"):trading_log.columns.get_loc("Profit")] 
        '''2) Duration update'''
        if trading_log.iloc[i,trading_log.columns.get_loc("Duration")] > 0: 
            trading_log.iloc[i,trading_log.columns.get_loc("Duration")] += 1
        '''3) Principle update'''
        trading_log.iloc[i,trading_log.columns.get_loc("Principle")] = principle
        
        
        '''2. get profit/loss %'''
        positions = abs(trading_log.iloc[i,trading_log.columns.get_loc("Position")])
        if trading_log.iloc[i,trading_log.columns.get_loc("Position")] > 0: # Long
            unit_profit = trading_log.iloc[i,trading_log.columns.get_loc("Close")] - trading_log.iloc[i,trading_log.columns.get_loc("Cost")]
            temp_profit = unit_profit * positions
            trading_log.iloc[i,trading_log.columns.get_loc("Principle")] = principle + temp_profit # Principle
        elif trading_log.iloc[i,trading_log.columns.get_loc("Position")] < 0: # Short
            unit_profit = trading_log.iloc[i,trading_log.columns.get_loc("Cost")] - trading_log.iloc[i,trading_log.columns.get_loc("Close")]
            temp_profit = unit_profit * positions
            trading_log.iloc[i,trading_log.columns.get_loc("Principle")] = principle + temp_profit # Principle

        
        '''3. close/open position'''
        
        # Close
        if trading_log.iloc[i,trading_log.columns.get_loc("Position")] != 0:
            # buy to close
            if trading_log.iloc[i,trading_log.columns.get_loc("Position")] < 0 \
            and trading_log.iloc[i,trading_log.columns.get_loc("Label")] == 1: 
                trading_log.iloc[i,trading_log.columns.get_loc("Buy to close")] = 1 # Buy to close

                trading_log.iloc[i,trading_log.columns.get_loc("Profit")] = temp_profit # Profit
                trading_log.iloc[i,trading_log.columns.get_loc("Profit %")] = temp_profit / trading_log.iloc[i,trading_log.columns.get_loc("Invest Amt")] # Profit %
                principle += temp_profit
                
            # sell to close
            elif trading_log.iloc[i,trading_log.columns.get_loc("Position")] > 0 \
            and trading_log.iloc[i,trading_log.columns.get_loc("Label")] == -1:
                trading_log.iloc[i,trading_log.columns.get_loc("Sell to close")] = 1 # Sell to close
            
                trading_log.iloc[i,trading_log.columns.get_loc("Profit")] = temp_profit # Profit
                trading_log.iloc[i,trading_log.columns.get_loc("Profit %")] = temp_profit / trading_log.iloc[i,trading_log.columns.get_loc("Invest Amt")] # Profit %
                principle += temp_profit
            
        # Open
        elif trading_log.iloc[i,trading_log.columns.get_loc("Position")] == 0:
            bets = trading_log.iloc[i,trading_log.columns.get_loc("Principle")] * percent // trading_log.iloc[i,trading_log.columns.get_loc("Close")]
            
            # buy to open
            if trading_log.iloc[i,trading_log.columns.get_loc("Label")] == 1:       
                trading_log.iloc[i,trading_log.columns.get_loc("Buy to open")] = 1 # Buy to open
                trading_log.iloc[i,trading_log.columns.get_loc("Position")] += bets # Position
                trading_log.iloc[i,trading_log.columns.get_loc("Duration")] = 1 # Duration

            # sell to open
            elif trading_log.iloc[i,trading_log.columns.get_loc("Label")] == -1:
                trading_log.iloc[i,trading_log.columns.get_loc("Sell to open")] = 1 # Sell to open
                trading_log.iloc[i,trading_log.columns.get_loc("Position")] -= bets # Position
                trading_log.iloc[i,trading_log.columns.get_loc("Duration")] = 1 # Duration
            
            trading_log.iloc[i,trading_log.columns.get_loc("Cost")] = trading_log.iloc[i,trading_log.columns.get_loc("Close")] # Cost
            trading_log.iloc[i,trading_log.columns.get_loc("Invest Amt")] = trading_log.iloc[i,trading_log.columns.get_loc("Cost")] * abs(trading_log.iloc[i,trading_log.columns.get_loc("Position")]) # Invest Amt
            
    trading_log['Profit %'] = trading_log['Profit %'].apply(lambda x: round(x, 2))
    trading_log['Cum Profit'] = trading_log['Profit'].cumsum()
    trading_log['Daily Return'] = trading_log['Principle'].pct_change(1)
    trading_log.iloc[0,trading_log.columns.get_loc("Daily Return")] = 0
    
    return trading_log


def get_trading_report(trading_log, principle=10000, percent=0.7, show_fig=False):
    # Strategy
    s_profit = round(trading_log.iloc[-1,trading_log.columns.get_loc("Principle")]-principle,2)
    s_profit_pct = "%.2f%%" % (s_profit/principle * 100)
    duration = trading_log.shape[0]
    s_duration = duration - (trading_log['Duration'] == 0).sum()
    s_year = s_duration/252
    # annulized = (final/initial)^1/n - 1
    s_annulized = "%.2f%%" % ((pow((s_profit+principle)/principle,1/s_year) - 1) * 100)
    s_Sharpe = round(trading_log['Daily Return'].mean()/trading_log['Daily Return'].std() * np.sqrt(252),2)
    
    # Buy and Hold
    b_position = (principle * percent)//trading_log.iloc[0,trading_log.columns.get_loc("Close")]
    b_profit = round((trading_log.iloc[-1,trading_log.columns.get_loc("Close")]\
                                       -trading_log.iloc[0,trading_log.columns.get_loc("Close")]) * b_position,2)
    b_profit_pct = "%.2f%%" % (b_profit/principle * 100)
    b_duration = duration
    b_year = b_duration/252
    b_annulized = "%.2f%%" % ((pow((b_profit+principle)/principle,1/b_year) -1) * 100)
    b_daily_ret = trading_log['Close'].pct_change(1)
    b_Sharpe = round(b_daily_ret.mean()/b_daily_ret.std() * np.sqrt(252),2)
    
    # Table
    table = PrettyTable(['Model','Profit','Profit %','Annualized %','Duration','Sharpe'])
    table.add_row(['Strategy', s_profit, s_profit_pct, s_annulized, s_duration, s_Sharpe])
    table.add_row(['Buy & Hold', b_profit, b_profit_pct, b_annulized, b_duration, b_Sharpe])
    
    if show_fig:
        plt.tight_layout()
        
        sns.countplot(trading_log['Label'])
        plt.show()
        
        plt.figure(figsize = (12,10))
        plt.subplot(2, 2, 1)
        plt.plot(trading_log['Profit'])
        plt.title('Profit')
        plt.subplot(2, 2, 2)
        plt.plot(trading_log['Cum Profit'])
        plt.title('Cum Profit')
        plt.subplot(2, 2, 3)
        plt.plot(trading_log['Close'])
        plt.title('Close')
        plt.subplot(2, 2, 4)
        plt.plot(trading_log['Principle'])
        plt.title('Principle')
        plt.show()
        
#        profit_ps = trading_log[trading_log['Profit %']!=0]['Profit %']
#        profit_ps.plot(kind='hist', title='Single Trade Profit Histogram')
#        print('Max Single Trade Profit %', max(profit_ps))
#        print('Min Single Trade Profit %', min(profit_ps))
#        print('Mean Single Trade Profit %', round(profit_ps.mean(),2))
        
    return table, s_Sharpe



























