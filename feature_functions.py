#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script houses the functions from which custom features can be built for a machine learning model. 
These 3 examples are classic technical indicators for stock trading.
For the sake of exploring feature importance, each feature function is designed to add a single column to the input dictionary.
Each feature must have both a function and a dictionary with '_deap_params' added at the end of the dictionary variable name.
Each dictionary defines the optimization grid space; housing the range of values to be tested for each of the function's input parameters.
Depending on the temporal_granularity defined for the model, a corresponding 'lag' and 'growth_threshold' value is pulled from lag_growth_thresh_tiers.
The 'lag' and 'growth_threshold' value pulled from lag_growth_thresh_tiers is added to the grid space within the '_deap_params' dictionaries.
"""

lag_growth_thresh_tiers = {'one_minute': {'lag': [32,39], 'growth_threshold': [0.002,0.003]},
                           'three_minute': {'lag': [5,30], 'growth_threshold': [0.002,0.003]},
                           'five_minute': {'lag': [3,18], 'growth_threshold': [0.002,0.003]},
                           'fifteen_minute': {'lag': [1,4], 'growth_threshold': [0.002,0.004]},
                           'one_hour': {'lag': [1,2], 'growth_threshold': [0.002,0.005]},
                           'one_day': {'lag': [1,2], 'growth_threshold': [0.004,0.015]}}

def macd(source_df, macd_short, macd_long, macd_smooth):
    df = source_df.copy()
    #MACD Feature Creation
    short= df['close'].ewm(span=macd_short, adjust=False).mean()
    long = df['close'].ewm(span=macd_long, adjust=False).mean()
    df['macd'] = short - long
    return df

# test1 = MACD(source_df, macd_short, macd_long, macd_smooth)

macd_deap_params = {'macd_short': [8,15],
              'macd_long': [18, 28],
              'macd_smooth': [3,10]
    }
#####################################################################################

def macd_hist(source_df, macd_short, macd_long, macd_smooth):
    df = source_df.copy()
    #MACD Feature Creation
    short= df['close'].ewm(span=macd_short, adjust=False).mean()
    long = df['close'].ewm(span=macd_long, adjust=False).mean()
    macd = short - long
    #Signal Line Feature Creation
    signal_line = macd.ewm(span=macd_smooth, adjust=False).mean()
    df['macd_hist'] = macd -  signal_line
    return df

# test1 = MACD(source_df, macd_short, macd_long, macd_smooth)

macd_hist_deap_params = {'macd_short': [8,15],
              'macd_long': [18, 28],
              'macd_smooth': [3,10]
    }

#####################################################################################

def rsi(df, lookback=14):
    # Get the gains and losses
    gain = df['growth_rate'].where(df['growth_rate'] > 0, 0)
    loss = -df['growth_rate'].where(df['growth_rate'] < 0, 0)
    # Calculate the average gain and loss
    avg_gain = gain.rolling(window= lookback).mean()
    avg_loss = loss.rolling(window= lookback).mean()
    # Calculate the Relative Strength Index (RSI)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs)) 
    df_output = df.copy()
    df_output['rsi'] = rsi/100 #scaled to [0,1]
    df_output['rsi'].fillna(0)
    return df_output


rsi_deap_params = { 'lookback': [6,20] }
