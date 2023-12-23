#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The trigger concept: long positions should not be opened with a nosediving stock
This script finds a short- and long-duration weighted average of daily prices and calculates the percent
difference between them, finding an optimal percent threshold that eliminates certain days and 
increases annualized returns, if the optimal percent threshold exists.
"""


import numpy as np
from polygon import RESTClient
from datetime import datetime as dt
import pandas as pd

def identify_trigger_qualified_days(df_copy, l, s, pct_threshold):
    '''
    This function creates a list of days where the stock is not 'nose-diving', as defined by a
    short- and long-duration weighted average percent change, and a given percent threshold
    
    Parameters
    ----------
    df_copy : TYPE pandas.DataFrame
        DESCRIPTION. The daily pricing data for a given stock
    l : TYPE integer
        DESCRIPTION. The number of periods to include in the long-duration weighted average. There are 2/day.
    s : TYPE integer
        DESCRIPTION. The number of periods to include in the short-duration weighted average. There are 2/day.
    pct_threshold : TYPE float
        DESCRIPTION. The percent change threshold for which all days below are eliminated

    Returns
    -------
    true_days : TYPE
        DESCRIPTION.

    '''
    # In case of input error, return no days
    if l is None or s is None or pct_threshold is None:
        true_days = None
    else:
        # Automatically include initial days that compose the warm-up period for the long weighted average
        true_days = df_copy.day.unique()[:int(l/2)].tolist()
        ## Create a single series of alternating open/close prices in chronological order
        df_copy['close_shift'] = df_copy['close'].shift(1) 
        chronologic_prices = []
        for index, row in df_copy.iterrows():
            chronologic_prices.append(row['close_shift'])
            chronologic_prices.append(row['open'])
        chronologic_prices = pd.Series(chronologic_prices).dropna().reset_index(drop=True)
        ## Calculate the weighted rolling averages
        decay_factor = 0.97
        weights_l = np.array([decay_factor ** i for i in range(int(l))])
        weights_s = np.array([decay_factor ** i for i in range(int(s))])
        long_weighted_avg = chronologic_prices.rolling(window=int(l)).apply(lambda x: np.sum(x * weights_l) / np.sum(weights_l))[::2].tail(len(df_copy)).reset_index(drop=True)
        short_weighted_avg = chronologic_prices.rolling(window=int(s)).apply(lambda x: np.sum(x * weights_s) / np.sum(weights_s))[::2].tail(len(df_copy)).reset_index(drop=True)
        ## Identify all days with a the percent change above the pct_threshold
        percent_change = short_weighted_avg / long_weighted_avg
        day_bool = percent_change > pct_threshold
        true_days.extend(df_copy.loc[day_bool, 'day'].tolist())
    return true_days


def trigger_opt(key, tkr, rounding_threshold_results, lag):
    '''
    This function finds the optimal percent threshold to use for eliminating trading days on nosediving stocks.

    Parameters
    ----------
    key : TYPE string
        DESCRIPTION. The API key from polygon.io associated with the user's account.
    tkr : TYPE string
        DESCRIPTION. The ticker symbol of the stock of interest for the model
    rounding_threshold_results : TYPE dictionary
        DESCRIPTION. The output from the opt_rounding_thresh_htnb_logic() function in hold_strat.py
    lag : TYPE integer
       DESCRIPTION. The number of minutes over which the model is trained to predict growth. 

    Returns
    -------
    trigger_opt_dict : TYPE dictionary
        DESCRIPTION. The results of the trigger assessment.

    '''
    ## set up key variables from input and for optimization
    possible_buys = rounding_threshold_results['opt_thresh_buy_log']
    num_days_in_opt = rounding_threshold_results['assessment_df'].day.nunique()
    trade_strat_opt_days = rounding_threshold_results['assessment_df'].day.unique().tolist()
    last_trading_day = rounding_threshold_results['assessment_df']['timemerge_dt'].max().date()
    first_trading_day = rounding_threshold_results['assessment_df']['timemerge_dt'].min().date()

    l_high = 28
    l_low = 10
    s_high = 6
    s_low = 1
    l_grid = np.linspace(l_low, l_high, l_high - l_low + 1)
    s_grid = np.linspace(s_low, s_high, s_high - s_low + 1)
    pct_grid = np.linspace(-0.08,0,35)
    
    ## Establish baseline
    trigger_free_returns = possible_buys.returns
    trigger_free_total_return = trigger_free_returns.prod()
    trigger_free_geom_mean = trigger_free_returns.prod() ** (1/len(trigger_free_returns))
    grid_df = pd.DataFrame({'geom_mean': [trigger_free_geom_mean],
                            'return': [trigger_free_total_return],
                            'conf_int': [np.percentile(trigger_free_returns, [2.5, 97.5])],
                            'annualized_return': [trigger_free_geom_mean ** ((len(trigger_free_returns) / num_days_in_opt) * 250)],
                            'days_trading': [num_days_in_opt], 
                            'num_buys': [len(trigger_free_returns)], 
                            'buys/day': [len(trigger_free_returns) / num_days_in_opt], 
                            'trigger_l': [None], 
                            'trigger_s': [None], 
                            'trigger_pct': [None]})
    
    ## Pull data
    client = RESTClient(key)
    data = client.get_aggs(tkr, multiplier=1, timespan = 'day', from_ = first_trading_day, to = last_trading_day)
    df = pd.DataFrame(data).drop(columns='otc')
    df['date_EST'] = pd.to_datetime(df['timestamp'], unit = "ms").dt.tz_localize('UTC').dt.tz_convert('US/Eastern')#adding eastern timezone
    df['timemerge'] = df['date_EST'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['day'] = df['date_EST'].dt.strftime('%Y-%m-%d') #creating day column
    
    ## Grid search for optimal trigger parameters
    start = dt.now()
    for l in l_grid:
        for s in s_grid:
            if s >= l:  #short run cannot be longer than the long run, skips if so
               continue
            for pct in pct_grid:
                df_copy = df.copy() #getting copy of daily data to avoid overriding 
                true_days = identify_trigger_qualified_days(df_copy, l, s, pct) # Calculate days that satisfy 'trigger' requirements
                if len(true_days) > (0.4 * len(trade_strat_opt_days)): # Can't eliminate more than 60% of trading days
                    true_days_possible_buys = possible_buys[possible_buys.day.isin(true_days)] # Find all buy orders that fall on qualified days
                    true_days_returns = true_days_possible_buys.returns # Isolate the returns from the true_days
                    true_days_total_return = true_days_returns.prod() # Calculate total return
                    true_days_geom_mean = true_days_returns.prod() ** (1/len(true_days_returns)) 
                    days_trading = sum(day in trade_strat_opt_days for day in true_days) # Count of days in trade_strat_opt_days that also exist in true_days
                    buys_pday = len(true_days_returns) / num_days_in_opt
                    conf_int = np.percentile(true_days_returns, [2.5, 97.5])
                    grid_df = pd.concat([grid_df, pd.DataFrame({'geom_mean': [true_days_geom_mean], 
                                                                'return': [true_days_total_return],
                                                                'conf_int': [conf_int],
                                                                'annualized_return': [true_days_geom_mean ** (buys_pday * 250)],
                                                                'days_trading': [days_trading], 
                                                                'num_buys': [len(true_days_returns)],
                                                                'buys/day': [buys_pday],
                                                                'trigger_l': [l], 
                                                                'trigger_s': [s], 
                                                                'trigger_pct': [pct] }
                                                               )], ignore_index = True)
                   
    duration = dt.now() - start
    print(f'Trigger grid search lasted: {duration}.')
    
    ## Assimilating results
    max_return_row = grid_df[grid_df.annualized_return == grid_df['annualized_return'].max()]
    trigger_results = pd.DataFrame([grid_df.iloc[0], max_return_row.iloc[0]])
    trigger_results.reset_index(drop=True, inplace=True)
    opt_trigger_params = max_return_row.iloc[0][['trigger_l', 'trigger_s', 'trigger_pct']].to_dict()
    trigger_qualified_days = identify_trigger_qualified_days(df, opt_trigger_params['trigger_l'], opt_trigger_params['trigger_s'], opt_trigger_params['trigger_pct'])
    if trade_strat_opt_days is not None and trigger_qualified_days is not None:
        trigger_eliminated_days = [day for day in trade_strat_opt_days if day not in trigger_qualified_days]
    else: 
        trigger_eliminated_days = []
    trigger_opt_dict = {'trigger_results': trigger_results,
                        'trigger_grid_search': grid_df,
                        'opt_trigger_params': opt_trigger_params,
                        'trigger_qualified_days': trigger_qualified_days,
                        'trigger_eliminated_days': trigger_eliminated_days
                        }
    
    return trigger_opt_dict


