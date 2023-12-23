#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 11:04:16 2023

@author: Nick
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import Indicator_Building_Blocks as ind
import temporal_backend as tb

def max_annualized_return_given_signif_geom_mean(df, geom_mean_cutoff=1.001):
    '''
    Given a dataframe that reports both the geometric mean and annualized return of a strategy, this functions selects the row with the 
    highest annualized return, given that the geometric mean of the cumulative returns is above 1.001, or a given value
    '''
    filtered_df = df[df['geom_mean'] > geom_mean_cutoff] # Filter rows where 'geom_mean' is greater than 1.001
    if filtered_df.empty:
        return None  # No rows meet the criteria
    # Find the row with the highest 'annualized_returns' value
    max_annualized_return_row = filtered_df[filtered_df['annualized_returns'] == filtered_df['annualized_returns'].max()]
    return max_annualized_return_row

############################################################################################################################################
'''
HTNB Logic Grouping of Buys: 
'''
############################################################################################################################################

def htnb_logic_grouping_buys(threshold_df, opt_df, lag, temporal_granularity):
    '''
    Hold Til No Buy (HTNB) logic dictates that for any buy executed, successive buy orders within the lag period do not actually execute,
    but rather extend the sell time to that of the most recent buy order.
    This function groups consecutive buy orders together and calculates the returns based on HTNB logic.
    Parameters
    ----------
    threshold_df : TYPE pandas.DataFrame
        DESCRIPTION. All rows of historical stock data, model features, outcome variable and predictions values where the prediction
        values are high enough to merit a buy order. i.e. model.predit(X) > rounding_threshold
    opt_df : TYPE pandas.DataFrame
        DESCRIPTION. Dataframe of all stock data which includes purchase/sale price/times & returns should a buy order be executed at
        any given minute and the stock sold 'lag' number of minutes later.
    lag : TYPE integer
        DESCRIPTION. The number of minutes over which the model is trained to predict growth. 
    temporal_granularity : TYPE string
        DESCRIPTION. Duration of aggregate stock data bars used to generate the data.

    Returns
    -------
    htnb_buy_log : TYPE pandas.DataFrame
        DESCRIPTION. A log of the returns, times and prices for each purchase and sale, adhering to HTNB logic.
    buy_groupings : TYPE list of objects of type pandas.DataFrame
        DESCRIPTION. Groups of rows of data that generated buy orders which would collectively have resulted 
        in a single purchase and sale based on HTNB logic.

    '''
    
    ## Grouping overlapping buy orders based on lag time and temporal_granularity
    timing_df = threshold_df.copy()
    buy_groupings = []
    while not timing_df.empty:
        current_row = timing_df.iloc[0]
        time_range = (current_row['timemerge_dt'] - timedelta(minutes=float(lag *tb.hold_period_expansion[temporal_granularity])), 
                      current_row['timemerge_dt'] + timedelta(minutes=float(lag *tb.hold_period_expansion[temporal_granularity])))
        grouped_rows = timing_df[(timing_df['timemerge_dt'] >= time_range[0]) & (timing_df['timemerge_dt'] <= time_range[1])]
        prev_length = 1
        timing_df = timing_df.drop(grouped_rows.index)
        timing_df = timing_df.reset_index(drop=True)
        while prev_length != len(grouped_rows):
            first_row = grouped_rows.iloc[0]
            last_row = grouped_rows.iloc[-1]
            new_end_time = last_row['timemerge_dt'] + timedelta(minutes=float(lag *tb.hold_period_expansion[temporal_granularity]))
            previous_end_time = first_row['timemerge_dt'] + timedelta(minutes=float(lag *tb.hold_period_expansion[temporal_granularity]))
            new_rows = timing_df[(timing_df['timemerge_dt'] >= previous_end_time) & (timing_df['timemerge_dt'] <= new_end_time)]
            prev_length = len(grouped_rows)
            grouped_rows = pd.concat([grouped_rows, new_rows])
            timing_df = timing_df.drop(new_rows.index)
            timing_df = timing_df.reset_index(drop=True)
        if not any(group.equals(grouped_rows) for group in buy_groupings):  # Ensure no duplicate dataframes in the buy_groupings list
            buy_groupings.append(grouped_rows.reset_index(drop=True))

    htnb_returns = []
    htnb_buy_price = []
    htnb_sell_price = []
    htnb_buy_time = []
    htnb_sell_time = []
    htnb_day = []
    
    ## Calculate returns such that the purchase executes on the first buy order and sale executes on the final buy order's sell time
    for buy in buy_groupings:
        # buy = buy_groupings[0]
        # print(f'Currently testing buy grouping with initial purchase at {buy.timemerge_dt.head(1)[0]}')
        day_specific_df = opt_df[opt_df['day'] == buy.day.unique()[0]] # Grab historical data for only the day of the buy
        # print(f'day_specific_df index running from {day_specific_df.index.min()} to {day_specific_df.index.max()}.')
        first_buy_time = buy['timemerge_dt'][0] # Identify the time of the first purchase, saved as a datetime object
        # print(f'First time buy variable of value {first_buy_time}, of type {type(first_buy_time)}')
        start_index = day_specific_df[day_specific_df['timemerge_dt'] == first_buy_time].index[0] # Identify the index within the day-specific dataframe that matches the datetime of the first buy
        last_buy_time = buy['timemerge_dt'].iloc[-1]
        last_buy_index = day_specific_df[day_specific_df['timemerge_dt'] == last_buy_time].index[0]
        purchase_index = start_index+1
        sale_index = last_buy_index + lag + 1
        # For sale times after market hours, sell at the final minute of the day
        if sale_index >= max(day_specific_df.index):
            sale_index = max(day_specific_df.index)
        purchase_index = start_index+1
        if purchase_index >= max(day_specific_df.index):
            purchase_index = max(day_specific_df.index)
        
        # Estimate a purchase price and log the time of purchase
        high_at_buy = opt_df.iloc[purchase_index]['high']
        open_at_buy = opt_df.iloc[purchase_index]['open']
        buy_price = tb.calculate_purchase_price(temporal_granularity, open_at_buy, high_at_buy)
        htnb_buy_time.append(opt_df.iloc[purchase_index]['timemerge_dt'])
        htnb_buy_price.append(buy_price)
        
        # Estimate a sale price and log the time of sale
        low_at_sale = opt_df.iloc[sale_index]['low']
        open_at_sale = opt_df.iloc[sale_index]['open']
        sale_price = tb.calculate_sale_price(temporal_granularity, open_at_sale, low_at_sale)
        htnb_sell_price.append(sale_price)
        htnb_sell_time.append(opt_df.iloc[sale_index]['timemerge_dt'])
        
        htnb_returns.append(sale_price / buy_price)
        htnb_day.append(opt_df.iloc[sale_index]['day'])
        
    # Create a buy log outlining the performance of each buy order and subsequent sale
    htnb_buy_log = pd.DataFrame({'returns': htnb_returns,
                                 'buy_price': htnb_buy_price,
                                 'sell_price': htnb_sell_price,
                                 'buy_time': htnb_buy_time,
                                 'sell_time': htnb_sell_time,
                                 'day': htnb_day})
    
    return htnb_buy_log, buy_groupings


############################################################################################################################################
'''
Rounding Threshold Optimization: 
'''
############################################################################################################################################

def opt_rounding_thresh_htnb_logic(trade_strat_opt_data, lag, nn_model, temporal_granularity, req_buys_pday=1/20, allowed_buys_pday=100):
    '''
    This function finds the optimal threshold for model prediction values that maximizes annualized returns.
    Default grid spaces assume model prediction values fall between 0 and 1.
    
    Parameters
    ----------
    trade_strat_opt_data : TYPE pandas.DataFrame
        DESCRIPTION. The data selected to be used to optimize the trade strategy with, including X and y. 
        Usually a subset of the  output 'Xy' from cleaned_to_X() in optimization_pipleine.py
    lag : TYPE integer
        DESCRIPTION. The number of minutes over which the model is trained to predict growth. 
    nn_model : TTYPE keras.src.engine.functional.Functional (or any type of model)
        DESCRIPTION. A neural net model trained on the features present in trade_strat_opt
    temporal_granularity : TYPE string
        DESCRIPTION. Duration of aggregate stock data bars used to generate the data.
        From this point on in the pipeline, only 'one_minute' is allowed/supported
    req_buys_pday : TYPE, Float
        DESCRIPTION. The required number of buys per day that a threshold must generate to be 'in the running'.
    allowed_buys_pday : TYPE, integer
        DESCRIPTION. The maximum number of buys per day allowed, helping to eliminate thresholds that generate excessive buy volume. 

    Returns
    -------
    round_thresh_opt_results : TYPE dictionary
        DESCRIPTION. The results of the rounding threshold optimization process.

    '''
    ## Create dataframe to assess prediction output rounding threshold from
    assessment_df = trade_strat_opt_data.copy()
    assessment_df['preds'] = nn_model.predict(assessment_df.drop(columns = ['lag_growth', 'close_actual', 'open', 'high', 'low', 'timemerge', 'timemerge_dt', 'day', 'growth_ind']))
    assessment_df['timemerge_dt'] = pd.to_datetime(assessment_df['timemerge_dt'])
    ## Create a copy to allow for days to be dropped without affecting the original
    
    ## Setup for a shrinking grid searh
    all_thresh_perf = pd.DataFrame()
    step_size = 0.3
    zero_improvement_count = 0
    threshold_differentiation = True
    previous_annualized_return = 0
    optimal_thresh = 0.6
    performance_delta = 1
    
    while (zero_improvement_count < 5) and (threshold_differentiation == True): # Continue until 5 rounds with no improvement or uniform results are achieved amongst all thresholds tested
        opt_df = assessment_df.copy()
        days_need_dropping = True
        all_days_dropped = []
        first_round = True   
        
        while days_need_dropping == True: # If days are dropped at the end of the optimization round, perform another optimization round
            thresh_results = []    
            days_need_dropping = False
            total_days = opt_df['timemerge_dt'].dt.date.nunique()
            for thresh in np.arange((optimal_thresh - step_size), (optimal_thresh + step_size), step_size/5): 
                threshold_df = opt_df[opt_df['preds'] > thresh] # Identify every minute which would have triggered a buy order
                if not threshold_df.empty: # If any buy orders exist given the current threshold...
                    htnb_buy_log, buy_groupings = htnb_logic_grouping_buys(threshold_df, opt_df, lag, temporal_granularity) # Produce a buy log and buy groupings
                    htnb_logic_returns = np.array(htnb_buy_log.returns)
                    total_trades = len(htnb_logic_returns)
                  
                    if (total_trades / total_days) > req_buys_pday: # If the threshold produces the minimum trades/day required...
                        total_return = htnb_logic_returns.prod()
                        conf_interval = np.percentile(htnb_logic_returns, [2.5, 97.5]) # 95% confidence interval of returns
                        geom_mean = total_return ** (1/total_trades)
                        yagar_ratio = geom_mean / (conf_interval[1]-conf_interval[0]) / total_trades # A custom metric balancing geom_mean, size of std dev and total number of trades
                        annual_return = geom_mean ** ((total_trades/total_days) * 250)
                        day_count_df = pd.concat([df.iloc[[0]] for df in buy_groupings], ignore_index=True) # Grabbing the first row of each buy grouping
                        day_counts = day_count_df['day'].value_counts().to_dict() # Recording the number of purchases per day, as a dictionary
                        thresh_results.append({'threshold': thresh, 
                                               'trades/day':total_trades / total_days, 
                                               'total_trades':total_trades, 
                                               'total_days': total_days, 
                                               'geom_mean':geom_mean, 
                                               'positive_precision': sum(htnb_logic_returns>1) / len(htnb_logic_returns),
                                               'conf_int':conf_interval, 
                                               'yagar_ratio':yagar_ratio,
                                               'day_counts':day_counts,
                                               'total_return':total_return,
                                               'annualized_return': annual_return})
            
            ## Assimilate results and identify the optimal threshold value 
            if len(thresh_results) != 0:
                thresh_performance = pd.concat([pd.DataFrame([result]) for result in thresh_results], ignore_index=True)  
                if len(thresh_performance) > 1:
                    threshold_differentiation = (thresh_performance.total_trades.nunique() > 1)
                all_thresh_perf = pd.concat([all_thresh_perf,thresh_performance]).reset_index(drop=True)
            else:
                threshold_differentiation = False
            optimal_row = all_thresh_perf.sort_values(by=['annualized_return', 'geom_mean', 'threshold'], ascending=[False, False, False]).head(1)
            optimal_thresh = optimal_row['threshold'].iloc[0]
            
            ## Record the first round's buy groupings for reference, in case 'freak' days are eliminated below
            interim_threshold_df = opt_df[opt_df['preds'] > optimal_thresh]
            interim_buy_log, interim_buy_groupings = htnb_logic_grouping_buys(interim_threshold_df, opt_df, lag, temporal_granularity)
            optimal_thresh_buys_per_grouping = {df.iloc[0]['day']: len(df) for df in interim_buy_groupings}
            
            # Record the first round's buy groupings for reference, in case 'freak' days are eliminated below
            if first_round == True:
                original_thresh_buys_per_grouping = optimal_thresh_buys_per_grouping
                first_round = False
            
            ## Eliminate 'freak' growth days so as to choose a threshold with high precision in 'normal' conditions
            # Shrinking grid search mechanics essentially eliminated this functionality, but leaving it allows for future use
            if (max(optimal_thresh_buys_per_grouping.values()) > allowed_buys_pday):
                days_need_dropping = True
                for day, count in optimal_thresh_buys_per_grouping.items():
                    if (count > allowed_buys_pday):
                        all_days_dropped.append(day)
                opt_df = opt_df[~opt_df['day'].isin(all_days_dropped)].reset_index(drop=True)
            
        performance_delta = optimal_row['annualized_return'].iloc[0] - previous_annualized_return
        if performance_delta == 0:
            zero_improvement_count += 1
        previous_annualized_return = optimal_row['annualized_return'].iloc[0]
        step_size = step_size ** 1.75
        
    ## After exiting while loop, record optimal threshold performance and performance record for all thresholds
    round_thresh_opt_results = {'optimal_threshold':optimal_thresh,
                                'optimal_threshold_performance': optimal_row,
                                'possible_buys': interim_threshold_df,
                                'opt_thresh_buy_log': interim_buy_log,
                                'opt_thresh_buy_groupings': interim_buy_groupings, 
                                'each_thresholds_performance': all_thresh_perf,
                                'dropped_days': all_days_dropped,
                                'first_round_buys_pgrouping': original_thresh_buys_per_grouping, # Confirms days dropped indeed should have been dropped
                                'final_round_buys_pgrouping': optimal_thresh_buys_per_grouping, 
                                'assessment_df': opt_df}
    return round_thresh_opt_results


############################################################################################################################################
'''
    HOLD UNTIL NO BUYS STRATEGY PERFORMANCE:
'''
############################################################################################################################################

def htnb(opt_df, optimal_thresh, lag, temporal_granularity, apply_trigger = False):
    '''
    A function to calculate the performance of this baseline strategy, based on HTNB logic alone.

    Parameters
    ----------
    opt_df : TYPE pandas.DataFrame
        DESCRIPTION. The 'assessment_df' output from opt_rounding_thresh_htnb_logic()
    optimal_thresh : TYPE float
        DESCRIPTION. The optimal rounding threshold above which prediction values trigger a stock purchase.
    lag : TYPE integer
        DESCRIPTION. The number of minutes over which the model is trained to predict growth. 
    temporal_granularity : TYPE string
        DESCRIPTION. Duration of aggregate stock data bars used to generate the data.
        From this point on in the pipeline, only 'one_minute' is allowed/supported
    apply_trigger : TYPE, boolean
        DESCRIPTION. If true, the performance will be recorded as having been trigger adjusted.

    Returns
    -------
    htnb_performance_dict : TYPE dictionary
        DESCRIPTION. The results of the strategy's performance.

    '''
    ## Select all rows that generated buys and then estimate the returns
    threshold_df = opt_df[opt_df['preds'] > optimal_thresh]
    total_days = opt_df.day.nunique()
    
    htnb_buy_log, buy_groupings = htnb_logic_grouping_buys(threshold_df, opt_df, lag, temporal_granularity) # Produce a buy log and buy groupings
    htnb_logic_returns = np.array(htnb_buy_log.returns)
    total_trades = len(htnb_logic_returns)
    total_return = htnb_logic_returns.prod()
    conf_interval = np.percentile(htnb_logic_returns, [2.5, 97.5]) # 95% confidence interval of returns
    geom_mean = total_return ** (1/total_trades)
    annual_return = geom_mean ** ((total_trades/total_days) * 250)
    htnb_performance = {
        'buys_pday':total_trades / total_days, 
        'num_buys':total_trades, 
        'total_days': total_days, 
        'geom_mean':geom_mean, 
        'pos_precision': sum(htnb_logic_returns>1) / len(htnb_logic_returns),
        'model_precision': sum(threshold_df.growth_ind) / len(htnb_logic_returns),
        'conf_int':conf_interval, 
        'total_return':total_return,
        'annualized': annual_return
        }
    
    htnb_performance['hold_strategy'] = 'hold_til_no_buys'
    htnb_performance['sell_strategy'] = 'None'
                             
    
    optimal_hold_strat_params = {
        'rounding_threshold': optimal_thresh,
        'hold_strategy_employed': 'hold_til_no_buys'
        }
    
    # For datasets reduced by the trigger optimization, the strategy type is renamed
    if apply_trigger == True:
        htnb_performance['hold_strategy'] = 'trig_hold_til_no_buys'
    
    htnb_performance_dict = {
        'hold_strategy_performance':htnb_performance, 
        'trade_log_df':htnb_buy_log, 
        'hold_periods':buy_groupings,
        'optimal_hold_strat_params': optimal_hold_strat_params
        }
    
    return htnb_performance_dict


############################################################################################################################################
'''
    HOLD UNTIL NO BUYS AND NEG STRATEGY PERFORMANCE:
'''
############################################################################################################################################

def htnb_neg(rounding_threshold_results, lag, apply_trigger = False):
    '''
    A function that calculates the performance of holding onto a stock once purchased until the standard
    HTNB logic sale time, but additionally holding until the stock has negative growth.
    
    Parameters
    ----------
    round_thresh_opt_results : TYPE dictionary
        DESCRIPTION. The results of the rounding threshold optimization process, from opt_rounding_thresh_htnb_logic().
    lag : TYPE integer
        DESCRIPTION. The number of minutes over which the model is trained to predict growth. 
    apply_trigger : TYPE, boolean
        DESCRIPTION. If true, the performance will be recorded as having been trigger adjusted.

    Returns
    -------
    htnb_neg_performance_dict : TYPE dictionary
        DESCRIPTION. The results of the strategy's performance.

    '''  
    thresh = rounding_threshold_results['optimal_threshold']
    assessment_df = rounding_threshold_results['assessment_df']
    possible_buys = rounding_threshold_results['possible_buys']
    
    preds_df = assessment_df.copy()[['close', 'close_actual', 'low', 'high','growth_rate','day','timemerge_dt','preds', 'growth_ind']]

    total_days = preds_df.day.nunique()

    # for thresh in np.arange(0.6,0.75,0.0005):
    thresh_df = preds_df[preds_df['preds'] > thresh]
    
    buy_groupings = []
    timing_df = thresh_df.copy().reset_index(drop=True)
    # hold_duration = holds[5]
    while not timing_df.empty:
        current_row = timing_df.iloc[0]
        time_range = (current_row['timemerge_dt'] - timedelta(minutes=float(lag)), current_row['timemerge_dt'] + timedelta(minutes=float(lag)))
        grouped_rows = timing_df[(timing_df['timemerge_dt'] >= time_range[0]) & (timing_df['timemerge_dt'] <= time_range[1])]
        prev_length = 1
        timing_df = timing_df.drop(grouped_rows.index)
        timing_df = timing_df.reset_index(drop=True)
        while prev_length != len(grouped_rows):
            first_row = grouped_rows.iloc[0]
            last_row = grouped_rows.iloc[-1]
            new_end_time = last_row['timemerge_dt'] + timedelta(minutes=float(lag))
            previous_end_time = first_row['timemerge_dt'] + timedelta(minutes=float(lag))
            new_rows = timing_df[(timing_df['timemerge_dt'] >= previous_end_time) & (timing_df['timemerge_dt'] <= new_end_time)]
            prev_length = len(grouped_rows)
            grouped_rows = pd.concat([grouped_rows, new_rows])
            timing_df = timing_df.drop(new_rows.index)
            timing_df = timing_df.reset_index(drop=True)
        # Ensure no duplicate dataframes in the buy_groupings list
        if not any(group.equals(grouped_rows) for group in buy_groupings):
            buy_groupings.append(grouped_rows.reset_index(drop=True))
    
    buy_groupings = [df for df in buy_groupings if df.iloc[0]['timemerge_dt'] in possible_buys['timemerge_dt'].values]
    
    htnb_neg_returns = []
    htnb_neg_buy_index = []
    htnb_neg_hold_periods = []
    
    for buy in buy_groupings:
        # buy = buy_groupings[2]
        day_specific_df = preds_df[preds_df['day'] == buy.day.unique()[0]]
        first_buy_time = buy['timemerge_dt'][0]
        start_index = day_specific_df[day_specific_df['timemerge_dt'] == first_buy_time].index[0]
        last_buy_time = buy['timemerge_dt'].iloc[-1]
        last_buy_index = day_specific_df[day_specific_df['timemerge_dt'] == last_buy_time].index[0]
        end_index = last_buy_index + lag + 1
        if end_index < max(day_specific_df.index):
            hold_index = end_index
            # Extend the sale at final_end_index until the end of day or if growth is negative
            while hold_index+1 < max(day_specific_df.index) and preds_df.iloc[hold_index]['growth_rate'] > 0:
                hold_index += 1
            final_end_index = hold_index + 1 
            
            ## Given defined buy/sell indices, pull pricing and returns
            sale_price = preds_df.iloc[final_end_index]['low']
            buy_price = preds_df.iloc[start_index]['high']
            htnb_neg_returns.append(sale_price / buy_price)
            htnb_neg_buy_index.append(start_index)
    
            # Append hold period
            htnb_neg_hold_periods.append(preds_df[start_index:hold_index])
    
    # Use recorded returns and indices to record performance
    htnb_neg_buys = preds_df.loc[htnb_neg_buy_index]
    htnb_neg_num_buys = len(htnb_neg_returns)
    htnb_neg_buy_pday = htnb_neg_num_buys / total_days
    htnb_neg_return = np.array(htnb_neg_returns).prod()
    htnb_neg_geom_mean = htnb_neg_return ** (1/htnb_neg_num_buys)
    htnb_neg_conf_int = np.percentile(htnb_neg_returns, [2.5, 97.5])
    htnb_neg_annualized = htnb_neg_geom_mean ** ((250 / total_days) * htnb_neg_num_buys)
    htnb_neg_model_precision = sum(htnb_neg_buys['growth_ind']) / htnb_neg_num_buys
    htnb_neg_pos_precision = sum(np.array(htnb_neg_returns) > 1) / htnb_neg_num_buys
    
    htnb_neg_performance  = {'hold_strategy':'hold_til_no_buys_til_neg',
                                    'sell_strategy': 'None',
                                    'num_buys':htnb_neg_num_buys,
                                    'total_days': total_days,
                                    'buys_pday':htnb_neg_buy_pday, 
                                    'geom_mean':htnb_neg_geom_mean, 
                                    'total_return':htnb_neg_return, 
                                    'conf_int':htnb_neg_conf_int,
                                    'model_precision':htnb_neg_model_precision,
                                    'pos_precision':htnb_neg_pos_precision,
                                    'annualized':htnb_neg_annualized}
    # For datasets reduced by the trigger optimization, the strategy type is renamed
    if apply_trigger == True:
        htnb_neg_performance['hold_strategy'] = 'trig_hold_til_no_buys_til_neg'
    
    htnb_neg_trade_log_dict = {'preds':htnb_neg_buys['preds'], 'returns':htnb_neg_returns, 'day':htnb_neg_buys['day'], 'timemerge_dt':htnb_neg_buys['timemerge_dt'], 'growth_ind':htnb_neg_buys['growth_ind']}
    htnb_neg_trade_log_df = pd.DataFrame(htnb_neg_trade_log_dict)
    
    optimal_hold_strat_params = {'rounding_threshold': thresh,
                                 'hold_strategy_employed': 'hold_til_no_buys_til_neg'}
    
    htnb_neg_performance_dict = {'hold_strategy_performance':htnb_neg_performance, 
                                 'trade_log_df':htnb_neg_trade_log_df, 
                                 'hold_periods':htnb_neg_hold_periods,
                                 'optimal_hold_strat_params': optimal_hold_strat_params}
    
    return htnb_neg_performance_dict
    

############################################################################################################################################
'''
    BASELINE WITH ADDITONAL HOLD STRATEGY PERFORMANCE:
'''
############################################################################################################################################

def buy_w_hold(rounding_threshold_results, lag, holds=range(15), apply_trigger = False):
    '''
    This function estimates the performance of the strategy which follows HTNB logic with an additional hold period.

    Parameters
    ----------
    round_thresh_opt_results : TYPE dictionary
        DESCRIPTION. The results of the rounding threshold optimization process, from opt_rounding_thresh_htnb_logic().
    lag : TYPE integer
        DESCRIPTION. The number of minutes over which the model is trained to predict growth. 
    holds : TYPE range
        DESCRIPTION. The hold durations to test in the strategy. The default is range(15).
    apply_trigger : TYPE, boolean
        DESCRIPTION. If true, the performance will be recorded as having been trigger adjusted.

    Returns
    -------
    htnb_whold_performance_dict : TYPE dictionary
        DESCRIPTION. The results of the strategy's performance.

    '''
    thresh = rounding_threshold_results['optimal_threshold']
    assessment_df = rounding_threshold_results['assessment_df']
    possible_buys = rounding_threshold_results['possible_buys']
    # Organizing for easier use, then selecting all rows that generated buys
    preds_df = assessment_df.copy()[['close_actual', 'low', 'high','growth_rate','day','timemerge_dt','preds', 'growth_ind']]
    total_days = preds_df.day.nunique()
    thresh_df = preds_df[preds_df['preds'] > thresh]
    
    optimal_hold_tests = []
    
    for hold_duration in holds:
        ## Define time groupings based on a given hold duration
        buy_groupings = []
        timing_df = thresh_df.copy().reset_index(drop=True)
        # hold_duration = holds[5]
        while not timing_df.empty:
            current_row = timing_df.iloc[0]
            time_range = (current_row['timemerge_dt'] - timedelta(minutes=float(lag+hold_duration)), current_row['timemerge_dt'] + timedelta(minutes=float(lag+hold_duration)))
            grouped_rows = timing_df[(timing_df['timemerge_dt'] >= time_range[0]) & (timing_df['timemerge_dt'] <= time_range[1])]
            prev_length = 1
            timing_df = timing_df.drop(grouped_rows.index)
            timing_df = timing_df.reset_index(drop=True)
            while prev_length != len(grouped_rows):
                first_row = grouped_rows.iloc[0]
                last_row = grouped_rows.iloc[-1]
                new_end_time = last_row['timemerge_dt'] + timedelta(minutes=float(lag+hold_duration))
                previous_end_time = first_row['timemerge_dt'] + timedelta(minutes=float(lag+hold_duration))
                new_rows = timing_df[(timing_df['timemerge_dt'] >= previous_end_time) & (timing_df['timemerge_dt'] <= new_end_time)]
                prev_length = len(grouped_rows)
                grouped_rows = pd.concat([grouped_rows, new_rows])
                timing_df = timing_df.drop(new_rows.index)
                timing_df = timing_df.reset_index(drop=True)
            # Ensure no duplicate dataframes in the buy_groupings list
            if not any(group.equals(grouped_rows) for group in buy_groupings):
                buy_groupings.append(grouped_rows.reset_index(drop=True))
        
        buy_groupings = [df for df in buy_groupings if df.iloc[0]['timemerge_dt'] in possible_buys['timemerge_dt'].values]
        
        ## Calculate returns from generated hold periods defined in buy_groupings
        htnb_whold_returns = []
        htnb_whold_buy_index = []
        htnb_whold_hold_periods = []
        
        ## Generate hold periods given a time grouping
        for buy in buy_groupings:
            day_specific_df = preds_df[preds_df['day'] == buy.day.unique()[0]]
            first_buy_time = buy['timemerge_dt'][0]
            start_index = day_specific_df[day_specific_df['timemerge_dt'] == first_buy_time].index[0]
            last_buy_time = buy['timemerge_dt'].iloc[-1]
            last_buy_index = day_specific_df[day_specific_df['timemerge_dt'] == last_buy_time].index[0]
            end_index = last_buy_index + lag + hold_duration + 1
            if end_index >= max(day_specific_df.index):
                end_index = max(day_specific_df.index)
            purchase_index = start_index+1
            if purchase_index >= max(day_specific_df.index):
                purchase_index = max(day_specific_df.index)
            sale_price = day_specific_df.loc[end_index]['low']
            buy_price = day_specific_df.loc[purchase_index]['high']
            htnb_whold_returns.append(sale_price / buy_price)
            htnb_whold_buy_index.append(start_index)
            htnb_whold_hold_periods.append(day_specific_df.loc[start_index:end_index])
    
        htnb_whold_num_buys = len(htnb_whold_returns)    
        htnb_whold_return = np.array(htnb_whold_returns).prod()
        htnb_whold_geom_mean = htnb_whold_return ** (1/htnb_whold_num_buys)
        htnb_whold_annualized = htnb_whold_geom_mean ** ((htnb_whold_num_buys / total_days) * 250) 
        optimal_hold_tests.append({'hold_duration': hold_duration, 
                                   'total_return':htnb_whold_return, 
                                   'geom_mean':htnb_whold_geom_mean, 
                                   'annualized_returns': htnb_whold_annualized})
    
    # With results from each hold duration, combine them and find the ideal hold period
    optimal_hold_tests_df = pd.DataFrame(optimal_hold_tests)
    optimal_dict = max_annualized_return_given_signif_geom_mean(optimal_hold_tests_df, geom_mean_cutoff=0).reset_index(drop=True)
    opt_hold_duration = optimal_dict['hold_duration'][0]
    
    ## With optimal hold, regroup buys and calculate the performance of the strategy the same as above
    timing_df = thresh_df.copy()
    buy_groupings = []

    while not timing_df.empty:
        current_row = timing_df.iloc[0]
        time_range = (current_row['timemerge_dt'] - timedelta(minutes=float(lag+opt_hold_duration)), current_row['timemerge_dt'] + timedelta(minutes=float(lag+opt_hold_duration)))
        grouped_rows = timing_df[(timing_df['timemerge_dt'] >= time_range[0]) & (timing_df['timemerge_dt'] <= time_range[1])]
        prev_length = 1
        timing_df = timing_df.drop(grouped_rows.index)
        timing_df = timing_df.reset_index(drop=True)
        while prev_length != len(grouped_rows):
            first_row = grouped_rows.iloc[0]
            last_row = grouped_rows.iloc[-1]
            new_end_time = last_row['timemerge_dt'] + timedelta(minutes=float(lag+opt_hold_duration))
            previous_end_time = first_row['timemerge_dt'] + timedelta(minutes=float(lag+opt_hold_duration))
            new_rows = timing_df[(timing_df['timemerge_dt'] >= previous_end_time) & (timing_df['timemerge_dt'] <= new_end_time)]
            prev_length = len(grouped_rows)
            grouped_rows = pd.concat([grouped_rows, new_rows])
            timing_df = timing_df.drop(new_rows.index)
            timing_df = timing_df.reset_index(drop=True)
        # Ensure no duplicate dataframes in the buy_groupings list
        if not any(group.equals(grouped_rows) for group in buy_groupings):
            buy_groupings.append(grouped_rows.reset_index(drop=True)) 

    buy_groupings = [df for df in buy_groupings if df.iloc[0]['timemerge_dt'] in possible_buys['timemerge_dt'].values]    

    ## With optimal hold, redefine hold periods
    htnb_whold_returns = []
    htnb_whold_buy_index = []
    htnb_whold_hold_periods = []
    
    for buy in buy_groupings:
        # buy = buy_groupings[0]
        day_specific_df = preds_df[preds_df['day'] == buy.day.unique()[0]]
        first_buy = buy['timemerge_dt'][0]
        start_index = day_specific_df[day_specific_df['timemerge_dt'] == first_buy].index[0]
        last_buy = buy['timemerge_dt'].iloc[-1]
        last_buy_index = day_specific_df[day_specific_df['timemerge_dt'] == last_buy].index[0]
        end_index = last_buy_index + lag + opt_hold_duration + 1
        if end_index >= max(day_specific_df.index):
            end_index = max(day_specific_df.index)
        purchase_index = start_index+1
        if purchase_index >= max(day_specific_df.index):
            purchase_index = max(day_specific_df.index)
        sale_price = day_specific_df.loc[end_index]['low']
        buy_price = day_specific_df.loc[purchase_index]['high']
        htnb_whold_returns.append(sale_price / buy_price)
        htnb_whold_buy_index.append(start_index)
        htnb_whold_hold_periods.append(day_specific_df.loc[start_index:end_index])
    
    htnb_whold_buys = preds_df.loc[htnb_whold_buy_index]
    
    # Optimal hold strategy performance calculations
    htnb_whold_num_buys = len(htnb_whold_returns)
    htnb_whold_buy_pday = htnb_whold_num_buys / total_days
    htnb_whold_return = np.array(htnb_whold_returns).prod()
    htnb_whold_geom_mean = htnb_whold_return ** (1/htnb_whold_num_buys)
    htnb_whold_conf_int = np.percentile(htnb_whold_returns, [2.5, 97.5])
    htnb_whold_annualized = htnb_whold_geom_mean ** ((250 / total_days) * htnb_whold_num_buys)
    htnb_whold_model_precision = sum(htnb_whold_buys['growth_ind']) / htnb_whold_num_buys
    htnb_whold_pos_precision = sum(np.array(htnb_whold_returns) > 1) / htnb_whold_num_buys
    
    htnb_whold_performance  = {'hold_strategy':'hold_til_no_buys_with_hold',
                                    'sell_strategy': 'None',
                             'num_buys':htnb_whold_num_buys,
                             'total_days': total_days,
                             'buys_pday':htnb_whold_buy_pday, 
                             'geom_mean':htnb_whold_geom_mean, 
                             'total_return':htnb_whold_return, 
                             'conf_int':htnb_whold_conf_int,
                             'model_precision':htnb_whold_model_precision,
                             'pos_precision':htnb_whold_pos_precision,
                             'annualized':htnb_whold_annualized}
    
    # For datasets reduced by the trigger optimization, the strategy type is renamed
    if apply_trigger == True:
        htnb_whold_performance['hold_strategy'] = 'trig_hold_til_no_buys_with_hold'
        hold_strategy_employed = 'trig_hold_til_no_buys_with_hold'
    else:
        hold_strategy_employed = 'hold_til_no_buys_with_hold'

    htnb_whold_trade_log_dict = {'preds':htnb_whold_buys['preds'], 'returns':htnb_whold_returns, 'day':htnb_whold_buys['day'], 'timemerge_dt':htnb_whold_buys['timemerge_dt'], 'growth_ind':htnb_whold_buys['growth_ind']}
    htnb_whold_trade_log_df = pd.DataFrame(htnb_whold_trade_log_dict)
    
    optimal_hold_strat_params = {'rounding_threshold': thresh,
                                 'hold_strategy_employed': hold_strategy_employed,
                                 'hold': opt_hold_duration}
    
    htnb_whold_performance_dict = {'hold_strategy_performance':htnb_whold_performance, 
                                   'trade_log_df':htnb_whold_trade_log_df, 
                                   'hold_periods':htnb_whold_hold_periods,
                                   'optimal_hold_strat_params': optimal_hold_strat_params}
    
    return htnb_whold_performance_dict
    
