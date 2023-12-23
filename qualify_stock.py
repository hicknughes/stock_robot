#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script houses the 3 core functions needed to produce a predictive model, tune an optimal trading
strategy, and save that model and strategy for use in live trading.
"""

import pandas as pd
import numpy as np
import feature_origin as fo
import raw_data as raw
import Indicator_Building_Blocks as ind
import DEAP_ensemble as DPe
import DEAP_precision as DP
import optimization_pipeline as OP
import trigger as trig
import hold_strat as hs
import time
import datetime
from datetime import datetime as dt
import ast
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import tensorflow.keras.metrics
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, ReLU, Activation, LayerNormalization
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
import pickle
from keras.models import load_model
import os



def optimize_on_x_in_dict_list(list_of_dicts, x = 'annualized'):
    '''
    This function is used to select the optimal strategy parameters in optimal_trading_strategy() below.
    '''
    # Initialize variables to keep track of the maximum geom_mean and its index
    max_optimzation_parameter_value = float('-inf')
    max_optimzation_parameter_index = None
    
    # Iterate through the list of dictionaries
    for i, dictionary in enumerate(list_of_dicts):
        optimal_sell_strat_performance = dictionary.get('hold_strategy_performance')
        
        if optimal_sell_strat_performance is not None:
            current_dictionarys_max_value = optimal_sell_strat_performance[x].max()
            
            if current_dictionarys_max_value > max_optimzation_parameter_value:
                max_optimzation_parameter_value = current_dictionarys_max_value
                max_optimzation_parameter_index = i
    return list_of_dicts[max_optimzation_parameter_index]


def build_predictive_model(tkr, key="GpjjoRW_XKUaCLvWWurjuMUwF34oHvpD", days=(365*2-1), temporal_granularity='one_minute', deployment_model=True, pseudo_test_days=0, epochs=3, AutoKeras=False, population_scalar=10, clean_generations=1, assimilated_generations=3, max_trials=10, use_apex=True, use_apex_genesis=False):
    '''
    This function builds a predictive model for a given stock, employing genetic programming to create custom features

    Parameters
    ----------
    tkr : TYPE string
        DESCRIPTION. The ticker symbol of the stock of interest for the model
    key : TYPE string
        DESCRIPTION. The API key from polygon.io associated with the user's account.
    days : TYPE, integer
        DESCRIPTION. The number of calendar days of historical data desired.
    temporal_granularity : TYPE string
        DESCRIPTION. Duration of aggregate stock data bars used to generate the data.
        'one_minute', 'three_minute', 'five_minute', 'fifteen_minute', 'one_hour', or 'one_day' is allowed/supported
    deployment_model : TYPE boolean
        DESCRIPTION. True indicates that this model be used for live trading. This affects whether custom indicators will be saved or not.
    pseudo_test_days : TYPE integer
        DESCRIPTION. The number of most recent days to use as a custom test set.
    epochs : TYPE, integer
        DESCRIPTION. The number of epochs to train the neural net model with.
    AutoKeras : TYPE boolean
        DESCRIPTION. If true, Keras' AutoKeras AutoML package will be used to construct a neural net model
    population_scalar : TYPE, integer
        DESCRIPTION. For each feature parameter to be optimized within the genetic programming pipeline, this many individuals will be made per generation.
    clean_generations : TYPE, integer
        DESCRIPTION. The number of generations of features to test and evolve with flexible 'lag' and 'growth_threshold' values, after generation 0.
    assimilated_generations : TYPE, integer
        DESCRIPTION. The number of generations of features to test and evolve with fixed 'lag' and 'growth_threshold' values, after generation 0.
    max_trials : TYPE, integer
        DESCRIPTION. The number of different neural net model architectures that should be tested
    use_apex : TYPE, optional
        DESCRIPTION. If True, the first generation of custom features will be imbued with previous 'apex' indicators
    use_apex_genesis : TYPE, optional
        DESCRIPTION. If True, previous top performing custom indicators for the given temporal_granularity will be used in the first generation of custom features.

    Returns
    -------
    nn_model : TYPE keras.src.engine.functional.Functional
        DESCRIPTION. A neural net model trained on the defined training data set
    tkr_data : TYPE dictionary
        DESCRIPTION. The raw data, cleaned data for model training, and a list of both the days used for training and in the test set.
    merged_report : TYPE dictionary
        DESCRIPTION. A full NaN report for the data generation process
    cleaned_indicators : TYPE pandas.DataFrame
        DESCRIPTION. A dataframe of the best performing version of each custom indicator function. 
    selected_indicators : TYPE pandas.DataFrame
        DESCRIPTION. A dataframe of the 'top' indicators which have been selected for use in the predictive model.
    lag : TYPE integer
       DESCRIPTION. The number of minutes over which the model is trained to predict growth. 

    '''
    # Get today's date and date 1 year, 364 days ago
    today = datetime.date.today()
    end_date = today.strftime('%Y-%m-%d')
    start_date = today - datetime.timedelta(days=days)
    start_date = start_date.strftime('%Y-%m-%d')
    
    # Generate source data
    dataframe, nan_report_sourcedata = raw.raw_data(key, tkr, start_date, end_date, temporal_granularity, paid_polygon_account=False) # ~ 26 minute runtime for weekly dataranges
    print(f'number of days in dataframe: {dataframe.day.nunique()}')
    # Clean indicators
    start = time.time()
    cleaned_indicators = OP.clean_features(tkr, dataframe, temporal_granularity = temporal_granularity, troubleshooting=False, 
                                           deployment_model=deployment_model, pseudo_test_days=pseudo_test_days, 
                                           clean_generations=clean_generations, assimilated_generations=assimilated_generations,
                                           population_scalar=population_scalar, use_apex=use_apex, use_apex_genesis=use_apex_genesis)
    end = time.time()
    print(f'DEAP Duration: {(end-start) / 60}')
    
    # Generate X matrix and y vectors
    start = time.time()
    Xy, selected_indicators, nan_report_xmat = OP.cleaned_to_X(dataframe, cleaned_indicators, temporal_granularity, num_indicators=16)
    end = time.time()
    print(f'Clean_to_X() Duration: {(end-start) / 60}')
    lag = selected_indicators['lag'][0]
    
    # Merge NaN reports from the source data and in custom indicator generation
    merged_report = {}
    merged_report.update(nan_report_sourcedata)
    merged_report.update(nan_report_xmat)
    
    # Train Neural Net using AutoKeras' autoML pipeline or using a defined neural net architecture through standard_nn()
    if AutoKeras == True:
        nn_model, model_train_days, model_test_days = OP.AK(Xy, tkr, lag, temporal_granularity, max_trials=max_trials, epochs=epochs, deployment_model=deployment_model, pseudo_test_days=pseudo_test_days)
    elif AutoKeras == False:
        nn_model, model_train_days, model_test_days = OP.standard_nn(Xy, lag, temporal_granularity, epochs=epochs, deployment_model=deployment_model, pseudo_test_days=pseudo_test_days)
    
    tkr_data = {'raw_data': dataframe, 'master_Xy': Xy, 'model_trained_days': model_train_days, 'model_test_days': model_test_days}
    
    return nn_model, tkr_data, merged_report, cleaned_indicators, selected_indicators, lag

def optimal_trading_strategy(tkr, nn_model, tkr_data, lag, temporal_granularity='one_minute', key="GpjjoRW_XKUaCLvWWurjuMUwF34oHvpD", deployment_model=True, pseudo_test_days=0):
    '''
    Using the functions in hold_strat.py, this function builds optimal trading parameters to maximize annual returns.

    Parameters
    ----------
    tkr : TYPE string
        DESCRIPTION. The ticker symbol of the stock of interest for the model
    nn_model : TYPE keras.src.engine.functional.Functional
        DESCRIPTION. A neural net model trained on the defined training data set
    tkr_data : TYPE dictionary
        DESCRIPTION. The raw data, cleaned data for model training, and a list of both the days used for training and in the test set.
    lag : TYPE integer
       DESCRIPTION. The number of minutes over which the model is trained to predict growth. 
    temporal_granularity : TYPE string
        DESCRIPTION. Duration of aggregate stock data bars used to generate the data.
        From this point on in the pipeline, only 'one_minute' is allowed/supported
    key : TYPE string
        DESCRIPTION. The API key from polygon.io associated with the user's account.
    deployment_model : TYPE boolean
        DESCRIPTION. True indicates that this model be used for live trading. This affects whether custom indicators will be saved or not.
    pseudo_test_days : TYPE integer
        DESCRIPTION. The number of most recent days to use as a custom test set.

    Returns
    -------
    OOS_IS_comparison : TYPE pandas.DataFrame
        DESCRIPTION. If deployment_model == False, a dataframe is given to compare In-Sample and Out-Of-Sample performance of the model.
    intelligent_trade_opt_record : TYPE dictionary
        DESCRIPTION. A record of every trade strategy's performance.
    hold_strat_comparison : TYPE pandas.DataFrame
        DESCRIPTION. Every hold strategy's optimal parameter performance for rapid comparison.
    optimal_trade_strategy : TYPE pandas.DataFrame
        DESCRIPTION. A one-row dataframe of the optimal strategy's performance.

    '''
    ## Trade strategy optimization pipeline
    if deployment_model == True and pseudo_test_days == 0: #True deployment model
        trade_strat_opt_data = tkr_data['master_Xy'][tkr_data['master_Xy'].day.isin(tkr_data['model_trained_days'])].reset_index(drop=True)
    else: # A specified test set exists and is used to test the model and trading strategy's combined profitability
        trade_strat_opt_data = tkr_data['master_Xy'][tkr_data['master_Xy'].day.isin(tkr_data['model_trained_days'])].reset_index(drop=True)
        test_data = tkr_data['master_Xy'][tkr_data['master_Xy'].day.isin(tkr_data['model_test_days'])].reset_index(drop=True)
    
    rounding_threshold_results = hs.opt_rounding_thresh_htnb_logic(trade_strat_opt_data, lag, nn_model, temporal_granularity)## ~8-36 minutes, depending on when model produces less than 0.5 buys/day
    
    if deployment_model == False:
        test_set = test_data.copy()
        test_set['preds'] = nn_model.predict(test_set.drop(columns = ['lag_growth', 'close_actual', 'open', 'high', 'low', 'timemerge', 'timemerge_dt', 'day', 'growth_ind']))
        OOS_perf_stats_df = pd.DataFrame([hs.htnb(test_set, rounding_threshold_results['optimal_threshold'], lag, temporal_granularity)['hold_strategy_performance']])
        OOS_perf_stats_df['Data'] = 'Test_set'
        IS_perf_stats = rounding_threshold_results['optimal_threshold_performance'].copy()
        IS_perf_stats['Data'] = 'Train_set'
        common_columns = [
            'Data',
            'annualized_return',
            'total_return', 
            'geom_mean', 
            'positive_precision', 
            'trades/day', 
            'total_trades', 
            'total_days', 
            'conf_int', 
            ]
        OOS_IS_comparison= pd.concat([ OOS_perf_stats_df[common_columns], IS_perf_stats[common_columns]], axis=0, ignore_index=True)
        return OOS_IS_comparison, None, None
    
    trig_results = trig.trigger_opt(key, tkr, rounding_threshold_results, lag) ## ~12-17 minutes
    htnb_performance = hs.htnb(rounding_threshold_results['assessment_df'], rounding_threshold_results['optimal_threshold'], lag, temporal_granularity)  ## less than a second
    htnb_neg_performance = hs.htnb_neg(rounding_threshold_results, lag) ## less than a second
    buy_w_hold_performance = hs.buy_w_hold(rounding_threshold_results, lag) ## ~3 seconds
    hold_strat_record = [htnb_performance, htnb_neg_performance, buy_w_hold_performance]  
    ## Applying the trigger and re-optimizing
    if len(trig_results['trigger_eliminated_days']) != 0:
        #Eliminating the days identified in trig_results
        trig_rounding_threshold_results = rounding_threshold_results.copy()
        trig_rounding_threshold_results['possible_buys'] = trig_rounding_threshold_results['possible_buys'][trig_rounding_threshold_results['possible_buys']['day'].isin(trig_results['trigger_qualified_days'])]
        trig_rounding_threshold_results['assessment_df'] = trig_rounding_threshold_results['assessment_df'][trig_rounding_threshold_results['assessment_df']['day'].isin(trig_results['trigger_qualified_days'])]
        
        trig_htnb_performance = hs.htnb(rounding_threshold_results, lag, apply_trigger = True)
        trig_htnb_neg_performance = hs.htnb_neg(trig_rounding_threshold_results, lag, apply_trigger = True)
        trig_buy_w_hold_performance = hs.buy_w_hold(trig_rounding_threshold_results, lag, apply_trigger = True)
        hold_strat_record.extend([trig_htnb_performance, trig_htnb_neg_performance, trig_buy_w_hold_performance])
        
    ## Comparing and assimilating record from all strategies, then picking the optimal one
    hold_strat_comparison = pd.concat([pd.DataFrame([d['hold_strategy_performance']]) for d in hold_strat_record])
    hold_strat_comparison = hold_strat_comparison[['hold_strategy','sell_strategy','total_return', 'annualized','pos_precision', 'model_precision','buys_pday', 'num_buys', 'total_days', 'geom_mean','conf_int']]
    
    intelligent_trade_opt_record = {'rounding_threshold_results': rounding_threshold_results,
                                    'trigger_results': trig_results,
                                    'htnb_performance': htnb_performance,
                                    'htnb_neg_performance': htnb_neg_performance,
                                    'buy_w_hold_performance': buy_w_hold_performance,
                                    'hold_strat_comparison' : hold_strat_comparison}
    
    optimal_trade_strategy = optimize_on_x_in_dict_list(hold_strat_record)
    
    if 'trig' in optimal_trade_strategy['hold_strategy_performance']['hold_strategy']:
        # If it does, add key/value pairs from 'opt_trigger_params' to 'optimal_hold_strat_params'
        optimal_trade_strategy['optimal_hold_strat_params'].update(trig_results['opt_trigger_params'])    
    
    return intelligent_trade_opt_record, hold_strat_comparison, optimal_trade_strategy
    

def save_model_for_deployment(tkr, tkr_data, merged_report, cleaned_indicators, selected_indicators, intelligent_trade_opt_record, hold_strat_comparison, optimal_trade_strategy, nn_model):
    '''
    This function saves all the outputs from this script's pipeline to a local directory for deployment.

    Parameters
    ----------
    tkr : TYPE string
        DESCRIPTION. The ticker symbol of the stock of interest for the model
    tkr_data : TYPE dictionary
        DESCRIPTION. The raw data, cleaned data for model training, and a list of both the days used for training and in the test set.
    merged_report : TYPE dictionary
        DESCRIPTION. A full NaN report for the data generation process
    cleaned_indicators : TYPE pandas.DataFrame
        DESCRIPTION. A dataframe of the best performing version of each custom indicator function. 
    selected_indicators : TYPE pandas.DataFrame
        DESCRIPTION. A dataframe of the 'top' indicators which have been selected for use in the predictive model.
    intelligent_trade_opt_record : TYPE dictionary
        DESCRIPTION. A record of every trade strategy's performance.
    hold_strat_comparison : TYPE pandas.DataFrame
        DESCRIPTION. Every hold strategy's optimal parameter performance for rapid comparison.
    optimal_trade_strategy : TYPE pandas.DataFrame
        DESCRIPTION. A one-row dataframe of the optimal strategy's performance.
    nn_model : TYPE keras.src.engine.functional.Functional
        DESCRIPTION. A neural net model trained on the defined training data set.

    Returns
    -------
    output_dict : TYPE dictionary
        DESCRIPTION. A local record of the saved output dictionary.
    nn_model : TYPE
        DESCRIPTION. A local record of the saved predictive model.

    '''
    # Create output list 
    base_outputs = ['_data', '_report', '_cleaned_indicators', '_selected_indicators', 
                    '_intelligent_trade_opt_record', '_optimal_strat_comparison', '_optimal_trade_strategy']
    tkr_specific_outputs = [tkr+name for name in base_outputs]
    
    # Assign outputs
    outputs = tkr_data, merged_report, cleaned_indicators, selected_indicators, intelligent_trade_opt_record, hold_strat_comparison, optimal_trade_strategy
    output_dict = dict(zip(tkr_specific_outputs, outputs))
    
    today = datetime.date.today()
    # Create a deployment model folder with the name of 'tkr' combined with date of training start (year month and day)
    folder_name = f"{tkr}_{today.strftime('%Y-%m-%d')}"
    os.makedirs(folder_name, exist_ok=True)
    
    
    # Save 'output_dict' as a pickled object
    with open(os.path.join(folder_name, 'output_dict.pkl'), 'wb') as f:
        pickle.dump(output_dict, f)
    
    # Save the neural net model as an h5 file
    nn_model.save(os.path.join(folder_name, 'nn_model.h5'))
    
    return output_dict, nn_model



def save_data_and_indicators(tkr, temporal_granularity, dataframe, merged_report, cleaned_indicators, selected_indicators):
    '''
    This function saves the raw data and optimal custom features as an intermediate step for future use in development.

    Parameters
    ----------
   tkr : TYPE string
       DESCRIPTION. The ticker symbol of the stock of interest for the model
    temporal_granularity : TYPE string
        DESCRIPTION. Duration of aggregate stock data bars used to generate the data.
    dataframe : TYPE pandas.DataFrame
        DESCRIPTION. The raw data pulled from polygon.io
    merged_report : TYPE dictionary
        DESCRIPTION. A full NaN report for the data generation process
    cleaned_indicators : TYPE pandas.DataFrame
        DESCRIPTION. A dataframe of the best performing version of each custom indicator function. 
    selected_indicators : TYPE pandas.DataFrame
        DESCRIPTION. A dataframe of the 'top' indicators which have been selected for use in the predictive model.
    
    Returns
    -------
    None.

    '''
    # Create output list 
    base_outputs = ['_dataframe', '_report', '_cleaned_indicators', '_selected_indicators']
    tkr_specific_outputs = [tkr+temporal_granularity+name for name in base_outputs]
    
    # Assign outputs
    outputs = dataframe, merged_report, cleaned_indicators, selected_indicators
    saved_data_indicators = dict(zip(tkr_specific_outputs, outputs))
    
    # Create a deployment model folder with the name of 'tkr' combined with date of training start (year month and day)
    folder_name = f"{tkr}_saved_data_indicators_{temporal_granularity}"
    os.makedirs(folder_name, exist_ok=True) # If folder doesn't exist, create it
    
    # Save 'output_dict' as a pickled object
    with open(os.path.join(folder_name, 'saved_data_indicators.pkl'), 'wb') as f:
        pickle.dump(saved_data_indicators, f)
        
    print(f'Data and custom features saved for {tkr} with temporal granularity of {temporal_granularity}')
    
# # TEST ##
# tkr = 'NVDA'
# key =  #Polygon subscription key
# days=(365*2-1)
# temporal_granularity = 'one_minute'
# deployment_model=True
# pseudo_test_days=0
# epochs=3
# AutoKeras=False
# clean_generations=1
# assimilated_generations=1
# population_scalar = 4
# max_trials=10
# use_apex=True
# use_apex_genesis=False
# troubleshooting = False