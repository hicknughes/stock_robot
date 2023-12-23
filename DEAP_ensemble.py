#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEAP_ensemble() applies the DEAP_clean() optimization function to a list of functions and their corresponding input parameter ranges.
The first round, dubbed 'cleaning', allows for 'lag' and 'growth_threshold' values to be optimized.
The scond round, dubbed 'assimilated', uses a weighted average of 'lag' and 'growth_threshold' values from top performing indicators, fixing those values for all future optimization.
"""
import DEAP_precision as DP
import pandas as pd
import numpy as np
import warnings
#The next 5 lines suppress the Precision=0 warning
from sklearn.exceptions import UndefinedMetricWarning
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def DEAP_ensemble(tkr, stock_df, func_list, dict_list, temporal_granularity, population_scalar=23, clean_generations=1, assimilated_generations=3, min_scoring_quantile=0.7, use_apex=True, use_apex_genesis=False):
    '''
    Parameters
    ----------
    tkr : TYPE string
        DESCRIPTION. The ticker symbol of the stock of interest for the model
    stock_df : TYPE pandas.DataFrame
        DESCRIPTION. All stock data to be utilized for feature engineering; adhering to the format of the output from raw_data() in raw_data.py
    func_list : TYPE A list of objects of type function.
        DESCRIPTION. Each function in the list produces a custom indicator, based on flexible input parameters
    dict_list : TYPE A list of objects of type dictionary.
        DESCRIPTION. Each dictionary in the list provides ranges of values for the flexible input parameters of the functions in the list above
    temporal_granularity : TYPE string
        DESCRIPTION. Duration of aggregate stock data bars used to generate the data.
    population_scalar : TYPE, integer
        DESCRIPTION. The default is 23. For each feature parameter listed in the input parameter dictionary, this many individuals will be made per generation.
    clean_generations : TYPE, integer
        DESCRIPTION. The number of generations of features to test and evolve with flexible 'lag' and 'growth_threshold' values, after generation 0.
    assimilated_generations : TYPE, integer
        DESCRIPTION. The number of generations of features to test and evolve with fixed 'lag' and 'growth_threshold' values, after generation 0.
    min_scoring_quantile : TYPE, optional
        DESCRIPTION. The default is 0.7. The quantile threshold above which indicators must perform, based on the primary objective scoring metric, in order 
        to contribute to the fixed 'lag' and 'growth_threshold' values.
    use_apex : TYPE, boolean
        DESCRIPTION. When True, and if a previous record of high performing features is available, their top performing 'Individuals' will be injected into generation 0.
    use_apex_genesis : TYPE boolean
        DESCRIPTION. When True, apex 'Individuals' with the same temporal_granularity (irregardless of the stock) will be used. Ideal when training a model on a new stock.

    Returns
    -------
    evolved_df : TYPE pandas.DataFrame
        DESCRIPTION. A dataframe of the best performing version of each custom indicator function. 
        Columns include the indicator name, lag value, growth threshold value, primary objective score of the optimal input parameters, and a dictionary of optimal input parameters.

    '''
    assert isinstance(stock_df, pd.DataFrame), "stock_df should be a Pandas DataFrame."
    assert stock_df.index.equals(pd.RangeIndex(len(stock_df))), "Please reset input stock_df's index."
    assert isinstance(func_list, list), "func_list should be a list"
    assert all(callable(item) for item in func_list), "All items in func_list should be functions"
    assert isinstance(dict_list, list), "dict_list should be a list"
    assert all(isinstance(item, dict) for item in dict_list), "All items in dict_list should be dictionaries"
    
    #Round one of DEAP_clean
    ind_lineages = []
    for i in range(len(func_list)):
        try:
            ind_lineages.append(DP.DEAP_clean(tkr, stock_df, func_list[i], dict_list[i], temporal_granularity, population_scalar=population_scalar, 
                                              max_generations=clean_generations,use_apex=use_apex, use_apex_genesis=use_apex_genesis))
        except KeyboardInterrupt:
            return
        except Exception as e:
            # Handle the error from the 'Initialize' function
            print(f"Error occurred at iteration i = {i}. Indicator: {func_list[i]}.")
            print(f"Error message: {str(e)}")
            continue
    round1_indicators = pd.concat(ind_lineages)
    
    # Establish 'voting' indicators such that top performing 30% decide the assimilated 'lag' and 'growth_threshold' values
    voting_indicators = round1_indicators[round1_indicators['scoring_metric'] >= round1_indicators['scoring_metric'].quantile(min_scoring_quantile)]
    
    # Define lag and growth threshold through a weighted average of all 'voting_indicators'
    average_lag, average_growth_threshold = np.average(voting_indicators[['lag', 'growth_threshold']], weights=voting_indicators['scoring_metric'], axis=0)
    average_lag = round(average_lag)
    
    # Re-optimize custom features with DEAP_clean, this time with fixed 'lag' and 'growth_threshold' values
    assimilated_lineages = []
    for i in range(len(func_list)):
        try:
            assimilated_lineages.append(DP.DEAP_clean(tkr, stock_df, func_list[i], dict_list[i], temporal_granularity, 
                                                      population_scalar=population_scalar, max_generations=assimilated_generations, 
                                                      fix_lag_growth=True, assigned_lag=average_lag, assigned_growth_thresh=average_growth_threshold,
                                                      use_apex=use_apex, use_apex_genesis=use_apex_genesis))
        except KeyboardInterrupt:
            return
        except Exception as e:
              # Handle the error from the 'Initialize' function
              print(f"Error occurred at iteration i = {i}. Indicator: {func_list[i]}.")
              print(f"Error message: {str(e)}")
              continue
    
    # Make DF with assimilated, optimized features
    evolved_df = pd.concat(assimilated_lineages).reset_index(drop=True)
    
    return evolved_df
