#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script houses all the temporal_granularity-specific information used throughout the pipeline to
differentiate the way data and workflows are handled.
"""
# A list of all temporal params currently supported. Used to verify temporal_granularity validity in raw_data()
valid_temportal_values = {'one_minute', 'three_minute', 'five_minute', 'fifteen_minute', 'one_hour', 'one_day'}

# The granularities for which indicator values are calculated day-by-day, resulting in a morning warm-up period
# Used in raw_data() and optimization_pipeline's cleaned_to_X()
daily_granularities = {}#'one_minute', 'three_minute', 'five_minute'

# The granularities for which indicator values are calculated with the entire datafame all at once, resulting in a warm-up period at the beginning of the dataframe
# Used in raw_data() and optimization_pipeline's cleaned_to_X()
overnight_granularities = {'fifteen_minute', 'one_hour', 'one_day', 'one_minute', 'three_minute', 'five_minute'}

## Temporal Params used in raw_data() to correctly pull and assimilate data from polygon.io, as well as create the NaN report
one_minute_temporal_params = {'polygon_timespan': 'minute',
                              'consecutive_nan_limit': 45,
                              'cons_zero_growth_limit': 15,
                              'polygon_multiplier': 1,
                              'base_index_freq': "1min",
                              'daterange_freq': "W"} # "W" = Weekly
three_minute_temporal_params = {'polygon_timespan': 'minute',
                              'consecutive_nan_limit': 15,
                              'cons_zero_growth_limit': 5,
                              'polygon_multiplier': 3,
                              'base_index_freq': "3min",
                              'daterange_freq': "W"} # "W" = Weekly; "SM" = Twice monthly (15th and end of month)
five_minute_temporal_params = {'polygon_timespan': 'minute',
                              'consecutive_nan_limit': 9,
                              'cons_zero_growth_limit': 3,
                              'polygon_multiplier': 5,
                              'base_index_freq': "5min",
                              'daterange_freq': "W"} # "W" = Weekly; "SM" = Twice monthly (15th and end of month)
fifteen_minute_temporal_params = {'polygon_timespan': 'minute',
                              'consecutive_nan_limit': 3,
                              'cons_zero_growth_limit': 2,
                              'polygon_multiplier': 15,
                              'base_index_freq': "15min",
                              'daterange_freq': "W"} # "W" = Weekly; "SM" = Twice monthly (15th and end of month)
one_hour_temporal_params = {'polygon_timespan': 'hour',
                              'polygon_multiplier': 1, 
                              'base_index_freq': "1h",
                              'daterange_freq': "M"} # "A" = Annual; Q = Quarterly
one_day_temporal_params = {'polygon_timespan': 'day',
                              'polygon_multiplier': 1, 
                              'base_index_freq': "1D",
                              'daterange_freq': "A"} # "A" = Annual; Q = Quarterly
# Merging all temporal params into a single dictionary to be accessed in raw_data()
temporal_params = {'one_minute': one_minute_temporal_params,
                   'three_minute': three_minute_temporal_params,
                   'five_minute': five_minute_temporal_params,
                   'fifteen_minute': fifteen_minute_temporal_params,
                   'one_hour': one_hour_temporal_params,
                   'one_day': one_day_temporal_params}


# Differentiating datapull time windows, used in raw_data()
datapull_8am_to_4pm_granularities = {'one_minute'}
datapull_7am_to_4pm_granularities = {'three_minute', 'five_minute'}
datapull_4am_to_8pm_granularities = {'fifteen_minute', 'one_hour'}

# Selecting granularities from which to eliminate days with many consecutive NaN rows, based on the 'consecutive_nan_limit' value in the granularity's temporal_params dictionary
# Used in raw_data()
granularities_to_eliminate_cons_nans_and_0growth = {'one_minute', 'three_minute', 'five_minute', 'fifteen_minute'}

# Used in Inidcator_Building_Blocks' growth_calc() and hold_strat for price estimation
def calculate_purchase_price(temporal_granularity, open_price, high_price):
    # A purchase price estimation based on open and high prices during the period in which a buy is executed.
    calculations = {
        'one_minute': (open_price + high_price * 2) / 3,
        'three_minute': (open_price * 2 + high_price) / 3,
        'five_minute': (open_price * 3 + high_price) / 4,
        'fifteen_minute': (open_price * 4 + high_price) / 5,
        'one_hour': (open_price * 10 + high_price) / 11,
        'one_day': open_price,
    }
    return calculations.get(temporal_granularity, None) # Apply the corresponding calculation based on temporal_granularity

# Used in Inidcator_Building_Blocks' growth_calc() and hold_strat for price estimation
def calculate_sale_price(temporal_granularity, open_price, low_price):
    # A sale price estimation based on open and low prices during the period in which a sale is executed.
    calculations = {
        'one_minute': (open_price + low_price * 2) / 3,
        'three_minute': (open_price * 2 + low_price) / 3,
        'five_minute': (open_price * 3 + low_price) / 4,
        'fifteen_minute': (open_price * 4 + low_price) / 5,
        'one_hour': (open_price * 10 + low_price) / 11,
        'one_day': open_price,
    }
    return calculations.get(temporal_granularity, None) # Apply the corresponding calculation based on temporal_granularity

# Used in growth_calc() and single_day_growth_calc(). It is the cutoff for LIDs' sale times
end_of_day = {
    'one_minute': '15:59',
    'three_minute': '15:57',
    'five_minute': '15:55',
    'fifteen_minute': '15:45',
    'one_hour': '15:00',
    'one_day': None,
}


## Used in Optimization Pipeline to add the correct lag and growth_threshold value ranges in preparation for DEAP optimization at the given temporal_granularity
lag_growth_thresh_tiers = {'one_minute': {'lag': [32,39], 'growth_threshold': [0.002,0.003]},
                           'three_minute': {'lag': [5,30], 'growth_threshold': [0.0025,0.0035]},
                           'five_minute': {'lag': [5,18], 'growth_threshold': [0.003,0.004]},
                           'fifteen_minute': {'lag': [2,4], 'growth_threshold': [0.0025,0.004]},
                           'one_hour': {'lag': [1,2], 'growth_threshold': [0.0035,0.006]},
                           'one_day': {'lag': [1,2], 'growth_threshold': [0.012,0.03]}}

# Used in hold_strat's HTNB logic to properly group consecutive buy orders
hold_period_expansion = {
    'one_minute': 1,
    'three_minute': 3,
    'five_minute': 5,
    'fifteen_minute': 15,
    'one_hour': 60,
    'one_day': 24*60,
}