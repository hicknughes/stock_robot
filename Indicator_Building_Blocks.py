#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
These functions are written for a dataframe of stock data that has been cleaned of NaN values
They output the same dataframe with the column(s) of calculated values added on
"""

import numpy as np
import pandas as pd
import temporal_backend as tb
from datetime import datetime, time, timedelta


############################################################
'''
********* Calculate VWAP ************
'''
############################################################

def calc_vwap(df, tradingdays):
   vwap = []
   for day in tradingdays:
       #day = tradingdays[1]
       data = df[df['day'] == day]
       typical_prices = (data['high'] + data['low'] + data['close']) / 3  # calculate typical prices
       cum_price_x_vol = np.cumsum(typical_prices * data['volume']) # running total of price times volume
       cum_volume = np.cumsum(data['volume']) # running total of volume
       vwap_values = cum_price_x_vol / cum_volume # vwap calculation for each row
       vwap.extend(list(vwap_values)) # append the last vwap value for the given day
       
   return vwap

############################################################
'''
********* Scale and Diff 'Staple' Stock Data Columns ************
'''
############################################################

def scale_volume_with_buffer(series, buffer_percent):
    min_val = 0
    max_val = 500000
    range_val = max_val - min_val
    buffer_size = buffer_percent * range_val
    scaled_series = (series - (min_val - buffer_size)) / (range_val + 2 * buffer_size)
    return scaled_series

def scale_transactions_with_buffer(series, buffer_percent):
    min_val = 0
    max_val = 10000
    range_val = max_val - min_val
    buffer_size = buffer_percent * range_val
    scaled_series = (series - (min_val - buffer_size)) / (range_val + 2 * buffer_size)
    return scaled_series

def diff_scale_source_data(stock_twoyear):
    df_diffed = stock_twoyear.copy()
    df_diffed['close'] = df_diffed['close'].diff()
    df_diffed['vwap'] = df_diffed['vwap'].diff()
    df_diffed['volume'] = scale_volume_with_buffer(df_diffed['volume'], .1)
    df_diffed['transactions'] = scale_transactions_with_buffer(df_diffed['transactions'], .1)
    df_diffed = df_diffed.dropna().reset_index(drop=True)
    return df_diffed




############################################################
'''
********* DEPLOYMENT: Scale and Diff 'Staple' Stock Data Columns ************
Keeping NaN's'
'''
############################################################

def deployment_diff_scale_source_data(stock_twoyear):
    df_diffed = stock_twoyear.copy()
    df_diffed['close'] = df_diffed['close'].diff()
    df_diffed['vwap'] = df_diffed['vwap'].diff()
    df_diffed['volume'] = scale_with_buffer(df_diffed['volume'], .1)
    df_diffed['transactions'] = scale_with_buffer(df_diffed['transactions'], .1)
    # df_diffed = df_diffed.dropna().reset_index(drop=True)
    return df_diffed
 
############################################################
'''
********* Scale Values With Buffer ************
'''
############################################################

def scale_with_buffer(series, buffer_percent):
    min_val = series.min()
    max_val = series.max()
    range_val = max_val - min_val
    buffer_size = buffer_percent * range_val
    scaled_series = (series - (min_val - buffer_size)) / (range_val + 2 * buffer_size)
    return scaled_series

############################################################
'''
********* Growth Calc for Overnight_Granularities < 1Day ************
'''
############################################################

def growth_calc(df, lag, growth_threshold, temporal_granularity):
    df_func = df.copy()
    end_of_day = tb.end_of_day[temporal_granularity] 
 
    # Calculate purchase prices 
    buy_prices = tb.calculate_purchase_price(temporal_granularity, df_func['open'].shift(-1), df_func['high'].shift(-1))
    
    sale_index_original = np.array(df_func.index + 1 + lag) # Indexing so that all values < len(df_func)    
    sale_index_original[-(1+lag):] = sale_index_original[-(1+lag):] - 1 - lag
    df_func['sale_index_original'] = sale_index_original
    
    end_of_day_time = datetime.strptime(end_of_day, '%H:%M').time()
    df_func['new_timemerge_dt'] = df_func['timemerge_dt'].dt.normalize() + pd.to_timedelta(end_of_day_time.strftime('%H:%M:%S'))
    index_of_days_end = df_func[df_func['new_timemerge_dt'] == df_func['timemerge_dt']].index
    replacements = df_func['day'].map(dict(zip(df_func['day'].unique(), index_of_days_end)))

    df_func['sale_timemerge_dt'] = df_func['timemerge_dt'].shift(-(lag+1))
    need_replacing = (df_func['sale_timemerge_dt'].dt.time > time(15, 59)).astype(int)
    dont_need_replacing = (df_func['sale_timemerge_dt'].dt.time < time(15, 59)).astype(int)

    sale_index_missing_LIDs = sale_index_original * dont_need_replacing
    sale_index_of_only_replacements = replacements * need_replacing
    sale_index = (sale_index_missing_LIDs + sale_index_of_only_replacements)
    
    low_at_sale = df_func['low'].iloc[sale_index].reset_index(drop=True) #Low at sale, included in all sale price calculations
    standard_sale_open_price = df_func['open'].iloc[sale_index].reset_index(drop=True) # Open price at all sales, used only for non-LIDs
    replacement_sale_close_price = df_func['close'].iloc[sale_index].reset_index(drop=True) # Close price at all sales, used only for LIDs
    sale_open_missing_LIDs = standard_sale_open_price * dont_need_replacing # Knocking out all LIDs rows to equal 0
    sale_close_replacements_only = replacement_sale_close_price * need_replacing # Knocking out all non-LIDs rows to equal 0
    open_close_on_sale_prices = sale_open_missing_LIDs + sale_close_replacements_only # Adding the two together so that LIDs and non-LIDs combine seamlessly

    sale_pricesv = tb.calculate_sale_price(temporal_granularity, open_close_on_sale_prices, low_at_sale)
    lag_growth = (sale_pricesv / buy_prices) - 1
    growth_ind = (lag_growth > growth_threshold).astype(int)
    
    df_func = df_func.drop(columns=['new_timemerge_dt', 'sale_index_original', 'sale_timemerge_dt'])

    df_func['lag_growth'] = lag_growth
    df_func['growth_ind'] = growth_ind
   
    return df_func

############################################################
'''
********* Growth Calc for 1Day Granularity ************
'''
############################################################

def oneday_growth_calc(df, lag, growth_threshold, temporal_granularity):
    df_func = df.copy()
    
    # Calculate purchase prices outside the loop
    buy_prices = tb.calculate_purchase_price(temporal_granularity, df_func['open'].shift(-1), df_func['high'].shift(-1))
    
    sale_index = np.array(df_func.index + 1 + lag) # Indexing so that all values < len(df_func)    
    sale_index[-(1+lag):] = sale_index[-(1+lag):] - 1 - lag
    open_at_sale = df_func['open'].iloc[sale_index].reset_index(drop=True) # Open price at all sales
    low_at_sale =  df_func['low'].iloc[sale_index].reset_index(drop=True) # Open price at all sales
    sale_prices = tb.calculate_sale_price(temporal_granularity, open_at_sale, low_at_sale)
    
    lag_growth = (sale_prices / buy_prices) - 1
    growth_ind = (lag_growth > growth_threshold).astype(int)
    
    df_func['lag_growth'] = lag_growth
    df_func['growth_ind'] = growth_ind
   
    return df_func

############################################################
'''
********* Single Day Growth Calc ************
'''
############################################################
def single_day_growth_calc(df, lag, growth_threshold, temporal_granularity):
    df_func = df.copy()  

    # Calculate purchase prices 
    buy_prices = tb.calculate_purchase_price(temporal_granularity, df_func['open'].shift(-1), df_func['high'].shift(-1))
    
    open_on_sale = df_func['open'].shift(-(1+lag))
    low_on_sale = df_func['low'].shift(-(1+lag))
    close_at_days_end = df_func['close'][df_func['timemerge_dt'].dt.time == datetime.strptime(tb.end_of_day[temporal_granularity], "%H:%M").time()].iloc[0]
    low_at_days_end = df_func['low'][df_func['timemerge_dt'].dt.time == datetime.strptime(tb.end_of_day[temporal_granularity], "%H:%M").time()].iloc[0]    
    open_on_sale = open_on_sale.fillna(close_at_days_end)
    low_on_sale = low_on_sale.fillna(low_at_days_end)
    
    sale_prices = tb.calculate_sale_price(temporal_granularity, open_on_sale, low_on_sale)
    
    lag_growth = (sale_prices / buy_prices) - 1
    growth_ind = (lag_growth > growth_threshold).astype(int)
    
    df_func['lag_growth'] = lag_growth
    df_func['growth_ind'] = growth_ind

    return df_func


############################################################
'''
********* Growth Calc on 'Close' ************
'''
############################################################

def growth_calc_on_close(df, lag, growth_threshold, temporal_granularity):
    df_func = df.copy()
    
    buy_prices = df_func['close']
    sale_prices = df_func['close'].shift(-lag) 
    
    lag_growth = (sale_prices / buy_prices) - 1
    growth_ind = (lag_growth > growth_threshold).astype(int)

    df_func['lag_growth'] = lag_growth
    df_func['growth_ind'] = growth_ind
   
    return df_func


############################################################
'''
********* Intra-Day Growth & Outcome Variable Calculation ************
'''
############################################################

def daily_growth_calc(df, lag, growth_threshold):
    grouped_df = df.groupby('day')
    result_dfs = []
    for day, data in grouped_df:
        subset_result = growth_calc(data.reset_index(drop=True), lag, growth_threshold)
        result_dfs.append(subset_result)
    final_df = pd.concat(result_dfs)
    return final_df.reset_index(drop=True)


############################################################
'''
********* Intra-Day Indicator Value Generation ************
'''
############################################################

def daily_indicator(df, ind_funshion, ind_params):
    
    # final_df = df.groupby('day').apply(ind_funshion(data.reset_index(drop=True), **ind_params))
    
    grouped_df = df.groupby('day')
    result_dfs = []
    for day, data in grouped_df:
        subset_result = ind_funshion(data.reset_index(drop=True), **ind_params)
        result_dfs.append(subset_result)
    final_df = pd.concat(result_dfs)
    return final_df.reset_index(drop=True)

############################################################
'''
********* ROLLING SUM ************
'''
############################################################

def rolling_sum(pd_series,lookback):
    rolling_sum = pd_series.rolling(lookback).sum()
    return rolling_sum
# Test
# masterbase['trans_sum'] = rolling_sum(masterbase['transactions'],3)

############################################################
'''
********* "Simple Moving Average" ************
''' 
############################################################


def sma(pd_series, lookback_period):
    sma = pd_series.rolling(window=lookback_period).mean()
    return sma

############################################################
'''
********* SCALED CROSSOVER ************
'''
############################################################

def scaled_crossover(indicator, rolling_average):
    assert len(indicator) == len(rolling_average)
    greater_mac = (indicator > rolling_average).astype(int)
    greater_ewa = (indicator < rolling_average).astype(int) * -1
    above_below = greater_mac + greater_ewa
    above_below_shift = above_below.shift()
    cross_ind = above_below + above_below_shift
    pos_cross_occurence = (cross_ind == 0).astype(int) * (above_below == 1).astype(int)
    abs_diff = abs(indicator-rolling_average)
    cross_magnitude = (abs_diff + abs_diff.shift())# / np.max(abs_diff + abs_diff.shift())
    scaled_cross_under_thresh = cross_magnitude * pos_cross_occurence
    #If the result is coming out with NaN's in the beginning, use next line
    scaled_cross_under_thresh[0] = 0
    return scaled_cross_under_thresh

############################################################
'''
********* SIMPLE CROSSOVER ************
'''
############################################################

def positive_crossover(indicator, rolling_average, half_life):
    assert len(indicator) == len(rolling_average)
    greater_mac = (indicator > rolling_average).astype(int)
    greater_ewa = (indicator < rolling_average).astype(int) * -1
    above_below = greater_mac + greater_ewa
    above_below_shift = above_below.shift()
    cross_ind = above_below + above_below_shift
    pos_cross_occurence = (cross_ind == 0).astype(int) * (above_below == 1).astype(int)
    positive_crossover = halflife_pd_series(pos_cross_occurence, half_life)
    return positive_crossover

############################################################
'''
********* SIMPLE CROSSOVER UNDER THRESHOLD ************
'''
############################################################

def positive_crossover_under_thresh(indicator, rolling_average, ll_threshold_percentile, half_life):
    assert len(indicator) == len(rolling_average)
    greater_mac = (indicator > rolling_average).astype(int)
    greater_ewa = (indicator < rolling_average).astype(int) * -1
    above_below = greater_mac + greater_ewa
    above_below_shift = above_below.shift()
    cross_ind = above_below + above_below_shift
    pos_cross_occurence = (cross_ind == 0).astype(int) * (above_below == 1).astype(int)
    rolling_window = indicator.rolling(window=35)  # Creating a rolling window of the last 35 values
    ll_threshold = rolling_window.quantile(quantile=ll_threshold_percentile) 
    # ll_threshold = indicator.quantile(q=ll_threshold_percentile)
    underthresh = (indicator.shift() < ll_threshold).astype(int) * (rolling_average.shift() < ll_threshold).astype(int)
    pos_cross_occurence_underthresh = underthresh * pos_cross_occurence
    positive_crossover = halflife_pd_series(pos_cross_occurence_underthresh, half_life)
    return positive_crossover


############################################################
'''
********* SCALED CROSSOVER UNDER A THRESHOLD ************
'''
############################################################

#Crossover under a threshold function
def scaled_crossover_under_threshold(ll_threshold_percentile, indicator, rolling_average):
    assert len(indicator) == len(rolling_average)
    rolling_window = indicator.rolling(window=35)  # Creating a rolling window of the last 35 values
    ll_threshold = rolling_window.quantile(quantile=ll_threshold_percentile)      
    below_thresh = (indicator.shift() < ll_threshold).astype(int) * (rolling_average.shift() < ll_threshold).astype(int)
    greater_mac = (indicator > rolling_average).astype(int)
    greater_ewa = (indicator < rolling_average).astype(int) * -1
    above_below = greater_mac + greater_ewa
    above_below_shift = above_below.shift()
    cross_ind = above_below + above_below_shift
    pos_cross_occurence = (cross_ind == 0).astype(int) * (above_below == 1).astype(int) * below_thresh #(cross occurred * from below to above * below the threshold)
    abs_diff = abs(indicator-rolling_average)
    cross_magnitude = (abs_diff + abs_diff.shift()) #/ np.max(abs_diff + abs_diff.shift())
    scaled_cross_under_thresh = cross_magnitude * pos_cross_occurence
    #If the result is coming out with NaN's in the beginning, use next line
    scaled_cross_under_thresh[0] = 0
    return scaled_cross_under_thresh


############################################################
'''
********* HALF_LIFE ************
'''
############################################################

def halflife_pd_series(scaled_indicator_series, half_life):
    assert isinstance(scaled_indicator_series, pd.Series), "Input should be a pandas Series"
    assert scaled_indicator_series.index.is_unique, "Index should not contain duplicate values"
    assert scaled_indicator_series.index.min() == 0, "Input Index should start at 0. Please reset source data's Index"
    assert scaled_indicator_series.index.max() == len(scaled_indicator_series) - 1, "Index has missing values, please reset source data's Index"    
    assert half_life >= 0, "half_life must be greater than or equal to 0"
    if half_life == 0:
        return scaled_indicator_series
    else: 
        non_zeros = np.array(np.nonzero(np.array(scaled_indicator_series))[0])
        output_array = scaled_indicator_series.copy()
        for i in non_zeros:
            base = output_array[i]
            count = 1
            if i + half_life < len(scaled_indicator_series):
                for j in range(i+1, i+half_life):
                    
                        if output_array[j] == 0:
                            decay_value = ((half_life - count) / half_life) * base
                            output_array[j] = decay_value
                            count += 1
                        else:
                            break
            else:
                len_diff = len(scaled_indicator_series) - i
                for j in range(i+1, i+len_diff):
                    if output_array[j] == 0:
                        decay_value = ((half_life - count) / half_life) * base
                        output_array[j] = decay_value
                        count += 1
                    else:
                          break
    return output_array


############################################################
''' 
********* FIND ALL MINIMUMS IN TIME SERIES ************
'''
############################################################
# GPT Alternative:
def mins(pd_series):
    minima_index = []
    minima_values = []

    # Use vectorized operations to find the minima
    minima_mask = (pd_series.shift(1) > pd_series) & (pd_series.shift(-1) > pd_series)
    minima_values = pd_series[minima_mask]
    minima_index = pd_series.index[minima_mask]

    # Construct the DataFrame
    df = pd.DataFrame({'ind': minima_index, 'vals': minima_values}).reset_index(drop=True)

    return df



############################################################
''' 
********* FIND SIMPLE SLOPE ************
'''
############################################################

def calculate_slope(point1, point2):
    slope = (point2['vals'] - point1['vals']) / (point2['ind'] - point1['ind'])
    return slope

############################################################
''' 
********* FIND SLOPE OF MINIMUMS ************
'''
############################################################

#Requires a minima_df of point values, and the source_data that minima were found within
def slope_of_mins(source_data, minima_df, lookback_value): #took 11 minutes to run
    slopes_of_mins = []
    for k in range(minima_df['ind'][1]):
        slopes_of_mins.append(0)
    for w in range(minima_df['ind'][1],len(source_data)):
        lower = w - lookback_value
        upper = w
        window_points = minima_df[(minima_df['ind'] >= lower) & (minima_df['ind'] <= upper)] #select all mins in window
        while len(window_points) < 2:
            lower -= 1
            window_points = minima_df[(minima_df['ind'] >= lower) & (minima_df['ind'] <= upper)]
        slopes = []
        for i in range(len(window_points)): # Calculate the slope between every combination of two points
            for j in range(i + 1, len(window_points)):
                slope_ij = calculate_slope(window_points.iloc[i], window_points.iloc[j])
                slopes.append(slope_ij)
        average_slope = sum(slopes) / len(slopes) # Calculate the average slope
        slopes_of_mins.append(average_slope)
    return slopes_of_mins

def slope_of_mins_gpt(source_data, minima_df, lookback_value):
    slopes_of_mins = [0] * (minima_df['ind'][1])
    minima_df['diff'] = minima_df['vals'] - minima_df['vals'].shift()
    for i in range(minima_df['ind'][1], len(source_data)):
        lower = i - lookback_value
        upper = i
        window_points = minima_df[(minima_df['ind'] >= lower) & (minima_df['ind'] <= upper)]

        while len(window_points) < 2:
            lower -= 1
            window_points = minima_df[(minima_df['ind'] >= lower) & (minima_df['ind'] <= upper)]
        average_slope = np.mean(window_points['diff'])
        slopes_of_mins.append(average_slope)

    return slopes_of_mins

# start = time.time()
# test_minslopes_gpt = slope_of_mins_gpt(df, minima_df, 25)
# end = time.time()
# duration = (end-start) / 60
#General syntax:
# slopes = slope_of_mins(df_div, minima_df, lookback_value)
# Combined Syntax:
# slope_of_mins(df_div, mins(pd_series), lookback_value)

############################################################
''' 
********* DIVERGENCE ************
'''
############################################################

def divergence_on_lows(source_data, close_prices_series, indicator_series, lookback_value, half_life, decay=True):
    price_mins_slope = np.array(slope_of_mins(source_data, mins(close_prices_series),lookback_value))
    macd_mins_slope = np.array(slope_of_mins(source_data, mins(indicator_series),lookback_value))
    div_ind = (np.array(price_mins_slope) < 0) * (np.array(macd_mins_slope) > 0)
    divergence = abs(price_mins_slope * macd_mins_slope * div_ind)
    if decay == True:
        divergence = halflife_pd_series(pd.Series(divergence), half_life)
    return divergence

# df_div['divergence'] = divergence_on_lows(df_div, df_div['Close'], df_div['macd'], 10, 4, decay=True)

# threshold = 0
# indicator = df_div['macd']

############################################################
'''
********* SINGLE INDICATOR FALLING BELOW A DYNAMIC THRESHOLD ************
'''
############################################################
def mag_below_dynamic(threshold, indicator, half_life, decay=True):
    positive_difference = threshold - indicator
    positive_difference = positive_difference.apply(lambda x: x if x > 0 else 0)
    if max(positive_difference) == 0:
        positive_difference = [0] * len(positive_difference)
    else:
        positive_difference = positive_difference #/  max(positive_difference)
    if decay == True:
        positive_difference = halflife_pd_series(pd.Series(positive_difference), half_life)
    return positive_difference

# test_magunder = mag_below_dynamic(lower_band,func_rsi,3)


############################################################
'''
********* SINGLE INDICATOR FALLING ABOVE/BELOW A STATIC THRESHOLD ************
'''
############################################################


def mag_under_threshold(indicator_series, threshold_percentile, half_life, decay=True, percentile=True):
    if percentile == True: 
        rolling_window = indicator_series.rolling(window=35)  # Creating a rolling window of the last 35 values
        threshold = rolling_window.quantile(quantile=threshold_percentile)
        # threshold = indicator_series.quantile(q=threshold_percentile)
    else:
        threshold = threshold_percentile
    below = ((indicator_series - threshold) < 0)
    below = below.fillna(below.median())
    under_threshold = abs(indicator_series - threshold) #/ min(indicator_series - threshold)) * below) ** 2
    if decay == True:
        under_threshold = halflife_pd_series(pd.Series(under_threshold), half_life)
    return under_threshold

# # df_div['macd_subzero'] = mag_under_threshold(df_div['macd'], 0.25, 4)

# def mag_above_threshold(indicator_series, threshold_percentile, half_life, decay=True):
#     if percentile == True: 
#         threshold = indicator_series.quantile(q=threshold_percentile)
#     else:
#         threshold = threshold_percentile    
#     above = ((indicator_series - threshold) > 0)
#     above = above.fillna(above.median())
#     above_threshold = abs(((indicator_series - threshold) / max(indicator_series - threshold)) * above) ** 2
#     if decay == True:
#         above_threshold = halflife_pd_series(pd.Series(above_threshold), half_life)
#     return above_threshold

# df_div['macd_supra25'] = mag_above_threshold(df_div['macd'], .25, 3)

############################################################
'''
*********BELOW (DYNAMIC) BAND & BELOW (STATIC) THRESHOLD************
'''
############################################################


# def below_band_below_threshold(close, band, indicator, threshold_percentile, half_life, decay=True):
#     assert sum(band.isna()) == 0
#     assert sum(close.isna()) == 0
#     close_below_band = ((close - band) < 0) * abs((close - band) / abs(min(close-band)))
#     ind_below_threshold = mag_under_threshold(indicator,threshold, 0, False)
#     overlap_scaled = (close_below_band * ind_below_threshold) / max(close_below_band * ind_below_threshold)
#     if decay == True:
#         overlap_scaled = halflife_pd_series(pd.Series(overlap_scaled), half_life)
#     return overlap_scaled

# df_div['macd_oversold_price_llKeltner'] = below_band_below_threshold(df_div['Close'],df_bol['Lower Band'], df_div['macd'],0,3,True)


############################################################
'''
********* "Stochastic Swing Strategy" ************
'''
############################################################

# def sma(pd_series, lookback_period):
#     sma = pd_series.rolling(window=lookback_period).mean()
#     return sma


# lookback_period = 5
# close = df_div['Close']
# sma_series = sma(close, lookback_period)
# #Fixing NaN problem
# sma_series.fillna(sma_series.mean(), inplace=True)
# price_under_sma = ((close - sma_series) < 0) * abs((close - sma_series) / abs(min(close-sma_series)))




############################################################
'''
********* Downward Trend Warning Indicator ************
'''
############################################################
def downtrend_warning_slopenearfar_delta(pd_series, lookback_near, lookback_far):
    slope_near = pd_series.diff(periods=lookback_near) / lookback_near
    slope_far = pd_series.diff(periods=lookback_far) / lookback_far
    downtrend_warning_indicator = ((slope_near.fillna(0) - slope_far.fillna(0)) > 0).astype(int)
    return downtrend_warning_indicator


############################################################
'''
********* Bollinger Band Calculation ************
'''
############################################################

def bollinger_bands(df, window=20, n_std=2):
    TP = (df['high'] + df['close'] + df['low']) / 3
    # Calculate the rolling mean using an exponential moving average
    ema = TP.ewm(span=window, adjust=False).mean()
    # Calculate the rolling standard deviation using an exponential moving average
    ema_std = TP.ewm(span=window, adjust=False).std()
    # Calculate the upper and lower Bollinger Bands
    upper = ema + (n_std * ema_std)
    lower = ema - (n_std * ema_std)
    return upper, lower


############################################################
'''
********* Function Efficiency Test ************
'''
############################################################

# from line_profiler import LineProfiler

# profiler = LineProfiler()
# profiler.add_function(your_function)
# profiler.run('your_function()')
# profiler.print_stats()