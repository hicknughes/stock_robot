#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script houses a toolkit of functions that interact with Polygon.io's APIs
to execute data pulls and check realtime pricing. Some but not all are used in IB_live_trading.py
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime as dt
import pytz
from datetime import timedelta
from polygon import *
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import time
import Indicator_Building_Blocks as ind
import ast
import f_o_32_39 as fo

# key = "GpjjoRW_XKUaCLvWWurjuMUwF34oHvpD" #paid subscription key
# tkr = 'NVDA'

def market_price(key, tkr):
    client = RESTClient(key) 
    quote = client.get_last_quote(
        tkr,
    )
    quote_dict = vars(quote)
    current_bid = quote_dict['bid_price']
    trade = client.get_last_trade(
        tkr,
    )
    trade_dict = vars(trade)
    last_trade_price = trade_dict['price']
    market_price_best_guess = (current_bid + last_trade_price) / 2
    return market_price_best_guess

def profit_loss(key, tkr, purchase_price):
    current_price = market_price(key, tkr)
    profit_loss = current_price / purchase_price
    return profit_loss

def datapull_until_now(tkr, key, yesterday=False):
    
    if yesterday == False:    
    #Establish start_time to be the beginning of today
        current_date = dt.utcnow().date()
        beginning_of_day = dt.combine(current_date, dt.min.time())
        start_date = int(beginning_of_day.timestamp() * 1000) 
        
        # End time is current time
        current_time = dt.utcnow()
        one_minute_ago = current_time - timedelta(minutes=1)
        end_date = int(one_minute_ago.timestamp() * 1000)
    
    if yesterday == True:    
        current_date = dt.utcnow()
        # For grabbing the previous trading day
        if current_date.weekday() == 0:  # Monday
            start_date = current_date - timedelta(days=3)
        else:
            start_date = current_date - timedelta(days=1)
        day_of_trading = start_date
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(hours=16.5)
        start_date = int(start_date.timestamp() * 1000)
        end_date = int(end_date.timestamp() * 1000)
        
    client = RESTClient(key)
    resp = client.get_aggs(tkr, multiplier=1, timespan='minute', from_ = start_date, to = end_date)

    df= pd.DataFrame(resp) #making a dataframe
    df['date_EST'] = pd.to_datetime(df['timestamp'], unit = "ms").dt.tz_localize('UTC').dt.tz_convert('US/Eastern')#adding eastern timezone
    df['timemerge'] = df['date_EST'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['day'] = df['date_EST'].dt.strftime('%Y-%m-%d') #creating day column
    
    # Creating the time 'base'
    day_of = dt.fromtimestamp(start_date / 1000)
    # Define the trading day as 'start_date' without the time component
    day_of_trading = day_of.replace(hour=0, minute=0, second=0, microsecond=0)
    minute_intervals = pd.DataFrame(columns=['timemerge'],# Create a list of minute interval datetime objects
                                    index=pd.date_range(day_of_trading, (day_of_trading + timedelta(days=1)),
                                                        freq='1T'))
    minute_intervals = minute_intervals.between_time('09:30', '16:00')  # Adjusted to 9:30 AM to 4:00 PM
    minute_intervals = minute_intervals.index.strftime('%Y-%m-%d %H:%M:%S').tolist()

    base = pd.DataFrame({'timemerge': minute_intervals})
    base['timemerge_dt'] = pd.to_datetime(base['timemerge'])
    
    masterbase = base.merge(df, on = ['timemerge'], how = 'left')
      
    current_time = dt.now().astimezone(pytz.timezone('America/New_York'))# Get the current time
    current_hour_minute = current_time.replace(second=0, microsecond=0)
    one_minute_ago_ET = current_hour_minute - timedelta(minutes=1)
    # Select the row where 'timemerge_dt' has the same hour and minute as the current time
    masterbase = masterbase[masterbase['timemerge_dt'].dt.strftime('%H:%M') <= one_minute_ago_ET.strftime('%H:%M')]
    
    masterbase['growth_rate'] = masterbase.close.pct_change() #growth rates
    
    column_cleanup1 = ['close', 'open', 'high', 'low', 'growth_rate', 'volume', 'vwap', 
                      'transactions', 'day', 
                      'timemerge', 'timemerge_dt'] 
    
    column_cleanup_iter = ['close', 'open', 'high', 'low', 'growth_rate', 'volume', 'vwap', 
                      'transactions']  #column names for after filling NA
    
    masterbase = masterbase[column_cleanup1] 
    
    impute_it = IterativeImputer(max_iter=20,random_state=1234)
    impute_df = pd.DataFrame(impute_it.fit_transform(masterbase.iloc[:,:-3]), columns = column_cleanup_iter) #index to take out time columns
    
    #need to concat impute_df with the time columns of masterbase 
    masterbase = pd.concat([impute_df, masterbase.iloc[:,-3:]],axis = 1)
    
    # Converting VWAP, Volume and Transactions to integers and diff'ed values
    masterbase['volume'] = masterbase['volume'].astype('int')
    masterbase['transactions'] = masterbase['transactions'].astype('int')
    
    vwap = ind.calc_vwap(masterbase, str(current_date))
    masterbase['vwap'].fillna(pd.Series(vwap, dtype='float64'), inplace=True)    
    
    masterbase['day'] = masterbase['day'].ffill().bfill()
    
    return masterbase.reset_index(drop=True)

# cleaned_indicators_dataframe = clnd.copy()


def current_minute_X(data, cleaned_indicators_dataframe):
    running_df = data.copy().reset_index(drop=True)
    
    for i in range(len(cleaned_indicators_dataframe)):
        function = getattr(fo, cleaned_indicators_dataframe['indicator'][i])
        cleaned_params = ast.literal_eval(str(cleaned_indicators_dataframe['cleaned_params'][i]))
        next_df = function(running_df, **cleaned_params)
        running_df = next_df.copy()
    
    scaled = ind.deployment_diff_scale_source_data(running_df) 
    # print(f'length of scaled_df: {len(scaled)}')
    current_time = dt.now().astimezone(pytz.timezone('America/New_York'))# Get the current time
    current_hour_minute = current_time.replace(second=0, microsecond=0)
    one_minute_ago = current_hour_minute - timedelta(minutes=1)
    # Select the row where 'timemerge_dt' has the same hour and minute as the current time
    predict_on = scaled[scaled['timemerge_dt'].dt.strftime('%H:%M') == one_minute_ago.strftime('%H:%M')]
    # print(f'time from current time calc:  {current_hour_minute}')
    # print(f"scaled data timerge column: {scaled['timemerge_dt'].dt.strftime('%H:%M')}")
    current_minute_X = predict_on.drop(columns = ['open', 'high', 'low', 'day', 'timemerge', 'timemerge_dt'])
    
    return current_minute_X, scaled

# cleaned_indicators_dataframe = indicator_params.copy()
# test_x, test_x_scaled = current_minute_X(test, cleaned_indicators_dataframe)

def seconds_until_next_whole_minute():

    current_time = dt.now().astimezone(pytz.timezone('America/New_York'))# Get the current time
    next_minute = (current_time + timedelta(minutes=1)).replace(second=0, microsecond=0)
    time_difference = next_minute - current_time
    seconds_until_next_minute = time_difference.total_seconds()
    return int(seconds_until_next_minute)

# seconds = seconds_until_next_whole_minute()



def gen_hold_periods(buy_log, tkr, key):
    buy_log_copy = buy_log.copy()
    buy_log_copy['buy_submitted_utc'] = buy_log_copy['buy_submitted_time'].apply(lambda x: pd.Timestamp(x).tz_convert('UTC'))
    buy_log_copy['sold_utc'] = buy_log_copy['sold_time'].apply(lambda x: pd.Timestamp(x).tz_convert('UTC'))
    
    holds = []
    
    for i in range(len(buy_log_copy)):
        try:
            client = RESTClient(key)
            buy = int(buy_log_copy['buy_submitted_utc'][i].timestamp() * 1000) 
            sell = int(buy_log_copy['sold_utc'][i].timestamp() * 1000) 
            resp = client.get_aggs(tkr, multiplier=1, timespan='minute', from_=buy, to=sell)
            df = pd.DataFrame(resp).drop(columns=['otc', 'volume', 'vwap', 'transactions'])
            df['date_EST'] = pd.to_datetime(df['timestamp'], unit="ms").dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
            df['growth_rate'] = df.close.pct_change()
            df['return'] = df['growth_rate'] + 1
            df['profit_loss'] = df['return'].cumprod()
            df['sold_at'] = None
            df.loc[df.index[-1], 'sold_at'] = buy_log_copy.loc[i, 'sold_price']
            df = df[['growth_rate', 'profit_loss', 'sold_at', 'open', 'high', 'low', 'close', 'date_EST', 'return']]
            holds.append(df)
        except ValueError:
            # Append NaN to the 'holds' list if a ValueError occurs
            holds.append(np.nan)
        
    return holds

def check_4_immediate_buy(tkr, key, pred_log, indicator_params, nn_model, rounding_threshold, minute_wise_scaled_data, minute_wise_data, loop_start_time_string, nan_report):
    data = datapull_until_now(tkr, key=key, yesterday=False)

    ## Model Prediction on this minute's X-Vector
    X_vector, scaled_data = current_minute_X(data, indicator_params)
    # print(X_vector)
    if X_vector.isna().any().any():
        # Record the current time and DataFrame in a list
        nan_report.append([tkr, loop_start_time_string, X_vector])
        buy_or_nah = 0
        pred = None
    else:
        pred = nn_model.predict(X_vector)
        buy_or_nah = (pred > rounding_threshold)[0][0]
    pred_made_dt = dt.now().astimezone(pytz.timezone('America/New_York'))
    pred_made_string = pred_made_dt.strftime('%H:%M:%S.%fZ')
    
    latest_pred = {'tkr': [tkr], 'pred': [pred], 'time_of_pred': [pred_made_string]}
    # pred_log = pred_log.append(latest_pred, ignore_index=True)
    pred_log = pd.concat([pred_log, pd.DataFrame(latest_pred)], ignore_index = True)     
    minute_wise_scaled_data[loop_start_time_string] = scaled_data
    minute_wise_data[loop_start_time_string] = data 
    
    immediate_buy_again = buy_or_nah
    
    return immediate_buy_again, pred_log, minute_wise_scaled_data, minute_wise_data







