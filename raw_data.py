
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For a given stock, the raw_data function creates a pandas dataframe of stock data between the dates, and at the temporal granularity supplied. 
For gaps in data from polygon.io, linear interpolation on OHLC columns and scikit-learn's IterativeImputer is used on all others excluding VWAP.
VWAP is calculated from the resulting matrix^ using the custom function below.
Each temporal_granularity has a corresponding tolerance for missing data and time range it pulls data from, which is selected to allow for 
optimized custom features to warm up in time to be trading no later than 11 am, though 9:30 am is the objective
"""
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
from polygon import RESTClient
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import time
class NoResultsError(Exception):
    pass

def calc_vwap(df, tradingdays):
   vwap = []
   for day in tradingdays:
       data = df[df['day'] == day]
       typical_prices = (data['high'] + data['low'] + data['close']) / 3  # calculate typical prices
       cum_price_x_vol = np.cumsum(typical_prices * data['volume']) # running total of price times volume
       cum_volume = np.cumsum(data['volume']) # running total of volume
       vwap_values = cum_price_x_vol / cum_volume # vwap calculation for each row
       vwap.extend(list(vwap_values)) # append the last vwap value for the given day
   return vwap


def raw_data(polygon_api_key, tkr, start_date, end_date, temporal_granularity, paid_polygon_account=False):
    '''
    Parameters
    ----------
    polygon_api_key : TYPE string
        DESCRIPTION. The API key from polygon.io associated with the user's account.
    tkr : TYPE string
        DESCRIPTION. The desired stock's ticker symbol.
    start_date : TYPE string
        DESCRIPTION. Historical data pull start date.
    end_date : TYPE string
        DESCRIPTION. Historical data pull end date.
    temporal_granularity : TYPE string
        DESCRIPTION. Duration of aggregate stock data bars used to generate the data.
        'one_minute', 'three_minute', 'five_minute', 'fifteen_minute', 'one_hour', or 'one_day' is allowed/supported
    paid_polygon_account : TYPE, optional
        DESCRIPTION. The default is False. If the user has a paid subscription with polygon.io which 
        has no API limitations, change this input to True

    Returns
    -------
    output_dataframe : TYPE pandas DataFrame
        DESCRIPTION. The resulting dataframe has columns for OHLC prices, percentage growth rate, 
        volume, transactions, VWAP, date and timestamps represented as strings and datetime objects.
    report_stats : TYPE
        DESCRIPTION. An NaN report on any missing or problematic data recieved from polygon.io

    '''
    assert dt.strptime(start_date, "%Y-%m-%d") < dt.strptime(end_date, "%Y-%m-%d"), "Start date must come before end date"
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
    temporal_params = {'one_minute': one_minute_temporal_params,
                       'three_minute': three_minute_temporal_params,
                       'five_minute': five_minute_temporal_params,
                       'fifteen_minute': fifteen_minute_temporal_params,
                       'one_hour': one_hour_temporal_params,
                       'one_day': one_day_temporal_params}
    valid_temportal_values = {'one_minute', 'three_minute', 'five_minute', 'fifteen_minute', 'one_hour', 'one_day'}
    if temporal_granularity not in valid_temportal_values:
        raise ValueError(f"Invalid temporal_granularity. Please choose one of {valid_temportal_values}")
    
    # Connecting to Poligon.io API
    client = RESTClient(polygon_api_key) 

    #Creating date intervals creates date ranges to pull data in chunks that don't violate API limits/data limit restrictions
    daterange = pd.date_range(start_date, end_date, freq=temporal_params[temporal_granularity]['daterange_freq']) #bi monthly date list between start/end dates
    date_intervals = [start_date]
    for i in range(len(daterange)):
        date_intervals.append(str(daterange[i].date()))
    date_intervals.append(end_date)
    date_intervals = [*set(date_intervals)] #remove duplicates
    date_intervals.sort(key = lambda date: dt.strptime(date, '%Y-%m-%d')) #Adhering to Polygon date format
    
    trade_days = []
    masterbase = pd.DataFrame()
    
    ## Pulling data for each interval within date_intervals
    for j in range(len(date_intervals)-1):  #keeping within API limits by iterating through date intervals 
        date_start = date_intervals[j]
        if j+1 == range(len(date_intervals))[-1]:
            date_end = date_intervals[j+1]
        if j+1 != range(len(date_intervals))[-1]:
            date_end = dt.strptime(date_intervals[j+1],'%Y-%m-%d') #converting to datetime
            date_end = str((date_end - timedelta(days = 1)).date()) #subtracting a day and making string to pass into polygon.io request
        
        # Data query with if/else to control for date ranges that return empty frames
        if date_start != date_end:
            resp = client.list_aggs(tkr, from_=date_start, to=date_end,
                                   timespan = temporal_params[temporal_granularity]['polygon_timespan'], 
                                   multiplier = temporal_params[temporal_granularity]['polygon_multiplier'],
                                   limit=50000) 
        else:
            continue
        
        # Adding time columns to Polygon source data
        df= pd.DataFrame(resp) #making a dataframe
        if len(df) > 4999:
            print("Polygon.io data point limit of 5,000 reached. Please increase 'freq' of date intervals in variable 'daterange'.")
            # return
        df['date_EST'] = pd.to_datetime(df['timestamp'], unit ="ms").dt.tz_localize('UTC').dt.tz_convert('US/Eastern')#adding eastern timezone
        df['timemerge'] = df['date_EST'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['day'] = df['date_EST'].dt.strftime('%Y-%m-%d') #creating day column
        
        # Create list of unique trading days from polygon data 
        trading_days = list(df['day'].unique())
        trade_days.extend(trading_days)
        
        # Building a base of timestamps on which to drop the polygon data
        base = pd.DataFrame()
        for i in range(len(trading_days)):
            today = trading_days[i]
            today_dt = dt.strptime(today,'%Y-%m-%d')
            tomorrow_dt = (today_dt + timedelta(days=1)).date()
            tomorrow = str(tomorrow_dt)
            # Conditionally set the time range for 'between_time' to align with temporal_granularity
            if temporal_granularity == 'one_day':
                date_strings_with_time = [day + ' 00:00:00' for day in trading_days]
                base = pd.DataFrame({'Timestamps': date_strings_with_time})
            else:
                if temporal_granularity in {'one_minute'}:
                    timepull_range = ('08:00', '16:00')
                elif temporal_granularity in {'three_minute', 'five_minute'}:
                    timepull_range = ('07:00', '16:00')
                elif temporal_granularity in {'fifteen_minute', 'one_hour'}:
                    timepull_range = ('04:00', '19:59')
                l = (pd.DataFrame(columns=['NULL'], 
                                  index=pd.date_range(today, tomorrow,
                                                      freq=temporal_params[temporal_granularity]['base_index_freq']))
                       .between_time(timepull_range[0], timepull_range[1])
                       .index.strftime('%Y-%m-%d %H:%M:%S')
                       .tolist()
                )
                ldf = pd.DataFrame(l)
                base_temp = base
                base = pd.concat([base_temp,ldf],axis=0)
                base.reset_index(drop = True, inplace=True)
        base.columns = ['timemerge']
        base['timemerge_dt']=base['timemerge'].apply(lambda x:dt.strptime(x,"%Y-%m-%d %H:%M:%S"))
    
        # Creating Day column in base
        base['day'] = base['timemerge_dt'].dt.strftime('%Y-%m-%d') #creating day column
    
        # Merge polygon source data df with timestamp base df
        masterbase_it = base.merge(df, on = ['timemerge', 'day'], how = 'left')
        masterbase_temp = masterbase
        masterbase = pd.concat([masterbase_temp,masterbase_it],axis=0)
        
        # Reseting index
        masterbase.reset_index(drop = True, inplace = True) #want to reset index
        
        # For unpaid polygon.io accounts, pause before moving to next date range due to API restrictions 
        if paid_polygon_account == False:
            time.sleep(13) # 5 API calls per minute are allowed

    
    days_source_data = len(masterbase.day.unique())

    # Remove days with excess consecutive rows of missing data within sub-1 hour temporal granularities
    cons_nan_days_dropped = []    
    if temporal_granularity in {'one_minute', 'three_minute', 'five_minute', 'fifteen_minute'}:
        grouped = masterbase.groupby('day')
        for day, group_df in grouped:     
            consecutive_nan_count = group_df['close'].isna().astype(int).groupby(group_df['close'].notna().cumsum()).cumsum()
            if max(consecutive_nan_count) > temporal_params[temporal_granularity]['consecutive_nan_limit']:
                cons_nan_days_dropped.append(day)
        cons_nan_dropped_data = masterbase[masterbase.day.isin(cons_nan_days_dropped)].reset_index(drop=True)
        masterbase = masterbase[~masterbase.day.isin(cons_nan_days_dropped)].reset_index(drop=True)
        
    
    # Get NaN count after dropping days with excessive NaN's
    na_count = masterbase['open'].isna().sum()
    pct_mfg_data = 100*round(na_count / len(masterbase),3)
    pct_mfg_data_string = str(pct_mfg_data) + '%'
    
    # Removing NaN's for close, open, high, low using a linear interpolation (values that would be on linear line between the point above and below)
    masterbase.loc[:, ['close', 'open', 'high', 'low']] = masterbase[['close', 'open', 'high', 'low']].interpolate(method='linear', limit_direction='both')
    
    # Calculating growth_rate and remove NaN row created in each caclulation period.
    masterbase['growth_rate'] = masterbase.close.pct_change()
    masterbase = masterbase[~masterbase.growth_rate.isna()].reset_index(drop=True)
    
    organized_columns = ['close', 'open', 'high', 'low', 'growth_rate', 'volume', 'vwap', 
                      'transactions', 'day', 
                      'timemerge', 'timemerge_dt'] 
    
    columns_to_impute = ['close', 'open', 'high', 'low', 'growth_rate', 'volume', 'vwap', 
                      'transactions']  #column names for after filling NA
    
    # Ordering the dataframe for imputation
    masterbase = masterbase[organized_columns] 
    
    # Filling in all NaN with the IterativeImputer, defualt is bayseian ridge model
    impute_it = IterativeImputer(max_iter=20)
    impute_df = pd.DataFrame(impute_it.fit_transform(masterbase.iloc[:,:-3]), columns = columns_to_impute) #index to disregard time columns

    # Joining the imputed data with the date and time data
    masterbase = pd.concat([impute_df, masterbase.iloc[:,-3:]],axis = 1)
    
    # Converting Volume and Transactions to integers
    masterbase['volume'] = masterbase['volume'].astype('int')
    masterbase['transactions'] = masterbase['transactions'].astype('int')
        
    # Calculate and add VWAP to the dataframe
    vwap = calc_vwap(masterbase, trade_days) # Calculating VWAP
    masterbase['vwap'] = vwap # Add VWAP to dataframe
    
    # Remove days with compromised data, indicated by static data and thus zero growth readings
    cons_0growth_days_dropped = []
    zero_growth_count = sum(masterbase['growth_rate'] == 0)
    zero_growth_percent = zero_growth_count / len(masterbase)
    if temporal_granularity in {'one_minute', 'three_minute', 'five_minute', 'fifteen_minute'}:
        grouped = masterbase.groupby('day') # Group the DataFrame by 'day'
        for day, group_df in grouped:     
            consecutive_zero_growth_count = (group_df['growth_rate'] == 0).astype(int).groupby(group_df['close'].notna().cumsum()).cumsum()
            if max(consecutive_zero_growth_count) > temporal_params[temporal_granularity]['cons_zero_growth_limit']:
                cons_0growth_days_dropped.append(day)
        zero_growth_dropped_data = masterbase[masterbase.day.isin(cons_0growth_days_dropped)].reset_index(drop=True)
        output_dataframe = masterbase[~masterbase.day.isin(cons_0growth_days_dropped)].reset_index(drop=True)
        output_dataframe.reset_index(drop=True, inplace=True) # Reset the index of the filtered DataFrame
    else:
       output_dataframe = masterbase.copy() 
    days_filtered_data = len(output_dataframe.day.unique())
    obvs_per_day = len(output_dataframe) / days_filtered_data
    
    report_stats = {'Datapull_NaN_Count_Post_Day_Drop': na_count,
                    'Datapull_Percent_MFG_Data': pct_mfg_data_string,
                    'Datapull_Total_Days': days_source_data,
                    'Datapull_Total_Usable_Days': days_filtered_data,
                    'Datapull_Observations_per_day_in_Usable_Data': obvs_per_day,
                    'Consecutive_Zero_Growth_Dropped_Days': cons_0growth_days_dropped,
                    'Consecutive_NaN_Dropped_Days': cons_nan_days_dropped,
                    'Datapull_Start_Date':start_date,
                    'Datapull_End_Date': end_date,
                    'Consecutive_NaN_Dropped_Data': cons_nan_dropped_data,
                    'Zero_Growth_Dropped_Data': zero_growth_dropped_data,
                    'Zero_Growth_Count': str(zero_growth_percent) + "% of data. " + str(zero_growth_count) + " rows in total."}

    return output_dataframe, report_stats

# ### TEST ###
# polygon_api_key = "GpjjoRW_XKUaCLvWWurjuMUwF34oHvpD" 
# tkr = "NVDA"
# temporal_granularity = 'fifteen_minute'
# start_date = "2021-11-23"
# end_date = "2023-11-19"
# paid_polygon_account=False

# dataframe, nan_report = raw_data(polygon_api_key, tkr, start_date, end_date, temporal_granularity, paid_polygon_account=False)





