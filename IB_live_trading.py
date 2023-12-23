#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script leverages the ib_insync package to connect and execute live trading functions
in real time. It deploys the most recently trained model for a given ticker symbol, as well
as the trading strategy loaded with the model.
It is recommended to run this script 15 minutes after markets open.
Note: static sell strategies are a deprecated feature, having been replaced by dynamic ones
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime as dt
import pytz
import random
from datetime import timedelta
import polygon_realtime_functions as prf
import pickle
from keras.models import load_model
import os
from ib_insync import *
from ib_insync import IB
import datetime
import IB_trade_functions as ib_funct

def establish_connection(ib, ib_port, max_retries=10):
    '''
    With Interactive Brokers' Trader Workstation application open, this function establishes 
    a connection to it for all API interactions.
    '''
    retries = 0
    while retries < max_retries:
        client_id = random.randint(1, 50000)
        try:
            print(f"Trying to connect with client ID {client_id}...")
            IB.connect(ib, port=ib_port, clientId=client_id)
            if IB.isConnected(ib):
                print(f"Connection established with client ID {client_id}")
                return client_id
        except Exception as e:
            print(f"Connection failed with client ID {client_id}: {str(e)}")
        retries += 1
        time.sleep(2)  # Wait for a moment before retrying
    print("Max retries reached, unable to establish a connection.")
    return False

def find_most_recent_folder(tkr):
    '''
    This function identifies the name of the most recent folder and thus model for a given stock ticker
    '''
    folder_names = [name for name in os.listdir() if os.path.isdir(name) and name.startswith(tkr)]
    if not folder_names:
        return None

    sorted_folders = sorted(folder_names, reverse=True)
    return sorted_folders[0]

## Pull the output dictionary and model from the model folder
def pull_output_dict_and_nn_model(folder_name):
    '''
    This function loads and locally saves the model and output_dict from the model training process.
    '''
    with open(os.path.join(folder_name, 'output_dict.pkl'), 'rb') as f:
        output_dict = pickle.load(f)
    model_file_path = os.path.join(folder_name, 'nn_model.h5')
    nn_model = load_model(model_file_path, compile=False)
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return output_dict, nn_model

def log_pred_and_data(pred_log, minute_wise_data, minute_wise_scaled_data):
    '''
    This model logs the current time period's prediction and data for future verifiability.
    '''
    loop_start_time_string = dt.now().astimezone(pytz.timezone('America/New_York')).strftime('%H:%M:%S')
    data = prf.datapull_until_now(tkr, key=paid_polygon_key, yesterday=False)
    X_vector, scaled_data = prf.current_minute_X(data, indicator_params)
    if X_vector.isna().any().any():
        # Record the current time and DataFrame in a list
        nan_report.append([tkr, loop_start_time_string, X_vector])
        pred = None
    else:
        pred = nn_model.predict(X_vector)
    pred_made_dt = dt.now().astimezone(pytz.timezone('America/New_York'))
    pred_made_string = pred_made_dt.strftime('%H:%M:%S.%fZ')
    latest_pred = {'tkr': [tkr], 'pred': [pred], 'time_of_pred':[pred_made_string]}
    pred_log = pd.concat([pred_log, pd.DataFrame(latest_pred)], ignore_index = True)        
    minute_wise_scaled_data[loop_start_time_string] = scaled_data
    minute_wise_data[loop_start_time_string] = data
    return pred_log, minute_wise_data, minute_wise_scaled_data
    

## API KEYS
paid_polygon_key = "GpjjoRW_XKUaCLvWWurjuMUwF34oHvpD" #Paid subscription key

# Defining the stock to be traded
tkr = 'NVDA'

# Load model and model information
folder_name = find_most_recent_folder(tkr)
output_dict, nn_model = pull_output_dict_and_nn_model(folder_name)

# Selected indicator parameters for generating minute by minute data
indicator_params = output_dict[tkr + '_selected_indicators']

# Prediction parameters from loaded dictionary
rounding_threshold = output_dict[tkr + '_optimal_trade_strategy']['optimal_sell_strat_params']['rounding_threshold']
lag = output_dict[tkr + '_selected_indicators']['lag'][0]
growth_threshold = output_dict[tkr + '_selected_indicators']['growth_threshold'][0]
delay_past_minute = 5

## Trading parameters from loaded dictionary
hold_strategy = output_dict[tkr + '_optimal_trade_strategy']['optimal_sell_strat_params']['hold_strategy_employed']
sell_strategy = output_dict[tkr + '_optimal_trade_strategy']['optimal_sell_strat_params']['optimal_strategy']
first_hold_period = lag
if sell_strategy != 'hold_strategy_alone':
    loss_threshold = output_dict[tkr + '_optimal_trade_strategy']['optimal_sell_strat_params']['opt_loss_thresh']
if sell_strategy in ['lock_profit_htn_wstop_loss', 'lock_profit_wtrail_wstop_loss']:
    profit_target = output_dict[tkr + '_optimal_trade_strategy']['optimal_sell_strat_params']['opt_profit_target']
if sell_strategy == 'lock_profit_wtrail_wstop_loss':
    trail_percent = output_dict[tkr + '_optimal_trade_strategy']['optimal_sell_strat_params']['opt_trail_percent']
if hold_strategy == 'hold_til_no_buys_with_hold':
    hold = output_dict[tkr + '_optimal_trade_strategy']['optimal_sell_strat_params']['hold']
    first_hold_period = first_hold_period + hold

## Record Keeping
nan_report = []
buy_log = pd.DataFrame(columns = ['tkr', 'return', 'pred', 'percent_filled', 'pred_price', 'purchase_price', 'sold_price', 'sale_triggered_by', 'expected_sale_price',
                                  'buy_trade_id', 'num_shares_desired', 'num_shares_purchased', 'time_to_sell', 
                                  'day', 'pred_made_time', 'buy_submitted_time', 'sale_submitted_time', 'sold_time' ]) 
minute_wise_data = {}
minute_wise_scaled_data = {}
canceled_orders = {}
fulfilled_orders = {}
pred_log = pd.DataFrame(columns = ['tkr', 'pred', 'time_of_pred'])
sale_orders = {}
status_log = {}
prediction_vectors = {}

# Timezone establishment
eastern_tz = pytz.timezone('US/Eastern')

util.startLoop() # Allows for synchronous connections
ib = IB()
ib_port = 1234

client_id = establish_connection(ib, ib_port)

## Investment size parameters for each buy
num_stocks_in_basket = 4
buying_power = ib_funct.acct_buying_power(ib)
stock_allowance = int(np.floor((buying_power / num_stocks_in_basket) * 0.96)) # Giving a 4% loss cushion

### Loop Start ###
today = dt.now(eastern_tz)
today_day = today.astimezone(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')
target_time = eastern_tz.localize(dt(today.year, today.month, today.day, 16, 0) - timedelta(minutes=float(lag)))
finish_line = eastern_tz.localize(dt(today.year, today.month, today.day, 16, 0))- timedelta(minutes=1.0)

# Wait until the next whole minute to start the live trading loop
IB.sleep(prf.seconds_until_next_whole_minute()+ delay_past_minute)
# Continue the loop until the current time is before the target time  
while (dt.now(eastern_tz) < target_time):
    ## Document loop start time
    loop_start_time_string = dt.now().astimezone(pytz.timezone('America/New_York')).strftime('%H:%M:%S')
    loop_start_time_dt = dt.now().astimezone(pytz.timezone('America/New_York'))    

    ## Data Pull up to the current minute
    data = prf.datapull_until_now(tkr, key=paid_polygon_key, yesterday=False)
    current_price = round(float(data.close.tail(1).iloc[0]),2)

    ## Model Prediction on this minute's X-Vector
    X_vector, scaled_data = prf.current_minute_X(data, indicator_params)
    # print(X_vector)
    if X_vector.isna().any().any():
        # Record the current time and DataFrame in a list
        nan_report.append([tkr, loop_start_time_string, X_vector])
        buy_or_nah = 0
        pred = None
        prediction_vectors[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = X_vector
    else:
        pred = nn_model.predict(X_vector)
        buy_or_nah = (pred > rounding_threshold)[0][0]
        prediction_vectors[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = X_vector
    pred_made_dt = dt.now().astimezone(pytz.timezone('America/New_York'))
    pred_made_string = pred_made_dt.strftime('%H:%M:%S.%fZ')

    # ##Simulate buy periodically to test functionality throughout the day
    # if pred_made_dt.minute in []: # Check if it is the 5th, 25th, or 45th minute of the day
    #     buy_or_nah = np.array([True])[0]
    
    # # ##Force Simulate buy
    # buy_or_nah = np.array([True])[0]
    # pred = 'forced_buy' 
    
    if buy_or_nah:
        # Don't buy at a price that sacrifices more than 20% of predicted growth
        limit_price = round(current_price * (1 + (growth_threshold)/5), 2) 
        num_shares_tobuy = np.floor(stock_allowance / limit_price)
        
        ## Execute buy
        buy_trade, buy_order, buy_contract = ib_funct.buy_adaptive(ib,tkr, num_shares_tobuy, limit_price, urgency = "Normal") 
        buy_trade_dict = ib_funct.trade_to_dict(buy_trade)
        we_sold = False

        # Print this minutes result in the console and save it to the status_log
        print(f'Prediction: {pred}. At time: {pred_made_string}. Buy: {buy_or_nah}')
        status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = f'Prediction: {pred}. At time: {pred_made_string}. Buy: {buy_or_nah}'
        status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = f'BUY ORDER SENT. Intelligent trade process initiated with Hold Strategy: "{hold_strategy}" and Sell Strategy: "{sell_strategy}".'
        
        ## Create dict to append into buy_log
        stock_transaction = {column: None for column in buy_log.columns} #creating empty dictionary 
        
        # Keeping up with minute-by-minute log of preds and data
        latest_pred = {'tkr': [tkr], 'pred': [pred], 'time_of_pred':[pred_made_string]}
        pred_log = pd.concat([pred_log, pd.DataFrame(latest_pred)], ignore_index = True)      
        minute_wise_scaled_data[loop_start_time_string] = scaled_data
        minute_wise_data[loop_start_time_string] = data 
        
        ## Wait until the next minute
        IB.sleep(prf.seconds_until_next_whole_minute() + 5)
        ## Perform prediction to check for immediate consecutive buy order
        immediate_buy_again, pred_log, minute_wise_scaled_data, minute_wise_data = prf.check_4_immediate_buy(tkr, paid_polygon_key, pred_log, indicator_params, nn_model, rounding_threshold, minute_wise_scaled_data, minute_wise_data, loop_start_time_string, nan_report)
        ## Sleep until end of minute to allow original order to (hopefully) fulfill
        IB.sleep(prf.seconds_until_next_whole_minute() - 5)

        ## RECORD FULFILLMENT INFORMATION, SET SELL TIME in stock_transaction
        buy_trade_dict = ib_funct.trade_to_dict(buy_trade)
                
        ## See if the trade was filled and update status_log accordingly
        trade_filled = False
        num_filled = buy_trade_dict['orderStatus']['filled']
        remaining_tofill = buy_trade_dict['orderStatus']['remaining']
        if remaining_tofill > 0 and num_filled > 0:
            status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = f'BUY ORDER PARTIALLY FILLED: {num_filled} of {remaining_tofill+num_filled} shares purchased. {100*round(num_filled / (remaining_tofill+num_filled),2)}% fulfilled.'
            canceled_buy = ib.cancelOrder(buy_order) ## CANCEL ORDER
            canceled_orders[str(dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S'))] = ib_funct.trade_to_dict(canceled_buy)
            trade_filled = True
        elif remaining_tofill == 0:
            status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = f'BUY ORDER FULFILLED: {num_filled} shares purchased.'
            trade_filled = True
        ## If trade wasn't filled after 2 minutes, cancel order
        elif num_filled == 0:
            canceled_buy = ib.cancelOrder(buy_order) ## CANCEL ORDER
            canceled_orders[str(dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S'))] = ib_funct.trade_to_dict(canceled_buy)
            status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = 'ORDER CANCELLED: No shares purchesd.'
            IB.sleep(prf.seconds_until_next_whole_minute()+ delay_past_minute)
        
        ## If trade was filled, proceed with record keeping and Intelligent Trade Strategy
        if trade_filled:
            buy_trade_dict = ib_funct.trade_to_dict(buy_trade) # Pulling fresh buy order data
            fulfilled_orders[str(dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S'))] = buy_trade_dict
            status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = 'Recording buy order data now...'
            stock_transaction['tkr'] = [tkr]
            stock_transaction['day'] = [today_day]
            stock_transaction['pred'] = [pred[0][0]]
            stock_transaction['percent_filled'] = [100 * (buy_trade_dict['orderStatus']['filled'] / (buy_trade_dict['orderStatus']['filled'] + buy_trade_dict['orderStatus']['remaining']))]
            stock_transaction['pred_price'] = [current_price]
            stock_transaction['purchase_price'] = [buy_trade_dict['orderStatus']['avgFillPrice']]
            stock_transaction['expected_sale_price'] = [current_price * (1 + growth_threshold)]
            
            stock_transaction['num_shares_desired'] = [num_shares_tobuy]
            stock_transaction['num_shares_purchased'] = [buy_trade_dict['orderStatus']['filled']]
            
            stock_transaction['pred_made_time'] = [pred_made_string]
            for log_entry in buy_trade_dict['log']:
                if log_entry.get('status') == 'Filled':
                    buy_filled_time = pytz.utc.localize(dt.strptime(log_entry.get('time'), '%Y-%m-%d %H:%M:%S.%f'))
                    buy_filled_time_et = buy_filled_time.astimezone(eastern_tz)
                    stock_transaction['buy_filled_time'] = [buy_filled_time_et]
                elif log_entry.get('status') == 'PendingSubmit':
                    buy_submitted_time = pytz.utc.localize(dt.strptime(log_entry.get('time'), '%Y-%m-%d %H:%M:%S.%f'))
                    buy_submitted_time_et = buy_submitted_time.astimezone(eastern_tz)
                    stock_transaction['buy_submitted_time'] = [buy_submitted_time_et]
            
            # Establish the time to sell based on model's 'lag' and the hold_strategy                          
            time_to_sell_prelim = buy_submitted_time_et + timedelta(minutes=float(first_hold_period))
            rounded_time = time_to_sell_prelim.replace(second=0, microsecond=0) + timedelta(milliseconds=1.0)
            stock_transaction['time_to_sell'] = [rounded_time]
            if immediate_buy_again ==True:
                stock_transaction['time_to_sell'] = [rounded_time + timedelta(minutes=1.0)]
            stock_transaction['pred_made_time'] = [pred_made_string]
            
            
            ## Wait until the next minute to begin intelligent trade execution
            IB.sleep(prf.seconds_until_next_whole_minute()+ delay_past_minute)
            
            while we_sold == False:
                if dt.now(eastern_tz) >= (finish_line - timedelta(seconds=60.0)): ## If its the end of the trading day, sell stock
                    ## SELL STOCK()
                    sale_order = ib_funct.sell_adaptive(ib, tkr, stock_transaction['num_shares_purchased'][0])
                    status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = 'End of day sale executed.'
                    we_sold = True
                    stock_transaction['sale_triggered_by'] = ['End of Day']
                
                ## Sell Strategy check then execution if thresholds are met, given the optimal sell strat.
                ## The entire Sell Strategy is implemented before moving on to time_to_sell adjustment with the Hold Strategy
                if sell_strategy != 'hold_strategy_alone':
                    
                    profit_loss = prf.profit_loss(paid_polygon_key, tkr, stock_transaction['purchase_price'][0])  ## Find profit/loss
                    
                    if profit_loss < loss_threshold: # Applies to all sell strategies ('stop_loss', 'lock_profit_htn_wstop_loss', and 'lock_profit_wtrail_wstop_loss')
                        ## SELL STOCK() ## Initiating sale if losses exceed loss_threshold
                        sale_order = ib_funct.sell_adaptive(ib, tkr, stock_transaction['num_shares_purchased'][0])
                        status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = f"SALE INITIATED. Profit/Loss dropped below optimal loss threshold of {loss_threshold}."
                        we_sold = True
                        stock_transaction['sale_triggered_by'] = ['Loss Threshold Crossed']
                        pred_log, minute_wise_data, minute_wise_scaled_data = log_pred_and_data(pred_log, minute_wise_data, minute_wise_scaled_data) # 1 Keeping up with minute-by-minute log of preds and data
                    
                    ## Initiating sale if profit target is met    
                    elif (profit_loss > profit_target) and (sell_strategy != 'stop_loss'): 
                        status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = f'Profit/Loss exceeded target profit threshold of {profit_target}. Selling based on Sell Strategy: "{sell_strategy}"'    
                        if sell_strategy == 'lock_profit_htn_wstop_loss':
                            while we_sold == False: # This loop continues until the stock is sold (no hold time extension possible)
                                data = prf.datapull_until_now(tkr, key=paid_polygon_key, yesterday=False)
                                if (data['growth_rate'].iloc[-1] < 0) or (dt.now(eastern_tz) >= finish_line - timedelta(minutes=1.0)):
                                    ## SELL STOCK()
                                    sale_order = ib_funct.sell_adaptive(ib, tkr, stock_transaction['num_shares_purchased'][0])
                                    status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = 'SALE INITIATED. First minute of negative growth recorded (or end of day).'
                                    we_sold = True
                                    stock_transaction['sale_triggered_by'] = ['Neg. Growth After Profit Target Met']
                                    pred_log, minute_wise_data, minute_wise_scaled_data = log_pred_and_data(pred_log, minute_wise_data, minute_wise_scaled_data) # 2 Keeping up with minute-by-minute log of preds and data
                                else:
                                    status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = 'Positive growth recorded, holding stock for another minute...'
                                    pred_log, minute_wise_data, minute_wise_scaled_data = log_pred_and_data(pred_log, minute_wise_data, minute_wise_scaled_data) # 3 Keeping up with minute-by-minute log of preds and data
                                    IB.sleep(prf.seconds_until_next_whole_minute()+ delay_past_minute)
                        
                        elif sell_strategy == 'lock_profit_wtrail_wstop_loss':
                            sale_order, trail_order = ib_funct.sell_trailing_pct_adaptive(ib, tkr, stock_transaction['num_shares_purchased'][0], trail_percent*100) #trail percent argument should be entered as full percentage points
                            portfolio = IB.portfolio(ib)
                            status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = f'SALE INITIATED. Trailing stop loss order placed with a trail percentage of {trail_percent}.'
                            
                            while we_sold == False: # This loop continues until a sale is confirmed
                                pred_log, minute_wise_data, minute_wise_scaled_data = log_pred_and_data(pred_log, minute_wise_data, minute_wise_scaled_data) # 4 Keeping up with minute-by-minute log of preds and data    
                                if dt.now(eastern_tz) >= (finish_line - timedelta(minutes=1.0)):
                                    canceled_trail = ib.cancelOrder(trail_order) ## CANCEL ORDER
                                    sale_order = ib_funct.sell_adaptive(ib, tkr, stock_transaction['num_shares_purchased'][0])
                                while (prf.seconds_until_next_whole_minute() >= 4) and (we_sold == False): 
                                    portfolio = IB.portfolio(ib)
                                    if not any(item.contract.symbol == tkr for item in portfolio):
                                        status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = f'Trailing stop loss sale triggered. {tkr} stocks sold.'
                                        we_sold = True
                                        stock_transaction['sale_triggered_by'] = ['Trailing Stop Loss']
                                    else:
                                        IB.sleep(2.0)
                                if we_sold == False:
                                    IB.sleep(prf.seconds_until_next_whole_minute()+ delay_past_minute)
                
                ## Hold Strategy Execution Begins                    
                ## If current time is before time_to_sell, and we haven't sold due to the sell_strategy...
                if (dt.now(eastern_tz) < stock_transaction['time_to_sell'][0]) and (we_sold == False):
                    
                    ## Perform data pull and model prediction again to extend hold period as needed
                    data = prf.datapull_until_now(tkr, key=paid_polygon_key, yesterday=False)
                    current_price = round(float(data.close.tail(1).iloc[0]),2)
                    X_vector, scaled_data = prf.current_minute_X(data, indicator_params)
                    if X_vector.isna().any().any():
                        nan_report.append([tkr, loop_start_time_string, X_vector])
                        buy_again = False
                    else:
                        pred = nn_model.predict(X_vector)
                        buy_again = (pred > rounding_threshold)[0][0]
                    
                    ## If we get another buy order, extend the time_to_sell
                    if buy_again:
                        status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = f'Additional buy order generated. Updating time_to_sell based on Hold Strategy: "{hold_strategy}" '
                        # Extract the original time_to_sell for later comparison
                        original_time_to_sell = stock_transaction['time_to_sell'][0]
                        status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = f'Original time_to_sell: {original_time_to_sell}'
                        
                        # Calculate the new_time_to_sell and assign it based on hold_strategy
                        if hold_strategy in ['hold_til_no_buys', 'hold_til_no_buys_til_neg']:
                            time_to_add = float(lag)
                        elif hold_strategy == 'hold_til_no_buys_with_hold':
                            time_to_add = float(lag + hold)
                        else:
                            time_to_add = 0.0
                        
                        # Check if assigning time_to_sell to be after 4 PM and adjust accordingly
                        if stock_transaction['time_to_sell'][0] + timedelta(minutes = time_to_add) < finish_line:
                            stock_transaction['time_to_sell'] = [dt.now(eastern_tz).replace(second=0, microsecond=0) + timedelta(minutes = time_to_add)]
                        else:
                            stock_transaction['time_to_sell'] = [finish_line - timedelta(minutes=1.0)]  # Selling at 3:59 PM
                        
                        # Update the time_to_sell
                        new_time_to_sell = stock_transaction['time_to_sell'][0]
                        status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = f'Variable time_to_sell timestamp updated under Hold Strategy: "{hold_strategy}"'
                        status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = f'Variable time_to_sell updated from {original_time_to_sell} to {new_time_to_sell}, which should be {time_to_add} minutes from now.'
                        
                        pred_log, minute_wise_data, minute_wise_scaled_data = log_pred_and_data(pred_log, minute_wise_data, minute_wise_scaled_data)  # Keeping up with minute-by-minute log of preds and data
                        ## With another generated buy order, we have adjusted the time_to_sell time, and continue on to the next minute to apply Hold/Sell Strategies again
                        IB.sleep(prf.seconds_until_next_whole_minute()+ delay_past_minute)
                    else: # If no sale is triggered and hold period is left as is, we wait until the next minute to apply Hold/Sell Strategies again
                        status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = 'No additional buy generated. Sell time remains as is.'
                        pred_log, minute_wise_data, minute_wise_scaled_data = log_pred_and_data(pred_log, minute_wise_data, minute_wise_scaled_data) # 5 Keeping up with minute-by-minute log of preds and data    
                        IB.sleep(prf.seconds_until_next_whole_minute()+ delay_past_minute)
                
                # If it is past the time_to_sell and we haven't sold yet from the sell_strategy
                if (dt.now(eastern_tz) >= stock_transaction['time_to_sell'][0]) and (we_sold == False):
                    if hold_strategy in ['hold_til_no_buys', 'hold_til_no_buys_with_hold']:
                        sale_order = ib_funct.sell_adaptive(ib, tkr, stock_transaction['num_shares_purchased'][0]) ## SELL STOCK()
                        status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = 'SALE INITIATED. Lag (+ Hold) time expired.'
                        we_sold = True
                        stock_transaction['sale_triggered_by'] = ['Lag (+Hold) Expired']
                        pred_log, minute_wise_data, minute_wise_scaled_data = log_pred_and_data(pred_log, minute_wise_data, minute_wise_scaled_data) # 6 Keeping up with minute-by-minute log of preds and data
                    elif hold_strategy == 'hold_til_no_buys_til_neg':
                        status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = 'Lag (+ Hold) time expired. Waiting negative growth to initiate sale...'
                        while we_sold == False: # This loop continues until a sale is executed
                            data = prf.datapull_until_now(tkr, key=paid_polygon_key, yesterday=False)
                            if (data['growth_rate'].iloc[-1] < 0) or (dt.now(eastern_tz) >= finish_line - timedelta(minutes=1.0)):
                                sale_order = ib_funct.sell_adaptive(ib, tkr, stock_transaction['num_shares_purchased'][0]) ## SELL STOCK()
                                status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = 'SALE INITIATED. Negative growth recorded after expiry or lag (+hold).'
                                we_sold = True
                                stock_transaction['sale_triggered_by'] = ['Neg. Growth After Hold Expired']
                                pred_log, minute_wise_data, minute_wise_scaled_data = log_pred_and_data(pred_log, minute_wise_data, minute_wise_scaled_data) # 7 Keeping up with minute-by-minute log of preds and data
                            else:
                                pred_log, minute_wise_data, minute_wise_scaled_data = log_pred_and_data(pred_log, minute_wise_data, minute_wise_scaled_data) # 8 Keeping up with minute-by-minute log of preds and data
                                IB.sleep(prf.seconds_until_next_whole_minute()+ delay_past_minute)
        
            ## In the case of a sale, wait for the details to be settled then record transaction data
            status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = 'sale assumed to be executed, waiting until :55 to collect data'
            IB.sleep(prf.seconds_until_next_whole_minute() -4) # Wait for sale execution to complete
            
            ## RECORD DETAILS OF SALE INTO stock_transaction
            sale_order_dict = ib_funct.trade_to_dict(sale_order)
            sale_orders[str(dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S'))] = sale_order_dict
            
            stock_transaction['sold_price'] = [sale_order_dict['orderStatus']['avgFillPrice']]
            for log_entry in sale_order_dict['log']:
                if log_entry.get('status') == 'Filled':
                    sell_filled_time = pytz.utc.localize(dt.strptime(log_entry.get('time'), '%Y-%m-%d %H:%M:%S.%f'))
                    sell_filled_time_et = sell_filled_time.astimezone(eastern_tz)
                    stock_transaction['sold_time'] = [sell_filled_time_et]
                elif log_entry.get('status') == 'PendingSubmit':
                    sale_submitted_time = pytz.utc.localize(dt.strptime(log_entry.get('time'), '%Y-%m-%d %H:%M:%S.%f'))
                    sale_submitted_time_et = sale_submitted_time.astimezone(eastern_tz)
                    stock_transaction['sale_submitted_time'] = [sale_submitted_time_et]
            stock_transaction['return'] = [stock_transaction['sold_price'][0] / stock_transaction['purchase_price'][0]]
            
            # Record the stock_transaction in the buy_log
            buy_log = pd.concat([buy_log,pd.DataFrame(stock_transaction)], ignore_index = True)
            status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = f"Sale execution recorded at {dt.now().astimezone(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')}"
            IB.sleep(prf.seconds_until_next_whole_minute()+ delay_past_minute)
    
    # If no buy is triggered, record it in the status log and wait til the next whole minute
    elif buy_or_nah == False:
        # Print this minutes result in the console and save records accordingly
        print(f'Prediction: {pred}. At time: {pred_made_string}. Buy: {buy_or_nah}')
        status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = f'Prediction: {pred}. At time: {pred_made_string}. Buy: {buy_or_nah}'
        pred_log, minute_wise_data, minute_wise_scaled_data = log_pred_and_data(pred_log, minute_wise_data, minute_wise_scaled_data)
        IB.sleep(prf.seconds_until_next_whole_minute()+ delay_past_minute)  
status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = 'Minute-by-minute predictions halted.'

# At the end of the day, if any transactions took place, save day's logs
if len(buy_log) > 0:    
    hold_periods = prf.gen_hold_periods(buy_log, tkr, paid_polygon_key)
    order_record = {'fullfilled_orders': fulfilled_orders, 
                    'sale_orders':sale_orders,
                    'canceled_orders': canceled_orders}
    
    day_log = {'buy/sell_log': buy_log, 
               'status_log': status_log,
               'nan_report': nan_report, 
               'minutewise_data': minute_wise_data, 
               'minutewise_data_scaled': minute_wise_scaled_data, 
               'prediction_log': pred_log,
               'prediction_vectors': prediction_vectors,
               'order_record': order_record,
               'hold_periods': hold_periods}
    file_name = loop_start_time_dt.strftime('%Y-%m-%d_') + tkr + '_trade_log'
    with open(os.path.join(folder_name, file_name), 'wb') as f:
        pickle.dump(day_log, f)        
    status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = 'Trade record saved.'
    print('Trade record saved.')
else:
    status_log[dt.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S.%f')] = 'No trades made, thus no trade record saved.'
    print('No trades made, thus no trade record saved.')
