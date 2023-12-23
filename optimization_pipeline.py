#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains functions that connect the data collection and model creation process, collectively creating a flexible optimization pipeline.
clean_features() uses custom feature formulas to optimize their input parameters with historical stock data through the DEAP_ensemble function.
cleaned_to_X() uses the optimized custom features to generate all data needed for model training. It also lists selected features and provides an NaN report.
standard_nn() uses trains a neural net model on Xy, the output of cleaned_to_X(). Its defined model architecture stabilizes resulting model performance and shortens computation time.
AK() employs AutoKeras' StructuredDataClassifier to explore nueral net model architectures, spitting out the optimal model and graphing results if desired.
Not utilized in the final pipeline, RF_gutcheck() employs a random forest model to provide a scoring_metric score and graph similar to AK() for quick comparison. 
"""

import pandas as pd
import numpy as np
import feature_origin as fo
import temporal_backend as tb
import Indicator_Building_Blocks as ind
import DEAP_precision as DP
import time
import inspect
import ast
from datetime import datetime as dt
from datetime import time
import datetime
import time
import DEAP_ensemble as DPe
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
import autokeras as ak
import tensorflow.keras.metrics
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, ReLU, Dropout, BatchNormalization, Activation, LayerNormalization, LSTM
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier


def clean_features(tkr, dataframe, temporal_granularity, N_obvs_used4_deap=25000, deployment_model=False, pseudo_test_days=0, 
                   troubleshooting=False, clean_generations=1, assimilated_generations=3, population_scalar=23, use_apex=True, fresh_apex=False, use_apex_genesis=False):
    '''
    This function performs the entire genetic programming process for a given stock to create
    custom features.
    
    Parameters
    ----------
    tkr : TYPE string
        DESCRIPTION. The ticker symbol of the stock of interest for the model
    dataframe : TYPE pandas.DataFrame
        DESCRIPTION. All stock data avalailable for train/test set assignment; adhering to the format of the output from 
        raw_data() in raw_data.py
    temporal_granularity : TYPE string
        DESCRIPTION. Duration of aggregate stock data bars used to generate the data.
    N_obvs_used4_deap : TYPE, integer
        DESCRIPTION. The number of observations desired in the dataframe used for the DEAP optimization
        Because of slow computational speed, this dataset is reduced.
    deployment_model : TYPE, boolean
        DESCRIPTION. If intended to be deployed for live trading, train/test sets are adjusted accordingly.
    pseudo_test_days : TYPE, integer
        DESCRIPTION. The number of final days in the dataset to be used as a test set, if desired.
    troubleshooting : TYPE, boolean
        DESCRIPTION. The default is False. True limits the number of indicators to be cleaned, saving computational time 
        while troubelshooting.
    clean_generations : TYPE, integer
        DESCRIPTION. The number of generations of features to test and evolve with flexible 'lag' and 'growth_threshold' 
        values, after generation 0.
    assimilated_generations : TYPE, integer
        DESCRIPTION. The number of generations of features to test and evolve with fixed 'lag' and 'growth_threshold' 
        values, after generation 0.
    population_scalar : TYPE, integer
        DESCRIPTION. The default is 23. For each feature parameter listed in the input parameter dictionary, this many 
        individuals will be made per generation.
    use_apex : TYPE, boolean
        DESCRIPTION. When True, and if a previous record of high performing features is available, their top performing 
        'Individuals' will be injected into generation 0.
    fresh_apex : boolean
        DESCRIPTION. If there is a change in primary scoring metric or general strategy, setting this to True will reset 
        the apex record using the newest optimal indicators.
    use_apex_genesis : TYPE boolean
        DESCRIPTION. When True, apex 'Individuals' with the same temporal_granularity (irregardless of the stock) will be used. 
        Ideal when training a model on a new stock.


    Returns
    -------
    round_of_cleaned : TYPE, pandas.DataFrame
        DESCRIPTION. A dataframe of the best performing version of each custom indicator function. 
        Columns include the indicator name, lag value, growth threshold value, primary objective score of the optimal 
        input parameters, and a dictionary of optimal input parameters.


    '''    
    # Given the number of observatiosn desired for the DEAP optimization, how many days of data are needed
    deap_days = int(N_obvs_used4_deap / (len(dataframe) / dataframe.day.nunique()))
    if deap_days > dataframe.day.nunique():
        deap_days = dataframe.day.nunique()
    
    ## deployment_model and pseudo_test_days allow for precise or general selection of a test set
    if deployment_model == False: # 80/20 train/test split for general model performance review
        cutoff = round(0.8 * dataframe.day.nunique())
        unique_days_selected = dataframe.day.unique()[cutoff-deap_days:cutoff]
        stock_df = dataframe[dataframe['day'].isin(unique_days_selected)].reset_index(drop=True)
    
    # Select DEAP optimization data for deployment model with desired test days
    elif deployment_model == True and pseudo_test_days > 0:
        unique_days_selected = dataframe.day.unique()[-deap_days-pseudo_test_days:-pseudo_test_days]
        stock_df = dataframe[dataframe['day'].isin(unique_days_selected)].reset_index(drop=True)
        
    # Select DEAP optimization data for deployment model with no test days
    elif deployment_model == True and pseudo_test_days == 0:
        unique_days_selected = dataframe.day.unique()[-deap_days:]
        stock_df = dataframe[dataframe['day'].isin(unique_days_selected)].reset_index(drop=True)
    
    # Load all custome indicator (feature) functions present in feature_origin.py
    need_cleaning = [name for name, obj in inspect.getmembers(fo) if inspect.isfunction(obj)]
    
    if troubleshooting == True: # Reducing computational time if troubleshooting
        need_cleaning = need_cleaning[2:4]
    
    # Create lists to access functions and dictionaries properly
    func_list = [getattr(fo, name) for name in need_cleaning]
    dict_list = [getattr(fo, name + '_deap_params') for name in need_cleaning]
    for dictionary in dict_list:
        dictionary.update(tb.lag_growth_thresh_tiers[temporal_granularity]) # Add temporally specific lag/growth_threshold value ranges
    
    # Ensure that each dictionary of properly named parameters exists in 'fo' script
    for attribute_name in [name + '_deap_params' for name in need_cleaning]:
        assert hasattr(fo, attribute_name), f"{attribute_name} missing from feature_origin.py"

    # Ensure that each feature only adds one column to the df
    df_mini = dataframe[:500].copy()
    num_columns = len(df_mini.columns)
    for i in range(len(func_list)):
        df_plus_one = df_mini.copy()
        function = func_list[i]
        deap_params = dict_list[i]
        random_params = DP.initialize_individual(deap_params)
        remaining_params = {
            k: v for k, v in random_params.items()
            if k not in ["lag", "growth_threshold"]
        }
        df_plus_one = function(df_plus_one, **remaining_params)
        if len(df_plus_one.columns) != num_columns + 1:
            print(df_plus_one.columns)
        assert len(df_plus_one.columns) == num_columns + 1, f'{function} added more than one column to dataframe'

    # Perform cleaning on filtered, functioning, one-column indicators
    round_of_cleaned = DPe.DEAP_ensemble(tkr, stock_df, func_list, dict_list, temporal_granularity, 
                                         population_scalar=population_scalar, clean_generations=clean_generations, assimilated_generations=assimilated_generations, 
                                         min_scoring_quantile=0.75, use_apex=use_apex, use_apex_genesis=use_apex_genesis)
    
    ## Load, clean and save the new Apex Record
    if (use_apex == True) and (troubleshooting == False) and (fresh_apex == False): #Add optimal indicator parameters to apex_record for future reference
        try:    # If any record exists...
            apex_df = pd.read_csv(tkr + '_' + temporal_granularity + '_apex_record.csv') #load record
            if not apex_df.empty: 
                apex_df['cleaned_params'] = apex_df['cleaned_params'].apply(ast.literal_eval) # re-assignging the params column to be dictionaries    
                
                ## Drop duplicates based on 'cleaned_params', keeping the most recent occurrence's scoring_metric performance
                for indicator_value in round_of_cleaned['indicator'].unique():
                    round_rows = round_of_cleaned[round_of_cleaned['indicator'] == indicator_value]
                    apex_rows = apex_df[apex_df['indicator'] == indicator_value]
                    for _, round_row in round_rows.iterrows():
                        matching_rows = apex_rows[ # Check if both dataframes have the same optimal params for the same indicator
                            (apex_rows['cleaned_params'] == round_row['cleaned_params']) &
                            (apex_rows['indicator'] == round_row['indicator'])]
                        non_matching_rows = []
                        if not matching_rows.empty:
                            apex_df.loc[matching_rows.index, 'scoring_metric'] = round_row['scoring_metric'] # For all with the same parameters, swap out the scoring_metric for the newest one
                        else:
                            non_matching_rows.append(round_row.to_dict()) # All unique optimal parameter values saved to be added to the apex_record
                apex_df = pd.concat([apex_df, pd.DataFrame(non_matching_rows)], axis=0, ignore_index=True) # Concatenate updated apex_record with the unique optimal params from this round of DEAP
                apex_df = apex_df.drop_duplicates()
                
                apex_df.to_csv(tkr + '_' + temporal_granularity + '_apex_record.csv', index=False) #save duplicate-free, updated record
            elif apex_df.empty: # If record exists but is an empty dataframe
                round_of_cleaned.to_csv(tkr + '_' + temporal_granularity + '_apex_record.csv', index=False) # Save most recent round as a refreshed record
        except: # If no record exists...
            round_of_cleaned.to_csv(tkr + '_' + temporal_granularity + '_apex_record.csv', index=False) # Save most recent round as a refreshed record
            pass
    
    ## Reset the apex_record by saving it with only this model's optimal features
    elif fresh_apex == True: 
        round_of_cleaned.to_csv(tkr + '_' + temporal_granularity + '_apex_record.csv', index=False) # Save most recent round as a refreshed record
    
    ## Add these optimal parameters to Apex_Genesis
    if (troubleshooting == False) and (fresh_apex == False): 
        try:    
            apex_genesis_df = pd.read_csv('apex_genesis_' + temporal_granularity + '.csv') #load record
            apex_genesis_df = pd.concat([apex_genesis_df, round_of_cleaned])
            apex_genesis_df.to_csv('apex_genesis_' + temporal_granularity + '.csv', index=False) #save record
        except:
            round_of_cleaned.to_csv('apex_genesis_' + temporal_granularity + '.csv', index=False) # Save most recent round as a refreshed record
            pass
    
    return round_of_cleaned

# Function to count NaN values after the specified index, used in cleaned_to_X()
def count_nan_rows_after_index(group, morning_drops):
    index_to_check = group.reset_index(drop=True).index.get_loc(morning_drops) # Get the index equivalent to morning_drops
    sliced_group = group.reset_index(drop=True).iloc[index_to_check+1:] # Slice the group after the specified index
    nan_count = sliced_group.isna().sum() # Count the number of rows with NaN values
    return nan_count


def cleaned_to_X(dataframe, cleaned_indicators, temporal_granularity, select_intelligent=True, num_indicators=16, select_top=False, inverse_quantile = 0.4, assimilated=True):
    '''
    This function selects top performing featues and uses their optimal parameter values 
    to calculate and add them to the data for training/testing.
    
    Parameters
    ----------
    dataframe : TYPE pandas.DataFrame
        DESCRIPTION. All stock data avalailable for train/test set assignment; adhering to the format of the output from raw_data() in raw_data.py
    cleaned_indicators : TYPE, pandas.DataFrame
        DESCRIPTION. The output of clean_features(). A dataframe of the best performing version of each custom indicator function. 
    temporal_granularity : TYPE string
        DESCRIPTION. Duration of aggregate stock data bars used to generate the data.
    select_intelligent : TYPE, boolean
        DESCRIPTION. If True, optimized indicators are sorted into 'families' and the top performer from each family is chosen, so as to avoid collinearity.
    num_indicators : TYPE, integer.
        DESCRIPTION. Required if select_intelligent == True. This is the number of custom indicators that will be added to the X-matrix
    select_top : TYPE, integer
        DESCRIPTION. The default is False.
    inverse_quantile : TYPE, float (between 0 and 1)
        DESCRIPTION. Required if select_top == True. The performance cutoff for indicators to be used in model training, based on the primary objective score 
    assimilated : TYPE, boolean
        DESCRIPTION. Set to True if all the optimized indicators in cleaned_indicators share the same lag and growth_threshold value

    Returns
    -------
    Xy : TYPE pandas.DataFrame
        DESCRIPTION. Xy contains all the features to train the model, as well as the outcome variable and some additional columns for additional optimization and organization.
    selected_indicators : TYPE pandas.DataFrame
        DESCRIPTION. The selected custom indicator features and their corresponding optimal input parameters.
    nan_report : TYPE dictionary
        DESCRIPTION. A NaN record that helps to troubleshoot faulty indicators and other NaN-based issues

    '''
    if select_intelligent == True: ## Intelligently select top 16 indicators from different 'families'
        indicators_wfamily = cleaned_indicators.copy()
        indicators_wfamily['family'] = indicators_wfamily['indicator'].str.split('_').str[:3].str.join('_')# Group indicators into families based on the first 3 words
        filtered_indicators = indicators_wfamily.loc[indicators_wfamily.groupby('family')['scoring_metric'].idxmax()] # Select the indicator with the highest scoring_metric within each family
        filtered_indicators = filtered_indicators.drop(columns='family') # Drop the 'family' column from the selected_indicators DataFrame
        filtered_df = pd.DataFrame(filtered_indicators)# Create a new dataframe from the filtered indicators
        selected_indicators = filtered_df.sort_values(by='scoring_metric', ascending=False)[:num_indicators].reset_index(drop=True)
        
    if select_top == True: # Selecting top performing X% of the indicators. If inverse_quantile=0.4, the top performing 60% of indicators are chosen.
        selected_indicators = selected_indicators[selected_indicators['scoring_metric'] >= selected_indicators['scoring_metric'].quantile(inverse_quantile)].reset_index(drop=True)

    ## Identify lag and growth_threshold values to be used for outcome variable generation
    if assimilated == False:
        # Calculate weighted averages from unassimilated indicators
        average_lag, average_growth_threshold = np.average(selected_indicators[['lag', 'growth_threshold']], weights=selected_indicators['scoring_metric'], axis=0)
        average_lag = round(average_lag)
    elif assimilated == True:
        average_lag, average_growth_threshold = cleaned_indicators['lag'][0], cleaned_indicators['growth_threshold'][0]    
    
    # From selected_indicators, Add indicators to dataframe, column by column, day by day to avoid overlapping data across days
    running_df = dataframe.copy().reset_index(drop=True)
    if temporal_granularity in tb.daily_granularities:
        for i in range(len(selected_indicators)):
            function = getattr(fo, selected_indicators['indicator'][i])
            cleaned_params = ast.literal_eval(str(selected_indicators['cleaned_params'][i]))
            next_df = ind.daily_indicator(running_df, function, cleaned_params)
            running_df = next_df.copy()
    elif temporal_granularity in tb.overnight_granularities:
        for i in range(len(selected_indicators)):
            function = getattr(fo, selected_indicators['indicator'][i])
            cleaned_params = ast.literal_eval(str(selected_indicators['cleaned_params'][i]))
            next_df = function(running_df, **cleaned_params) #Generate indicator column
            running_df = next_df.copy()
    # NaN check to ensure features are functioning correctly & not producing unexpected NaN's    
    nan_counts = running_df.isna().sum()  # Count NaN values in each column
    num_days = len(running_df.day.unique()) 
    morning_drops = int(np.floor(max(nan_counts/num_days))) #for each indicator, average drops per day
    source_obvs_pday  = len(running_df) / num_days #for all days in data, how many observations per day
    obvs_after_morning_drop = source_obvs_pday - morning_drops #removing all morning drop NaNs, how many observations in the day
    
    # After a feature's morning 'warm up' period, how many NaN's are generated, using count_nan_rows_after_index() defined above
    grouped = running_df.groupby('day')
    nan_counts_after_morning = grouped.apply(count_nan_rows_after_index, morning_drops=morning_drops)
    # Identify and record indicators producing NaN's
    troublemaker_days = nan_counts_after_morning[nan_counts_after_morning.any(axis=1)].index.tolist()
    column_sums = nan_counts_after_morning.sum(axis=0)
    total_nans_after_morning = sum(column_sums)
    troublemaker_indicators = list(column_sums[column_sums > 0].index)

    # Establishing a record to check against later
    days_pre_na_drop = list(running_df.day.unique())
    
    # Calculate our outcome variable for model training based on lag and growth_threshold values
    if temporal_granularity in tb.daily_granularities:
        Xy = running_df.groupby('day').apply(lambda x: ind.single_day_growth_calc(x.reset_index(drop=True), lag=average_lag, growth_threshold=average_growth_threshold, temporal_granularity=temporal_granularity))
    elif (temporal_granularity in tb.overnight_granularities) and (temporal_granularity != 'one_day'):
        Xy = ind.growth_calc_on_close(running_df.reset_index(drop=True), lag=average_lag, growth_threshold=average_growth_threshold, temporal_granularity=temporal_granularity)
    elif (temporal_granularity == 'one_day'):
        Xy = ind.oneday_growth_calc(running_df.reset_index(drop=True), lag=average_lag, growth_threshold=average_growth_threshold, temporal_granularity=temporal_granularity)
         
    
    if temporal_granularity != 'one_day':    
        # Training model only on data that falls within trading hours    
        markets_open = datetime.time(9, 30)
        markets_close = datetime.time(15, 59)
        Xy = Xy[Xy['timemerge_dt'].dt.time.between(markets_open, markets_close)]
    
    # Preserve unscaled close values for later use (in intelligent trade optimization)
    Xy['close_actual'] = Xy['close']

    # Detrend 'close' and 'vwap' by differencing them; scale volume and transactions to be approximately [0.1]
    Xy = ind.diff_scale_source_data(Xy) #This function drops NaN's
    
    # In the rare case that excessive NaN's were produced and dropped in the previous line of code, record which days would have been dropped
    days_post_na_drop = list(Xy.day.unique())
    dropped_days = list(set(days_pre_na_drop) - set(days_post_na_drop))
    obvs_pday_model_data = len(Xy) / len(days_post_na_drop)

    nan_report = {'Observations_pDay_at_start': source_obvs_pday,
                  'Morning_Drops:': morning_drops,
                  'Observations_pDay_After_Morning_Drops': obvs_after_morning_drop,
                  'NaNs_Outside_Morning_Drops': total_nans_after_morning, 
                  'Troublemaker_Indicators': troublemaker_indicators,
                  'Troublemaker_Days': troublemaker_days,
                  'Observations_pDay_After_Lag_Drop': obvs_pday_model_data,
                  'Dropped_days_round2': dropped_days}
    
    return Xy, selected_indicators, nan_report


def LSTM_RNN(Xy, lag, temporal_granularity, sequence_length=5, num_layers=1, num_units=64, epochs=3, deployment_model=True, pseudo_test_days=0):
    '''
    This function creates a LSTM model with flexible architecture that the user can define.
    
    Parameters
    ----------
    Xy : TYPE pandas.DataFrame
        DESCRIPTION. Xy output of cleaned_to_X()
    lag : TYPE integer
       DESCRIPTION. The number of minutes over which the model is trained to predict growth. 
    temporal_granularity : TYPE string
        DESCRIPTION. Duration of aggregate stock data bars used to generate the data.
    sequence_length : TYPE, integer
        DESCRIPTION. Sequence length for LSTM model data pre-processing.
    num_layers : TYPE, integer
        DESCRIPTION. The number of layers in the LSTM model.
    num_units : TYPE, integer
        DESCRIPTION. The number of units in each LSTM layer.
    epochs : TYPE, integer
        DESCRIPTION. The number of epochs to train the neural net model with.
    deployment_model : TYPE, boolean
        DESCRIPTION. Whether the model will be used in live trading deployment; adjusting the train/test split accordingly
    pseudo_test_days : TYPE, integer
        DESCRIPTION. The number of final days in the dataset to be used as a test set, if desired.

    Returns
    -------
    output_model : TYPE keras.src.engine.functional.Functional
        DESCRIPTION. A neural net model trained on the defined training data set
    train_days : TYPE list of strings
        DESCRIPTION. A list of all the days from which data came to train the model
    test_days : TYPE list of strings
        DESCRIPTION. A list of all the days omitted from the model training process
    model_stats : TYPE dictionary
        DESCRIPTION. A dictionary outlining the number of layers and units used in model construction.
    '''
    #Train/Test split
    if deployment_model==False:
        cutoff = round(0.8 * Xy.day.nunique())
    elif deployment_model==True:
        cutoff = Xy.day.nunique() - pseudo_test_days
    
    # Removing the last 'lag' periods of the day so that the model is trained on lag periods allowing for full growth potential
    if temporal_granularity != 'one_day':
        Xy = Xy.groupby('day').apply(lambda group: group.iloc[:-lag]).reset_index(drop=True)
    
    train_days = Xy.day.unique()[:cutoff].tolist()
    test_days = Xy.day.unique()[cutoff:].tolist()
    train_Xy = Xy[Xy['day'].isin(train_days)].reset_index(drop=True)

    # Create X-matrix 'X_train' and the outcome variable vector 'y_train' 
    X_train = train_Xy.drop(columns = ['open', 'high', 'low', 'day', 'timemerge', 'timemerge_dt','lag_growth', 'growth_ind', 'close_actual'])
    y_train = train_Xy['growth_ind']
    
    # Function to create sequences from the data (mandatory for RNN's)
    def create_sequences(data, target, sequence_length):
        sequences, labels = [], []
        for i in range(len(data) - sequence_length + 1):
            sequence = data[i : i + sequence_length]
            label = target[i + sequence_length - 1]
            sequences.append(sequence)
            labels.append(label)
        return np.array(sequences), np.array(labels)
        
    # Create sequences and labels
    X_train_seqs, y_train_seqs = create_sequences(X_train, y_train, sequence_length)
    
    # # Define the input layer
    output_model = Sequential()
    for L in range(num_layers):
        # Add an LSTM layer with a defined number of units, input_shape should be (sequence_length, num_features)
        if num_layers == 1:
            output_model.add(LSTM(num_units, input_shape=(X_train_seqs.shape[1], X_train_seqs.shape[2])))
        elif num_layers > 1:
            if L == 0:
                # For the first layer, use the input_shape based on X_train_seqs
                output_model.add(LSTM(num_units, input_shape=(X_train_seqs.shape[1], X_train_seqs.shape[2]), return_sequences=True))
            elif L < num_layers-1:
                # For subsequent layers, dynamically adjust input_shape based on the previous layer's output shape
                output_model.add(LSTM(num_units, input_shape=(None, sequence_length, num_units), return_sequences=True))
            elif L == num_layers-1:
                # For the final layer, adjust return_sequence for proper output dimensions
                output_model.add(LSTM(num_units, input_shape=(None, sequence_length, num_units),  return_sequences=False))
        output_model.add(Dropout(0.5))
    # Add a dense layer for binary classification
    output_model.add(Dense(units=1, activation='sigmoid'))
    # Compile the model
    output_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Display model summary for confirmation
    output_model.summary()

    # Train model
    output_model.fit(X_train_seqs, y_train_seqs, epochs=epochs, verbose=1)
    
    # Save model stats for record
    model_stats = {'num_layers': num_layers,
                   'num_units': num_units}
    
    return output_model, train_days, test_days, model_stats

def standard_nn(Xy, lag, temporal_granularity, epochs=3, deployment_model=True, pseudo_test_days=0):
    '''
    This function creates a neural net model for price movement prediction.
    
    Parameters
    ----------
    Xy : TYPE pandas.DataFrame
        DESCRIPTION. Xy output of cleaned_to_X()
    lag : TYPE integer
       DESCRIPTION. The number of minutes over which the model is trained to predict growth. 
    epochs : TYPE, integer
        DESCRIPTION. The number of epochs to train the neural net model with.
    deployment_model : TYPE, boolean
        DESCRIPTION. Whether the model will be used in live trading deployment; adjusting the train/test split accordingly
    pseudo_test_days : TYPE, integer
        DESCRIPTION. The number of final days in the dataset to be used as a test set, if desired.

    Returns
    -------
    output_model : TYPE keras.src.engine.functional.Functional
        DESCRIPTION. A neural net model trained on the defined training data set
    train_days : TYPE list of strings
        DESCRIPTION. A list of all the days from which data came to train the model
    test_days : TYPE list of strings
        DESCRIPTION. A list of all the days omitted from the model training process
    '''
    #Train/Test split
    if deployment_model==False:
        cutoff = round(0.8 * Xy.day.nunique())
    elif deployment_model==True:
        cutoff = Xy.day.nunique() - pseudo_test_days
    
    # Removing the last 'lag' periods of the day so that the model is trained on lag periods allowing for full growth potential
    if temporal_granularity != 'one_day':
        Xy = Xy.groupby('day').apply(lambda group: group.iloc[:-lag]).reset_index(drop=True)
    
    train_days = Xy.day.unique()[:cutoff].tolist()
    test_days = Xy.day.unique()[cutoff:].tolist()
    train_Xy = Xy[Xy['day'].isin(train_days)].reset_index(drop=True)

    # Create X-matrix 'X_train' and the outcome variable vector 'y_train' 
    X_train = train_Xy.drop(columns = ['open', 'high', 'low', 'day', 'timemerge', 'timemerge_dt','lag_growth', 'growth_ind', 'close_actual'])
    y_train = train_Xy['growth_ind']

    ## Construct model architecture
    input_layer = Input(shape=(X_train.shape[1],), name='input_1') # Define the input layer
    dense_layer_1 = Dense(units=64, name='dense')(input_layer) # Define the first Dense layer with ReLU activation
    relu_layer = ReLU(name='re_lu')(dense_layer_1)
    output_layer = Dense(units=1, name='dense_1')(relu_layer)    # Define the second Dense layer for binary classification
    classification_head = Activation('sigmoid', name='classification_head_1')(output_layer)
    output_model = Model(inputs=input_layer, outputs=classification_head, name='model') # Combine the layers into a model
    output_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
    
    # Train model
    output_model.fit(X_train, y_train, epochs=epochs, verbose=1)
    
    return output_model, train_days, test_days


def AK(Xy, tkr, lag, temporal_granularity, max_trials=10, epochs=3, deployment_model=False, pseudo_test_days=0, graph_test_results=False):
    '''
    This function utilizes AutoKeras' AutoML framework to explore NN model architectures, exporting the optimal model.
    
    Parameters
    ----------
    Xy : TYPE pandas.DataFrame
        DESCRIPTION. Xy output of cleaned_to_X()
    tkr : TYPE string
        DESCRIPTION. The ticker symbol of the stock being modelled
    lag : TYPE integer
        DESCRIPTION. The number of minutes over which the model is trained to predict growth. 
    max_trials : TYPE, integer
        DESCRIPTION. The number of different neural net model architectures that should be tested
    epochs : TYPE, integer
        DESCRIPTION. The number of epochs that should be used for each model training process.
    deployment_model : TYPE, boolean
        DESCRIPTION. Whether the model will be used in live trading deployment; adjusting the train/test split accordingly
    pseudo_test_days : TYPE, integer
        DESCRIPTION. The number of final days in the dataset to be used as a test set, if desired.
    graph_test_results : TYPE, boolean
        DESCRIPTION. When true, the model will be deployed on the test set and model performance metrics will be graphed alongside
        the number of buy orders generated per day, based on varying buy order rounding thresholds for the prediction values.

    Returns
    -------
    best_fresh_AK : TYPE keras.src.engine.functional.Functional
        DESCRIPTION. A fully trained model constructed with optimal model architecture based on the AutoML optimization process
    train_days : TYPE list of strings
        DESCRIPTION. A list of all the days from which data came to train the model
    test_days : TYPE list of strings
        DESCRIPTION. A list of all the days omitted from the model training process

    '''
    #Train/Test split
    if deployment_model==False:
        cutoff = round(0.8 * Xy.day.nunique())
    elif deployment_model==True:
        cutoff = Xy.day.nunique() - pseudo_test_days
    
    # Removing the last 'lag' periods of the day so that the model is trained on lag periods allowing for full growth potential
    if temporal_granularity != 'one_day':
        Xy = Xy.groupby('day').apply(lambda group: group.iloc[:-lag]).reset_index(drop=True)
    
    train_days = Xy.day.unique()[:cutoff]
    test_days = Xy.day.unique()[cutoff:]
    train_Xy = Xy[Xy['day'].isin(train_days)].reset_index(drop=True)
    test_Xy = Xy[Xy['day'].isin(test_days)].reset_index(drop=True)

    # Create X and y
    X_train = train_Xy.drop(columns = ['open', 'high', 'low', 'day', 'timemerge', 'timemerge_dt','lag_growth', 'growth_ind', 'close_actual'])
    X_test = test_Xy.drop(columns = ['open', 'high', 'low', 'day', 'timemerge', 'timemerge_dt','lag_growth', 'growth_ind', 'close_actual'])
    y_train = train_Xy['growth_ind']
    y_test = test_Xy['growth_ind']
    
    #Define NN AutoKeras model type
    fresh_AK = ak.StructuredDataClassifier(max_trials=max_trials, overwrite=True)
    
    # Fit the model with pandas DataFrame and Series
    ak_start = time.time()
    fresh_AK.fit(X_train, y_train, epochs=epochs)
    ak_end = time.time()
    ak_duration = round((ak_end - ak_start)/60,3)
    print(f'AutoKeras completed in {ak_duration} minutes.')
    
    best_fresh_AK = fresh_AK.export_model()
    
    if graph_test_results == True:
        predictions_fresh = best_fresh_AK.predict(X_test)
        
        thresholds = np.arange(0.45, 0.8, 0.01)
        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []
        buys_per_day_list = []
        
        for threshold in thresholds:
            y_pred_cons = [1 if pred >= threshold else 0 for pred in predictions_fresh]
            accuracy = accuracy_score(y_test, y_pred_cons)
            f1 = f1_score(y_test, y_pred_cons)
            recall = recall_score(y_test, y_pred_cons)
            precision = precision_score(y_test, y_pred_cons)
            buys_per_day = sum(y_pred_cons) / 350
        
            accuracies.append(accuracy)
            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)
            buys_per_day_list.append(buys_per_day)
            
        # Plot the model performance metrics and buys per day
        fig, ax1 = plt.subplots()
        fig.set_size_inches(25, 11)
        # Plot accuracy, F1 score, precision, and recall
        ax1.plot(thresholds, accuracies, label='Accuracy', color='gray')
        ax1.plot(thresholds, f1_scores, label='F1 Score', color='brown')
        ax1.plot(thresholds, precisions, label='Precision', color='magenta')
        ax1.plot(thresholds, recalls, label='Recall', color='black')
        ax1.set_xlabel('Outcome Variable Rounding Threshold')
        ax1.set_ylabel('Metrics')
        ax1.legend(loc='lower left')
        ax1.grid(True)
        
        # Create a twin y-axis for buys per day on the right side
        ax2 = ax1.twinx()
        ax2.plot(thresholds, buys_per_day_list, color='cyan', label='Buys per Day')
        ax2.set_ylabel('Buys per Day', color='cyan')
        ax2.legend(loc='upper right')
        ax2.grid(True)
        
        # Title
        plt.title(f'{tkr} Fresh_AK w/ deap_cleaned top 16 intelligent')
        
        # Show the plot
        plt.show()

    return best_fresh_AK, train_days, test_days


def RF_gutcheck(X_train, X_test, y_train, y_test, graph_test_results=False):
    '''
    This function was used as a baseline gutcheck to compare performance against using a RandomForest model.
    
    Parameters
    ----------
    X_train : TYPE pandas.DataFrame
        DESCRIPTION. Random forest model training data, excluding the output variable
    X_test : TYPE pandas.DataFrame
        DESCRIPTION. Random forest model testing data, excluding the output variable
    y_train : TYPE pandas.Series
        DESCRIPTION. Output variable for random forest model training.
    y_test : TYPE pandas.Series
        DESCRIPTION. Output variable for random forest model testing.
    graph_test_results : TYPE, boolean
        DESCRIPTION. When true, the model will be deployed on the test set and model performance metrics will be graphed alongside
        the number of buy orders generated per day, based on varying buy order rounding thresholds for the prediction values.

    Returns
    -------
    rf_precision : TYPE float
        DESCRIPTION. The model's precision rate.

    '''
    rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_model.fit(X_train,y_train)
    probabilities = rf_model.predict_proba(X_test)
    rf_y_pred = rf_model.predict(X_test)
    rf_precision = precision_score(y_test, rf_y_pred)

    if graph_test_results == True:
        thresholds = np.arange(0.5, 0.8, 0.01)
        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []
        buys_per_day_list = []
    
        for threshold in thresholds:
            y_pred_cons = [1 if prob >= threshold else 0 for prob in probabilities[:, 1]]
            accuracy = accuracy_score(y_test, y_pred_cons)
            f1 = f1_score(y_test, y_pred_cons)
            recall = recall_score(y_test, y_pred_cons)
            precision = precision_score(y_test, y_pred_cons)
            buys_per_day = sum(y_pred_cons) / 350
    
            accuracies.append(accuracy)
            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)
            buys_per_day_list.append(buys_per_day)
    
        # Plot the model performance metrics and buys per day
        fig, ax1 = plt.subplots()
        fig.set_size_inches(18.5, 10.5)
        # Plot accuracy, F1 score, precision, and recall
        ax1.plot(thresholds, accuracies, label='Accuracy', color='gray')
        ax1.plot(thresholds, f1_scores, label='F1 Score', color='brown')
        ax1.plot(thresholds, precisions, label='Precision', color='magenta')
        ax1.plot(thresholds, recalls, label='Recall', color='black')
        ax1.set_xlabel('Outcome Variable Rounding Threshold')
        ax1.set_ylabel('Metrics')
        ax1.legend(loc='lower left')
        ax1.grid(True)
    
        # Create a twin y-axis for buys per day on the right side
        ax2 = ax1.twinx()
        ax2.plot(thresholds, buys_per_day_list, color='cyan', label='Buys per Day')
        ax2.set_ylabel('Buys per Day', color='cyan')
        ax2.legend(loc='upper right')
        ax2.grid(True)
    
        # Title
        plt.title('RF gut-check')
    
        # Show the plot
        plt.show()
    
    return rf_precision