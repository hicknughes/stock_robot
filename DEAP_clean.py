#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The culminating function, DEAP_clean(), flexibly optimizes custom indicator parameter values.
Genetic programming is employed to intelligently search through computationally prohibitive grid spaces.
Multiple objectives, weighted accordingly, drive optimization.
Specific parameters can be left out, in this case, 'lag' and 'growth_threshold'.
For their efficiency, logit models are used. F1 and precision rates are maximized while favoring small 'lag' and large 'growth_threshold' values.
""" 

import random
from deap import base, creator, tools
import Indicator_Building_Blocks as ind
import numpy as np
import pandas as pd
import ast
from datetime import datetime as dt
from datetime import time
import datetime
import time
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, precision_score
import time #Used in testing
from colorama import Fore, Style
#The next 4 lines suppress the Precision=0 warning
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# The initialize function creates an Individual with random parameter values within the given parameter_ranges supplied
def initialize_individual(parameter_ranges, fixed_lag_growth=False, given_lag=12, given_growth_thresh=0.001):
    random_parameters = creator.Individual()
    if fixed_lag_growth==True: #If lag/growth_threshold have been defined, exclude them from the random value initialization
        parameters_to_randomize = {
            k: v for k, v in parameter_ranges.items()
            if k not in ["lag", "growth_threshold"]
        }
        random_parameters['lag'] = given_lag
        random_parameters['growth_threshold'] = given_growth_thresh
    elif fixed_lag_growth==False:
        parameters_to_randomize = parameter_ranges
    # Generate random parameter values within the range defined in parameter_ranges
    for parameter, value_range in parameters_to_randomize.items():
        min_value = value_range[0]
        max_value = value_range[1]
        if isinstance(min_value, int) and isinstance(max_value, int):
            random_parameters[parameter] = random.randint(min_value, max_value)
        elif isinstance(min_value, float) and isinstance(max_value, float):
            random_parameters[parameter] = random.uniform(min_value, max_value)
        else:
            type1 = type(min_value)
            type2 = type(max_value)
            raise ValueError(f"'{parameter}' minimum value ({min_value}) is of type {type1} and maximum value ({max_value}) is of type {type2}. Must be either both integers or both floats.")
    return random_parameters

# Create a custom crossover (mating) operator
def crossover_individuals(ind1, ind2):
    offspring = creator.Individual()
    for key in ind1:
        if isinstance(ind1[key], int):
            offspring[key] = random.choice([ind1[key], ind2[key]])
        elif isinstance(ind1[key], float):
            offspring[key] = random.uniform(ind1[key], ind2[key])
        else:
            offspring[key] = ind1[key]  # Preserve non-numeric values as-is
    return offspring

# Create a custom mutation operator, mutating with a given frequency and variability, respecting fixed lag and growth_rate as needed
def mutate_individual(individual, fixed_lag_growth=False, mutation_probability=0.30, float_mutation_variance=0.4):
    mutated = creator.Individual()
    if fixed_lag_growth==True:#If lag/growth_threshold have been defined, exclude them from the mutation
        keys_to_mutate = list(individual.keys())
        keys_to_mutate.remove('lag')
        keys_to_mutate.remove('growth_threshold')
        mutated['lag'] = individual['lag']
        mutated['growth_threshold'] = individual['growth_threshold']
    elif fixed_lag_growth==False:
        keys_to_mutate = list(individual.keys())
    
    for key in keys_to_mutate:
        if random.uniform(0,1) > mutation_probability:
            mutated[key] = individual[key]
        else: 
            if isinstance(individual[key], int):
                mutated[key] = individual[key] + random.choice([-2, -1, 1, 2])
            elif isinstance(individual[key], float):
                mutated[key] = individual[key] + random.uniform(-(individual[key]*float_mutation_variance), individual[key]*float_mutation_variance)
            else:
                mutated[key] = individual[key]  # Preserve non-numeric values as-is
    return mutated


# Define the evaluation function which determines the Fitness of a particular Individual

def evaluate_features(dataframe, indicator_function, parameters, temporal_granularity):
    #Isolate the indicator-specific parameters used to generate a custom indicator 'Individual'
    remaining_params = {
        k: v for k, v in parameters.items()
        if k not in ["lag", "growth_threshold"]
    }
    
    if temporal_granularity in {'one_minute', 'three_minute', 'five_minute'}:
        df_wgrowth = ind.daily_growth_calc(dataframe, parameters['lag'], parameters['growth_threshold']) # Calculate outcome variable
        non_indicator_columns = df_wgrowth.columns #identify all columns before indicator calculation 
        param_it_df = ind.daily_indicator(df_wgrowth, indicator_function, remaining_params) #Generate indicator column
    elif temporal_granularity in {'fifteen_minute', 'one_hour', 'one_day'}:
        df_wgrowth = ind.growth_calc(dataframe, parameters['lag'], parameters['growth_threshold']) # Calculate outcome variable
        non_indicator_columns = df_wgrowth.columns #identify all columns before indicator calculation 
        param_it_df = indicator_function(df_wgrowth, **remaining_params) #Generate indicator column
    
    # Select only data during trading hours to train on if temporal_granularity != one_day
    if temporal_granularity != 'one_day':    
        markets_open = datetime.time(9, 30)
        markets_close = datetime.time(16, 0)
        param_it_df = param_it_df[param_it_df['timemerge_dt'].dt.time.between(markets_open, markets_close)]
    param_it_df = param_it_df.dropna().reset_index(drop=True) # No NaN's allowed
    
    # separate the target variable and feature of interest
    y = param_it_df['growth_ind']
    X = param_it_df.drop(columns=non_indicator_columns) # Eliminate all columns other than the feature of interest
    
    # Train/Test split for time series data
    cv_splits = 3
    gap = parameters['lag']
    tscv = TimeSeriesSplit(n_splits=cv_splits, gap=gap)

    # Initialize the lists to record the precision and F1 scores for each split
    f1_scores = []
    precision_scores = []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # perform logistic regression
        model = LogisticRegressionCV(class_weight = 'balanced', scoring='f1', solver='liblinear', random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
      
        f1_onsplit = f1_score(y_test, y_pred)
        f1_scores.append(f1_onsplit)
        precision_onsplit = precision_score(y_test, y_pred, zero_division=0)
        precision_scores.append(precision_onsplit)

    # Record this iterations values
    return np.mean(precision_scores), np.mean(f1_scores), parameters['growth_threshold'], parameters['lag'],

# Create custom classes for DEAP on which to optimize
creator.create("FitnessMax", base.Fitness, weights=(0.3, 1.0, 1.0, -0.2)) # The fitness attribute allows for multi-objective optimization with coordinating weights per objective
creator.create("Individual", dict, fitness=creator.FitnessMax) # The individual class is a dictionary with the attribute fitness
primary_objective = 1 #Index of optimization objectives that is the primary objective to select apex individual by
evaluate_features_output = {'0': 'Precision', # Objectives to index in generational printout statement
                            '1': 'F1_Score',
                            '2': 'Growth_Threshold',
                            '3': 'Lag'}

# If needing to redefine clases, it is best to delete existing classes with the lines below
# del creator.FitnessMax
# del creator.Individual

def DEAP_clean(tkr, dataframe, indicator_function, parameter_ranges, temporal_granularity, population_scalar=23, max_generations=1, fix_lag_growth=False, assigned_lag=12, assigned_growth_thresh=.001,use_apex=True, use_apex_genesis=False):
    '''
    Parameters
    ----------
    tkr : TYPE string
        DESCRIPTION. The ticker symbol of the stock of interest for the model
    dataframe : TYPE pandas.DataFrame
        DESCRIPTION. A dataframe of stock data produced by raw_data() in raw_data.py
    indicator_function : TYPE function
        DESCRIPTION. This is a feature generating function, taking in a dataframe of data and adding one column of custom feature values 
    parameter_ranges : TYPE dictionary
        DESCRIPTION. A dictionary with each feature parameter as keys and the parameter range as the values, saved as a list as such: [minimum_value, maximum_value]
    temporal_granularity : TYPE string
        DESCRIPTION. Duration of aggregate stock data bars used to generate the data. 
    population_scalar : TYPE, integer
        DESCRIPTION. The default is 23. For each feature parameter to be optimized, this many individuals will be made per generation.
    max_generations : TYPE, integer
        DESCRIPTION. The default is 1. The number of generations of features to test and evolve, including a generation 0.
    fix_lag_growth : TYPE, boolean
        DESCRIPTION. The default is False. If True, lag and growth_threshold will not be optimized and their fixed values must be supplied
    assigned_lag : TYPE, integer
        DESCRIPTION. The default is 12 as a placeholder. The fixed value for a lag time will not be changed in the DEAP process.
    assigned_growth_thresh : TYPE, float
        DESCRIPTION. The default is .001 as a placeholder. The fixed value for growth_threshold will not be changed in the DEAP process.
    use_apex : TYPE, boolean
        DESCRIPTION. When True, and if a previous record of high performing features is available, their top performing 'Individuals' will be injected into generation 0.
    use_apex_genesis : TYPE boolean
        DESCRIPTION. When True, apex 'Individuals' with the same temporal_granularity (irregardless of the stock) will be used. Ideal when training a model on a new stock.

    Returns
    -------
    final_gen_cleaned : TYPE pandas.DataFrame
        DESCRIPTION. A single-row dataframe with columns outlining the indicator, the precision of the logit model using optimal parameters, the lag and growth threshold values used, and a dictionary of optimized parameter values.

    '''
    assert dataframe.index.equals(pd.RangeIndex(len(dataframe))), "Please reset input dataframe's index."

    ## Create DEAP's toolbox
    toolbox = base.Toolbox()
    toolbox.register("initialize", initialize_individual) # Define the initialiaze function which creates a random individuals
    toolbox.register("evaluate", evaluate_features) # Define the evaluation function
    toolbox.register("mate", crossover_individuals) # Mating two Individuals
    toolbox.register("mutate", mutate_individual)  # Mutating Individuals
    toolbox.register("select_rankers", tools.selNSGA2) # Selecting top ranking individuals through Non-dominated Sorting Genetic Algorithm II
    toolbox.register("select_elites", tools.selBest) # Selecting top ranking individuals by Fitness Values alone
    
    # Define population_size and max_minutes based on parameter_ranges
    num_params = len(parameter_ranges)
    population_size = round(num_params * population_scalar) # The number of parameters dictates the size of the population
    
    #Offspring generations' composition
    num_apex = int(0.3 * population_size) # Number of apex individuals from the past to be incorporated in the first generation only
    num_ranked = int(0.3 * population_size) # Top ranking individuals to be selected for the gene pool
    num_elites = 3 # The number of top performers to be passed on to the next generation as-is
    num_wildcards = int(0.3 * population_size) # Number of random individuals to be generated and evaluated each generation
    growth_threshold_minimum = 0.00045 # Establishing a floor for growth_threshold parameter mutations
    
    start_time = time.time() # Optimization duration timekeeping
    
    if use_apex == True:
        #Load Apex individuals from previous optimizations
        try:# In case record doesn't exist...
            apex_df = pd.read_csv(tkr + '_' + temporal_granularity + '_apex_record.csv') #load record
            if not apex_df.empty: # If the record exists and is not empty...
                apex_df['cleaned_params'] = apex_df['cleaned_params'].apply(ast.literal_eval) # re-assignging the params column to be dictionaries
                
                # Select only individuals with lag and growth_threshold values within the currently sought ranges defined in parameter_ranges
                fitted_apex = apex_df[
                    (apex_df['lag'] >= parameter_ranges['lag'][0]) & (apex_df['lag'] <= parameter_ranges['lag'][1]) &
                    (apex_df['growth_threshold'] >= parameter_ranges['growth_threshold'][0]) & (apex_df['growth_threshold'] <= parameter_ranges['growth_threshold'][1])
                    ]
                
                # Adding fixed or dynamic lag/growth values to dictionaries
                if fix_lag_growth == False: #maintainig apex lag/growth
                    fitted_apex['cleaned_params'] = fitted_apex.apply(lambda row: {**row['cleaned_params'], 'lag': row['lag'], 'growth_threshold': row['growth_threshold']}, axis=1)
                elif fix_lag_growth == True: #Assigning apex lag/growth for 'assimilated' optimization
                    fitted_apex['cleaned_params'] = fitted_apex.apply(lambda row: {**row['cleaned_params'], 'lag': assigned_lag, 'growth_threshold': assigned_growth_thresh}, axis=1)

                # Select top performing apex individuals of the indicator function being optimized
                sliced_apex_df = fitted_apex[fitted_apex['indicator'] == str(indicator_function).strip().split()[1]].copy().sort_values(by='scoring_metric', ascending=False)[0:num_apex]
                apex_individuals = [creator.Individual(**d) for d in list(sliced_apex_df['cleaned_params'])] #convert dictionaries to 'Individual' class
            
            elif apex_df.empty: # If the record exists but is an empty dataframe...
                num_apex = 0
                apex_individuals = []    
        
        except:
            num_apex = 0
            apex_individuals = []    
    
    elif use_apex == False:
       num_apex = 0
       apex_individuals = []    
   
    if use_apex_genesis == True: # A first time round for a given stock at a given temporal_granularity
        num_apex = int(0.6 * population_size) # Increasing the use of past optimal values to establish new record
        #Load Apex Genesis from previous optimizations at the same temporal_granularity
        try:# In case record doesn't exist...
            apex_genesis_df = pd.read_csv('apex_genesis_' + temporal_granularity + '.csv') #load genesis record
            if not apex_genesis_df.empty:# If the record exists and is not empty...
                apex_genesis_df['cleaned_params'] = apex_genesis_df['cleaned_params'].apply(ast.literal_eval) # re-assignging the params column to be dictionaries    
                
                # Select only individuals with lag and growth_threshold values within the currently sought ranges defined in parameter_ranges
                fitted_apex = apex_genesis_df[
                    (apex_genesis_df['lag'] >= parameter_ranges['lag'][0]) & (apex_genesis_df['lag'] <= parameter_ranges['lag'][1]) &
                    (apex_genesis_df['growth_threshold'] >= parameter_ranges['growth_threshold'][0]) & (apex_genesis_df['growth_threshold'] <= parameter_ranges['growth_threshold'][1])
                    ]
                
                # Adding fixed or dynamic lag/growth values to dictionaries
                if fix_lag_growth == False: #maintainig apex lag/growth
                    fitted_apex['cleaned_params'] = fitted_apex.apply(lambda row: {**row['cleaned_params'], 'lag': row['lag'], 'growth_threshold': row['growth_threshold']}, axis=1)
                elif fix_lag_growth == True: #Assigning apex lag/growth for 'assimilated' optimization
                    fitted_apex['cleaned_params'] = fitted_apex.apply(lambda row: {**row['cleaned_params'], 'lag': assigned_lag, 'growth_threshold': assigned_growth_thresh}, axis=1)

                # Select top performing apex individuals of the indicator function being optimized
                sliced_apex_df = fitted_apex[fitted_apex['indicator'] == str(indicator_function).strip().split()[1]].copy().sort_values(by='scoring_metric', ascending=False)[0:num_apex]
                apex_individuals = [creator.Individual(**d) for d in list(sliced_apex_df['cleaned_params'])] #convert dictionaries to 'Individual' class
            
            elif apex_genesis_df.empty: # If the record exists but is an empty dataframe...
                num_apex = 0
                apex_individuals = []    
        except:
            num_apex = 0
            apex_individuals = []    
    elif use_apex == False:
       num_apex = 0
       apex_individuals = [] 
   
    # Create the initial population
    population = [toolbox.initialize(parameter_ranges, fixed_lag_growth=fix_lag_growth, given_lag=assigned_lag, given_growth_thresh=assigned_growth_thresh) for _ in range(population_size - num_apex)]
    population = population + apex_individuals

    # Evaluate the initial population
    fitness_values = []
    for individual in population:
        individual = population[0]
        try:
            fitness = toolbox.evaluate(dataframe, indicator_function, individual, temporal_granularity)
            fitness_values.append(fitness)
        except KeyboardInterrupt:
            break
        except Exception as e:
            fitness_values.append((0, 0, 0, 0))
            print(f'{Fore.RED}Indicator calculation error for {str(indicator_function).strip().split()[1]} with individual: {individual}{Style.RESET_ALL}')
            print(f'Error message: {str(e)}')
    for j in range(len(population)):
        population[j].fitness.values = fitness_values[j]

    # Get records from Gen 0
    gen_zero_best = max(population, key=lambda x: x.fitness.values[primary_objective])
    gen0_primary_objective_score = gen_zero_best.fitness.values[primary_objective]
    
    generation = 0
    elapsed_minutes = round((time.time() - start_time) / 60, 3)
    print(f"{Fore.CYAN}{str(indicator_function).strip().split()[1]} Generation {generation}{Style.RESET_ALL} Best Parameters = {gen_zero_best} {Fore.YELLOW}Primary Objective ({evaluate_features_output[str(primary_objective)]}) Score: {round(gen0_primary_objective_score, 3)}.{Fore.MAGENTA}Elapsed time: {elapsed_minutes} minutes.{Style.RESET_ALL}")

    while generation < max_generations: #Starting the evolutionary process stepping off from generation 0
        generation += 1
        # Select the next generation
        wildcards = [toolbox.initialize(parameter_ranges, fixed_lag_growth=fix_lag_growth, given_lag=assigned_lag, given_growth_thresh=assigned_growth_thresh) for _ in range(num_wildcards)]
        gene_pool = toolbox.select_rankers(population, num_ranked)
        elites = toolbox.select_elites(population, num_elites)
        
        # Apply crossover between individuals with non-zero F1 scores, then mutate
        offspring = []
        for o in range(len(population)-num_elites-num_wildcards):
            mommy = random.choice([individual for individual in gene_pool if individual.fitness.values[primary_objective] > 0])
            daddy = random.choice([individual for individual in gene_pool if individual.fitness.values[primary_objective] > 0])
            offspring.append(toolbox.mate(mommy, daddy))
        
        for child in offspring:
            toolbox.mutate(child, fixed_lag_growth=fix_lag_growth)
            while child['growth_threshold'] <= growth_threshold_minimum: #Ensure growth threshold is sufficient
                child = toolbox.mutate(child, fixed_lag_growth=fix_lag_growth) #else mutate again
        
        offspring = offspring + elites + wildcards
        
        # Evaluate the offspring, ensuring that errors from mutants are handled effectively
        fitness_values = []
        for individual in offspring:
            try:
                fitness_values.append(toolbox.evaluate(dataframe, indicator_function, individual, temporal_granularity))
            except KeyboardInterrupt:
                break
            except:
                fitness_values.append(tuple((0,0,0,0)))
                print(f'{Fore.RED}Indicator calculation error for {str(indicator_function).strip().split()[1]} with individual: {individual}{Style.RESET_ALL}')
                continue
        for k in range(len(population)):
            offspring[k].fitness.values = fitness_values[k]

        # Replace the population with the offspring
        population[:] = offspring

        # Print the best fitness value of each generation
        best_params = max(population, key=lambda x: x.fitness.values[primary_objective])
        genx_primary_objective_score = best_params.fitness.values[primary_objective]
        
        # Add record of generation
        ind_name = (str(indicator_function).strip().split()[1])
        gen_x_ind_params = dict(best_params)
        if fix_lag_growth == True:
            keys_to_remove = ['lag', 'growth_threshold']
            gen_x_ind_params = {key: value for key, value in gen_x_ind_params.items() if key not in keys_to_remove}
        
        last_time = elapsed_minutes
        elapsed_minutes = round((time.time() - start_time) / 60, 3)
        gen_time = round(elapsed_minutes - last_time, 3)
        print(f"{Fore.CYAN}{str(indicator_function).strip().split()[1]} Generation {generation}{Style.RESET_ALL} Best Parameters = {best_params} {Fore.YELLOW}Primary Objective ({evaluate_features_output[str(primary_objective)]}) Score: {round(genx_primary_objective_score, 3)}.{Fore.MAGENTA}Total Time: {elapsed_minutes} minutes. This generation: {gen_time} minutes.{Style.RESET_ALL}") 
        
    # Retrieve top performing parameter value combinations and their scores, insert into 'lineage' DF
    data = [[ind_name, genx_primary_objective_score, best_params['lag'], best_params['growth_threshold'], gen_x_ind_params]]
    columns = ["indicator", "scoring_metric", "lag", "growth_threshold", "cleaned_params"]
    final_gen_cleaned = pd.DataFrame(data, columns=columns)
    
    return final_gen_cleaned

### TEST ###
# tkr = 'NVDA'
# temporal_granularity = 'one_minute'
# data_small = dataframe[dataframe['day'].isin(dataframe.day.unique()[-50:].tolist())].reset_index(drop=True)

# deployment_model=True
# pseudo_test_days=0
# epochs=3
# use_apex=True
# max_generations=1
# population_scalar = 3
# fix_lag_growth = False
# assigned_lag=12
# assigned_growth_thresh=.001

# import feature_origin as fo
# indicator_function = getattr(fo,'macd')
# parameter_ranges = getattr(fo,'macd_deap_params')
# parameter_ranges.update(fo.lag_growth_thresh_tiers[temporal_granularity]) # Add temporally specific lag/growth_threshold value ranges

# DEAP_clean(tkr, data_big, test_func, test_dict, temporal_granularity, population_scalar=3, max_generations=1, fix_lag_growth=False, use_apex=True)

