# stock_robot
## A flexible, custom-feature engine and automated trading program powered by genetic programming, machine learning and real-time data.

The Concept: Build a model which can reliably predict stock price movements 'lag' number of time periods in the future. The model outcome is binary, with an outcome of 1 if the stock price changes more than a given value, referred to as the 'growth_threshold', and represented as a percent. The model should be flexible in temporal granularity (1 minute, 5 minute, 1 hour, etc.)

i.e. with a lag value of 4, a growth_threshold of 0.01 and a temporal_granularity of fifteen_minute, the model would be trained to predict if there would be 1% growth over the next (4 * fifteen_minute)= 60 minutes.

The chronology of processes to accomplish this are as follows:

- raw_data.py will generate historical data at the chosen time interval (1-minute, 1-hour, etc.) for a given stock.
- feature_functions.py houses the custom feature functions which utilize the above stock data, as well as input parameters, to generate the custom feature values used for predictive modeling. It also houses dictionaries of input parameter value ranges that define the gridspace on which to search for optimal feature input parameter values
- DEAP_clean.py utilizes genetic programming to select optimal input parameter values for a given custom indicator and its supplied input parameter value dictionary. As it is written, it uses multi-objective selection criteria with ML scoring metrics based on the computationally ideal Logit model
- DEAP_ensemble.py optimizes a group of custom features cohesively, first allowing the 'lag' and 'growth_thresold' variable to be adjusted before taking a weighted mean amongst the top performing features to establish a fixed 'lag' and 'growth_thresold' value that all features are optimized on once again in a second round
- optimization_pipeline.py connects the data collection and model creation process, collectively creating a flexible optimization pipeline. It pulls all the functions present in feature_functions.py, as well as their parameter dictionaries, preparing them as needed to be optimized through the DEAP process. It additionally uses the optimized features to generate an X and y matrix which are used to train a neural net model. 
[...]
