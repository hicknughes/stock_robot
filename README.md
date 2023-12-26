# stock_robot
## A flexible, custom-feature engine and automated trading program powered by genetic programming, machine learning and real-time data.

The Concept: Build a model which can reliably predict stock price movements 'lag' number of time periods in the future. The model outcome is binary, with an outcome of 1 if the stock price changes more than a given value, referred to as the 'growth_threshold', and represented as a percent. The model is flexible in temporal granularity (1 minute, 5 minute, 1 hour, etc.)

i.e. with a lag value of 4, a growth_threshold of 0.01 and a temporal_granularity of fifteen_minute, the model would be trained to predict if there would be 1% growth over the next (4 * fifteen_minute =) 60 minutes.

STAGE 1: Build a Predictive Model and Trading Strategy
- Historical data is used to optimize the parameters of custom technical trading indicators.
- Top performing custom indicators are added as features to the dataset for model training.
- Models can be assessed with a test set and performance estimated.
- Once satisfied, deployment models are re-trained on all data available and optimal trading parameters are calculated for live trading.
- The model and all optimal parameters as saved to a local directory.

STAGE 2: Deploy the Predictive Model for Live Trading
- Load the model and corresponding deployment data for live trading on the Interactive Brokers' TWS trading platform
- At the end of each trading day on which trades were executed, a stock-specific trade record is saved 

STAGE 1 Scripts:
- temporal_backend.py houses all the temporal_granularity-specific information used throughout the pipeline to differentiate the way data and workflows are handled.
- raw_data.py will generate historical data at the chosen time interval (1-minute, 1-hour, etc.) for a given stock.
- feature_functions.py houses the custom-feature functions which utilize the above stock data, as well as input parameters, to generate the custom-feature values used for predictive modeling. It also houses dictionaries of input parameter value ranges that define the gridspace on which to search for optimal input parameter values
- DEAP_clean.py utilizes genetic programming to select optimal input parameter values for a given custom indicator and its supplied input parameter value dictionary. As it is written, it uses multi-objective selection criteria with ML scoring metrics based on the computationally ideal Logit model
- DEAP_ensemble.py optimizes a group of custom features cohesively, first allowing the 'lag' and 'growth_thresold' variable to be adjusted before taking a weighted mean amongst the top performing features to establish a fixed 'lag' and 'growth_thresold' value that all features are optimized on once again in a second round
- optimization_pipeline.py connects the data collection and model creation process, collectively creating a flexible optimization pipeline. It pulls all the functions present in feature_functions.py, as well as their parameter dictionaries, preparing them as needed to be optimized through the DEAP process. It additionally uses the optimized features to generate an X and y matrix which are used to train a neural net model. 
- hold_strat.py optimizes the rounding threshold above which a prediction output will trigger a purchase order. It also finds the optimal hold duration, with the options being the number of time periods the model was trained on (referred to as 'lag'), a hold period that last 'lag' time periods and continues until negative growth is seen, and the final strategy is 'lag' + x number of additional hold periods as a fixed value.
- trigger.py assesses if there were historical price drops that can be measured and used as a way to increase performance by 'taking the finger off the trigger', thereby not trading when a stocks most recent prices are dropping significantly compared to generally recent prices.
- qualify_stock.py puts all of these pieces together: pulling the data, optimizing custom features, creating the X-matrix and output variable y-vector, building a predictive model, assessing its performance OOS if desired, or otherwise optimizing a trading strategy and finally saving the model and deployment data for later use.

STAGE 2 Scripts:
- polygon_realtime_functions.py houses a toolkit of functions that interact with Polygon.io's APIs to execute data pulls and check realtime pricing.
- IB_trade_functions.py houses a toolkit of functions that interact with Interactive Brokers' APIs to execute trades and pull account data. Some but not all are used in IB_live_trading.py. It relies on the ib_insync package.
- IB_live_trading.py leverages the ib_insync package to connect with Interactive Brokers and execute live trading functions in real time. It deploys the most recently trained model for a given ticker symbol, as well as the trading strategy loaded with the model. It is recommended to run this script 15 minutes after markets open.

