# Dataset from kaggle competition. 
In this competition, you are challenged to develop a model capable of predicting the closing price movements for hundreds of Nasdaq listed stocks using data from the order book and the closing auction of the stock. Information from the auction can be used to adjust prices, assess supply and demand dynamics, and identify trading opportunities.

We were 3 people on this project. 
So we started to test 3 different models (Proof of concept) and the one that we saw most potentical in was lightgbm since it was pretty fast on large dataset as this, the other 2 that we tried was 2 different timeseries models but the it was lagged time data so lightgbm was the best choice.
Then we figured out we need to do feature engineering which we never done before, so we studied up on that while keep on testing hyperparameters. We made feature importance plot to see what features to engineer and so we did and the results became very good. 

We ended up on ranking 800 0,9% from the top and that is very good for a first project with new learning in it.

This is shortly explained but i enjoyed it very much.
