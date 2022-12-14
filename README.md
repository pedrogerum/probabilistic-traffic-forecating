# probabilistic_traffic_forecating
In this folder, we host the code used to train and test the deep learning models for traffic occupancy forecasts. We implement a customized version of MQRNN that ensures monotonicity and compute probabilistic metrics on the forecasts to verify that the probabilistic forecasts are accurate. We use as example occupancy data from Chicago between 2008 and 2014. Training data comprises the ones between 2008 and 2011, and testing data includes 2012 and after.

While we use a lot of the common libraries for data manipulation and machine learning, we have a customized version of the GluonTS library that we altered to ensure the forecasted distributions are monotonic.
This library is located under GluonTS-local. When you download it, activate your environment, then go to the folder and "pip install ."

The files train_chicago.sh and train_milwaukee.sh contain the bash commands to train and test the models.
The main code is under quantile_forecast.py

You can see a simple implementation on how to run the model in the jupyter notebook file: "Quantile_forecasting_notebook_github.ipynb"
