# probabilistic_traffic_forecating
In this folder, we host the code used to train and test the models for traffic occupancy forecasts.

While we use a lot of the common libraries for data manipulation and machine learning, we have a customized version of the GluonTS library to ensure monotonicity.
This library is located under GluonTS-local. When you download it, activate your environment, then go to the folder and "pip install ."

The files train_chicago.sh and train_milwaukee.sh contain the bash commands to train and test the models.
The main code is under quantile_forecast.py
