import keras 
import pandas as pd
import numpy as np
import tensorflow as tf

#load files

abcb4_timedata = pd.read_csv("/content/ABCB4 Dados Históricos.csv", sep=",", index_col=None)
# print(abcb4_timedata)

#preprocessing
abcb4_timedata['Data'] = pd.to_datetime(abcb4_timedata['Data'], format="%d.%m.%Y")

abcb4_timedata['Último'] = [float(record.replace(",", ".")) for record in abcb4_timedata['Último']]
abcb4_timedata['Abertura'] = [float(record.replace(",", ".")) for record in abcb4_timedata['Abertura']]
abcb4_timedata['Máxima'] = [float(record.replace(",", ".")) for record in abcb4_timedata['Máxima']]
abcb4_timedata['Mínima'] = [float(record.replace(",", ".")) for record in abcb4_timedata['Mínima']]

abcb4_timedata['Vol.'] = [float(record.replace("K", "0").replace("M", "0000").replace(",", "")) for record in abcb4_timedata['Vol.']]

abcb4_timedata['Var%'] = [float(record.removesuffix("%").replace(",", "."))/100 for record in abcb4_timedata['Var%']]

df = pd.DataFrame(abcb4_timedata, index=None)
df = df.drop(columns='Data')
print(df)

#Train, Test and Validation groups

from sklearn.model_selection import train_test_split

label_list_train_test = list(range(df.shape[0]))
data_train, data_test, label_train, label_test = train_test_split(df, label_list_train_test, test_size=0.25, random_state=42, shuffle=False)

# print(data_train, "\n" ,data_test)

data_test, data_validation, label_test, label_validation = train_test_split(data_test, label_test, test_size=0.33, random_state=42, shuffle=False)

print(f"data train: \n{data_train} \ndata test\n: {data_test} \ndata validation\n: {data_validation}")

#Model 

import numpy as np
from keras import layers
from keras.models import Sequential
from keras.layers import LSTM
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

num_timesteps = 1

#target data
data_train_target = (data_train.loc[:,"Último"]).values
data_test_target = (data_test.loc[:,"Último"]).values
data_validation_target = (data_validation.loc[:,"Último"]).values

#reshape data
data_train_in = data_train.values
data_train_in = data_train_in.reshape(data_train_in.shape[0], num_timesteps, data_train_in.shape[1])

data_test_in = data_test.values
data_test_in = data_test_in.reshape(data_test_in.shape[0], num_timesteps, data_test_in.shape[1])

data_validation_in = data_validation.values
data_validation_in = data_validation_in.reshape(data_validation_in.shape[0], num_timesteps, data_validation_in.shape[1])

model = keras.Sequential([
    layers.LSTM(64, input_shape=(num_timesteps, data_train_in.shape[2])), #timesteps, features
    layers.Dense(1)
])

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
history = model.fit(data_train_in, data_train_target, epochs=50, batch_size=32, validation_data=(data_validation_in, data_validation_target))
print(history.history)
scores = model.evaluate(data_test_in, data_test_target)
print(f'scores:\nloss: {scores[0]}\nmean abs error: {scores[1]}')

predictions = model.predict(data_test_in)
print(predictions)
print(predictions.shape)

mae = round(mean_absolute_error(data_test_target, predictions) ,4)
mape = round(np.mean(np.abs((data_test_target - predictions) / data_test_target)) * 100, 4)
mse = round(mean_squared_error(data_test_target, predictions), 4)
rmse = round(mse**(0.5), 4)
rmspe = round((np.sqrt(np.mean(np.square((data_test_target - predictions) / data_test_target)))) * 100, 4)
r2 = round(r2_score(data_test_target, predictions), 4)
print(f"mae: {mae} \nmape: {mape} \nmse: {mse}, \nrmse: {rmse} \nr2: {r2}")