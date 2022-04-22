#%%
import tensorflow as tf
import os
import pandas as pd
import numpy as np

#%%
zip_path= tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True
)
csv_path, _ = os.path.splitext(zip_path)
# %%
df = pd.read_csv(csv_path)
# %%
df=df[5::6]
df
#%%
df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
df[:26]
# %%
temp = df['T (degC)']
temp.plot()
# %%
def df_to_X_y(df, window_size=5) :
    df_as_np= df.to_numpy()
    X=[]
    y=[]
    for i in range(len(df_as_np)-window_size) : 
        row = [[a] for a in df_as_np[i:i+5]]
        X.append(row) 
        label = df_as_np[i+5]
        y.append(label)
    return np.array(X), np.array(y)

#%%
WINDOW_SIZE=5
X, y = df_to_X_y(temp, WINDOW_SIZE)
X.shape, y.shape    
# %%
X_train, y_train = X[:60000], y[:60000]
X_val, y_val = X[60000:65000], y[60000:65000]
X_test, y_test = X[65000:], y[65000:]
X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape
# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
#%%
model1=Sequential()
model1.add(InputLayer((5,1)))
model1.add(LSTM(64))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))

model1.summary()
#%%
cp=ModelCheckpoint('model1/', save_best_only=True)
model1.compile(optimizer=Adam(learning_rate=0.0001,), loss=MeanSquaredError(), metrics=[RootMeanSquaredError()])

#%%
model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=cp)
# %%
from tensorflow.keras.models import load_model
model1=load_model('model1/')
# %%
train_predictions = model1.predict(X_train).flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train})
train_results
# %%
import matplotlib.pyplot as plt
plt.plot(train_results['Train Predictions'][50:80])
plt.plot(train_results['Actuals'][50:80])
# %%
from sklearn.metrics import mean_squared_error as mse
def plot_prediction(model, X, y, start=0, end=100):
    predictions = model.predict(X).flatten()
    results= pd.DataFrame(data={'Model Predictions':predictions, 'Actual Values':y})
    plt.plot(results['Model Predictions'][start:end])
    plt.plot(results['Actual Values'][start:end])
    return results, mse(y, predictions)

plot_prediction(model1, X_test, y_test)
# %%
temp_df = pd.DataFrame({'Temperature':temp})
temp_df['Seconds'] = temp_df.index.map(pd.Timestamp.timestamp)
temp_df
# %%
day=60*60*24
year=364.2425*day
temp_df['Day sin']=np.sin(temp_df['Seconds']*(2*np.pi/day))
temp_df['Day cos']=np.cos(temp_df['Seconds']* (2*np.pi/day))
temp_df['Year sin']=np.sin(temp_df['Seconds']*(2*np.pi/year))
temp_df['Year sin']=np.cos(temp_df['Seconds']*(2*np.pi/year))
temp_df
# %%
temp_df = temp_df.drop('Seconds', axis=1)
# %%
def df_to_X_y2(df, window_size=6) :
    df_as_np= df.to_numpy()
    X=[]
    y=[]
    for i in range(len(df_as_np)-window_size) : 
        row = [a for a in df_as_np[i:i+window_size]]
        X.append(row) 
        label = df_as_np[i+window_size][0]
        y.append(label)
    return np.array(X), np.array(y)

X2, y2 = df_to_X_y2(temp_df)
X2.shape, y2.shape
# %%
X2_train, y2_train = X2[:60000], y2[:60000]
X2_val, y2_val = X2[60000:65000], y2[60000:65000]
X2_test, y2_test = X2[65000:], y2[65000:]
X2_train.shape, y2_train.shape, X2_val.shape, y2_val.shape, X2_test.shape, y2_test.shape

# %%
temp_training_mean=np.mean(X2_train[:, :, 0])
temp_training_std=np.std(X2_train[:, :, 0])

def preprocess(X):
    X[:,:,0]=(X[:,:,0] - temp_training_mean)/temp_training_std
    return X

preprocess(X2_train)
preprocess(X_test)
preprocess(X_val)
# %%
