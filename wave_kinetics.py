
from utils import PatientLoader

from processing.signal import detrend
from processing.signal import high_gamma
from processing.signal import low_component
from processing.signal import analytic_amp
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, LSTM, Bidirectional, Activation

import numpy as np
from sklearn.model_selection import train_test_split

def split_with_overlap(data,window_size, window_overlap):
    length = data.shape[0]
    result = []
    for i in range(0,length,window_size-window_overlap):
        if length >= i + window_size:
            result.append(data[i:i+window_size,:])
    return result


pl = PatientLoader(4, session=1)
rawseeg = pl.get_seeg()
data = detrend(rawseeg)
hg = high_gamma(data, pl.seeg_fs)
X_high_gamma = analytic_amp(hg)
lp = low_component(rawseeg, pl.seeg_fs)
X_low_component = analytic_amp(lp)


batch_size = 60
overlap_size = 59

print('Loading data...')

Y = np.load('./data/kh4_1_metrics.npy')
Y = Y.T
X = np.hstack((X_high_gamma, X_low_component))
X = X[:Y.shape[0], :]

num_features = X.shape[1]

X = split_with_overlap(X,batch_size,overlap_size)
Y = split_with_overlap(Y,batch_size,overlap_size)

#transforming in a single element
Y = [y[int(batch_size/2)] for y in Y]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

y_train = np.array(y_train)
y_test = np.array(y_test)
x_train = np.array(x_train)
x_test = np.array(x_test)

print(f'X train shape: {x_train.shape}')
print(f'Y train shape: {y_train.shape}')
print(f'X test shape: {x_test.shape}')
print(f'Y test shape: {y_test.shape}')

model = Sequential()
model.add(Input(shape=(batch_size,num_features)))
model.add(Bidirectional(LSTM(20,return_sequences=True)))
model.add(Bidirectional(LSTM(10,return_sequences=True)))
model.add(Bidirectional(LSTM(3,)))
model.add(Dropout(0.33))
model.add(Activation('softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),loss='huber_loss', metrics=['mean_absolute_error'])

<<<<<<< HEAD:wave_kinetics.py
#model.fit(x_train, y_train,
#          epochs=100,
#          validation_data=(x_test, y_test))


print(model.summary())
=======
model.fit(x_train, y_train,
          epochs=10,
          validation_data=(x_test, y_test))


print(model.summary())

prediction = model.predict(np.array(X))
np.save('kh4_1_gen_kinetics.npy',prediction)
>>>>>>> eduardo:models/wave_kinetics.py
