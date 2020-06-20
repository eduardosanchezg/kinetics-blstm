from utils import PatientLoader

from processing.signal import detrend
from processing.signal import high_gamma
from processing.signal import low_component
from processing.signal import analytic_amp
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, Activation

import numpy as np
from sklearn.model_selection import train_test_split

def split_with_overlap(data,window_size, window_overlap):
    length = data.shape[0]
    result = []
    for i in range(0,length,window_size-window_overlap):
        if length >= i + window_size:
            result.append(data[i:i+window_size,:])
    return result


pl = PatientLoader(1, session=1)
rawseeg = pl.get_seeg()
data = detrend(rawseeg)
hg = high_gamma(data, pl.seeg_fs)
X_high_gamma = analytic_amp(hg)
lp = low_component(rawseeg, pl.seeg_fs)
X_low_component = analytic_amp(lp)


# cut texts after this number of words
# (among top max_features most common words)

batch_size = 1000
overlap_size = 200

print('Loading data...')

Y = np.load('/home/eduardo/master/mrp2/kinetics/kh4_1_metrics.npy')
Y = Y.T
X = np.hstack((X_high_gamma, X_low_component))
X = X[:Y.shape[0], :]

num_features = X.shape[1]

X = split_with_overlap(X,batch_size,overlap_size)
Y = split_with_overlap(Y,batch_size,overlap_size)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

y_train = np.array(y_train)
y_test = np.array(y_test)
x_train = np.array(x_train)
x_test = np.array(x_test)

model = Sequential()
model.add(Dense(6,input_shape=(batch_size,num_features)))
model.add(Bidirectional(LSTM(3,return_sequences=True)))
model.add(Dense(6))
model.add(Dropout(0.1))
model.add(Activation('softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy',tf.keras.metrics.MeanSquaredError()])




print('Train...')
model.fit(x_train, y_train,
          epochs=100,
          validation_data=(x_test, y_test))


print(model.summary())