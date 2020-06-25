
from utils import PatientLoader

from processing.signal import detrend
from processing.signal import high_gamma
from processing.signal import low_component
from processing.signal import analytic_amp
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional, Activation

import numpy as np
from sklearn.model_selection import train_test_split

def split_with_overlap(data,window_size, window_overlap):
    length = data.shape[0]
    result = []
    for i in range(0,length,window_size-window_overlap):
        if length >= i + window_size:
            result.append(data[i:i+window_size,:])
    return result


from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--patient', type=int)
args = parser.parse_args()

params = {
    'patient': args.patient
}

callbacks = []
try:
    import wandb
    from wandb.keras import WandbCallback
    wandb.init(project='mrp2',
               config=params)
    callbacks.append(WandbCallback())
    def log(name, val):
        wandb.log({name: val})
    def log_wav(audio, fs):
        wandb.log({'rec_audio': wandb.Audio(audio, sample_rate=fs)})
except:
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    callbacks.append(tensorboard_cb)
    logger = tf.summary.create_file_writer('./logs/metrics')
    def log(name, val, epoch):
        tf.summary.scalar(name, data=val, step=epoch)
    log_wav = lambda _: None # not implemented for TensorBoard - do nothing



pl = PatientLoader(params['patient'], session=1)
rawseeg = pl.get_seeg()
data = detrend(rawseeg)
hg = high_gamma(data, pl.seeg_fs)
X_high_gamma = analytic_amp(hg)
lp = low_component(rawseeg, pl.seeg_fs)
X_low_component = analytic_amp(lp)


batch_size = 60
overlap_size = batch_size - 1
train_batch_size = 2000

print('Loading data...')

Y = np.load(f'./data/kh{params["patient"]}_1_metrics_1k.npy')
Y = Y.T
X = np.hstack((X_high_gamma, X_low_component))
X = X[:Y.shape[0], :]

n, num_features = X.shape

del rawseeg, data, hg, lp, X_high_gamma, X_low_component

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)

x_train = tf.data.Dataset.from_tensor_slices(x_train)
x_train = x_train.window(batch_size, shift=1, drop_remainder=True)
x_train = x_train.flat_map(lambda win: win.batch(batch_size))
y_train = split_with_overlap(y_train, batch_size, overlap_size)
y_train = [y[int(batch_size / 2)] for y in y_train]
y_train = tf.data.Dataset.from_tensor_slices(y_train)
train = tf.data.Dataset.zip((x_train, y_train))
# train = train.cache()
train = train.shuffle(10000)
train = train.batch(train_batch_size)

x_test = tf.data.Dataset.from_tensor_slices(x_test)
x_test = x_test.window(batch_size, shift=1, drop_remainder=True)
x_test = x_test.flat_map(lambda win: win.batch(batch_size))
y_test = split_with_overlap(y_test, batch_size, overlap_size)
y_test = [y[int(batch_size / 2)] for y in y_test]
y_test = tf.data.Dataset.from_tensor_slices(y_test)
test = tf.data.Dataset.zip((x_test, y_test))
# test = test.cache()
test = test.shuffle(10000)
test = test.batch(train_batch_size)

model = Sequential()
model.add(Input(shape=(batch_size,num_features)))
model.add(Bidirectional(LSTM(20,return_sequences=True)))
model.add(Bidirectional(LSTM(10,return_sequences=True)))
model.add(Bidirectional(LSTM(3,)))
model.add(Dropout(0.33))
model.add(Dense(Y.shape[1]))

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),loss='huber_loss', metrics=['mean_absolute_error'])


model.fit(train,
          epochs=10,
          validation_data=test,
          callbacks=callbacks)


model.summary()


X = tf.data.Dataset.from_tensor_slices(X)
X = X.window(batch_size, shift=1, drop_remainder=True)
X = X.flat_map(lambda win: win.batch(batch_size))

predictions = model.predict(X.batch(train_batch_size))

np.save(f'./data/kh{params["patient"]}_1_gen_kinetics.npy',predictions)
print('Results saved.')

