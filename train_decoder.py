import tensorflow as tf
from tensorflow.keras import layers

import numpy as np

import constants as const
from utils import PatientLoader
from processing.speech import world_reconstruct_audio
from processing.speech import split_world_param
from stoi import stoi

from scipy.io import wavfile

np.random.seed(0)

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--patient', type=int)
args = parser.parse_args()


params = {
    'sequence_length': 60,
    'batch_size': 2000,
    'lr': 0.002,
    'predict_midpoint': True,
    'patient': args.patient,
    'skipped_last': 0
}


### MODEL DEFINITION ###
def decoder_model(in_shape, n_outputs, lr):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=in_shape))
    model.add(layers.Dense(100, activation='relu'))
    # model.add(layers.Dropout(0.25))
    model.add(layers.Bidirectional(layers.LSTM(100, activation='tanh', return_sequences=True)))
    # model.add(layers.Dropout(0.25))
    model.add(layers.Bidirectional(layers.LSTM(200, activation='tanh', return_sequences=True)))
    # model.add(layers.Dropout(0.1))
    model.add(layers.Bidirectional(layers.LSTM(200, activation='tanh')))
    # model.add(layers.Dropout(0.25))
    model.add(layers.Dense(n_outputs))
    opt = tf.keras.optimizers.Adam(learning_rate=lr,
                                   beta_1=0.9,
                                   beta_2=0.999,
                                   epsilon=1e-08)
    model.compile(optimizer=opt, loss="mean_squared_error", metrics=['accuracy'])
    return model

### DATA LOADING ###
loader = PatientLoader(params['patient'])
# art_features = loader.get_features('metrics_1k').T
art_features = loader.get_features('gen_kinetics')
world_features = loader.get_features(f'{const.WORLD_PARAM_NAME}_1')
if params['skipped_last']:
    world_features = world_features[:, :-params['skipped_last']]

def make_datasets(art_features,
                  world_features,
                  sequence_length,
                  batch_size,
                  predict_midpoint,
                  validation_pct,
                  step_size=1):
    # Get the features to the same length
    n = min(art_features.shape[0], world_features.shape[0])
    art_features = art_features[:n, ...]
    world_features = world_features[:n, ...]
    
    # Make training sequences
    X = []
    y = []
    for i in range(0, n - sequence_length, step_size):
        X.append(art_features[i:(i + sequence_length), ...])
        if predict_midpoint:
            y.append(world_features[i + (sequence_length // 2)]) # Middle element
        else:
            y.append(world_features[i + sequence_length - 1]) # Last element

    n = len(X)
    val_idx = np.round((1 - validation_pct) * n).astype(int)
    
    X = np.array(X)
    y = np.array(y)
    X_val, y_val = X[val_idx:, ...], y[val_idx:, ...]
    X, y = X[:val_idx, ...], y[:val_idx, ...]

    train_perm = np.random.permutation(val_idx)
    val_perm = np.random.permutation(n - val_idx)

    X = X[train_perm, ...]
    y = y[train_perm, ...]
    X_val_shuff = X_val[val_perm, ...]
    y_val_shuff = y_val[val_perm, ...]

    train = tf.data.Dataset.from_tensor_slices((X, y))
    val = tf.data.Dataset.from_tensor_slices((X_val_shuff, y_val_shuff))
        
    return train.batch(batch_size), val.batch(batch_size), (X_val, y_val)

train_ds, val_ds, rec_data = make_datasets(
    art_features,
    world_features,
    sequence_length=params['sequence_length'],
    batch_size=params['batch_size'],
    predict_midpoint=params['predict_midpoint'],
    validation_pct=0.2
)

### CALLBACKS ###
class DelayedEarlyStopping(tf.keras.callbacks.EarlyStopping):
    def __init__(self, start_epoch, **kwargs):
        super(DelayedEarlyStopping, self).__init__(**kwargs)
        self._start_epoch = start_epoch
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch > self._start_epoch:
            super().on_epoch_end(epoch, logs=logs)

stopping = DelayedEarlyStopping(start_epoch=5,
                                monitor='val_loss',
                                patience=6,
                                min_delta=0.001)
callbacks = [stopping]

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


### TRAINING ###
input_shape = (params['sequence_length'], art_features.shape[1]) # (seq_length, 6)
num_outputs = world_features.shape[1]
model = decoder_model(in_shape=input_shape,
                      n_outputs=num_outputs,
                      lr=params['lr'])
# run_path = 'antonwnk/mrp2/1dfspt5y'
# model = wandb.restore('model-best.h5', run_path=run_path)
# model = tf.keras.models.load_model(model.name)
model.summary()

### INTELLIGIBILITY CALLBACK ###
def stoi_cb(epoch, logs=None, final=False):
    pred_feats = split_world_param(model.predict(rec_data[0]))
    pred_rec = world_reconstruct_audio(**pred_feats,
                                       fs=16000,
                                       frame_period=1)
    label_feats = split_world_param(rec_data[1])
    label_rec = world_reconstruct_audio(**label_feats,
                                        fs=16000,
                                        frame_period=1)
    log('stoi', stoi(label_rec, pred_rec, fs_signal=16000))
    if (epoch < 2 and (epoch % 5) != 0) or final:
        log_wav(pred_rec, fs=16000)

callbacks.append(tf.keras.callbacks.LambdaCallback(on_epoch_end=stoi_cb,
                                                   on_train_end=lambda logs: stoi_cb(-1, final=True)))


model.fit(train_ds,
          epochs=500,
          validation_data=val_ds,
          callbacks=callbacks)
