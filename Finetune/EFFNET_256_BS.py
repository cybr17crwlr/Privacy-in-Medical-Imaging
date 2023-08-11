import numpy as np
from typing import Tuple
from scipy import special
from sklearn import metrics
import sklearn

from matplotlib import pyplot as plt
from datetime import datetime

from glob import glob
import os

import pandas as pd

import tensorflow as tf
tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

# --------------Set verbosity--------------
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from sklearn.exceptions import ConvergenceWarning

import warnings
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

## --------------Import Dataset--------------

print(tf.config.list_physical_devices('GPU'))

data_folder = 'datasets'

dataset_folder = os.path.join(data_folder,'training')
image_files = glob(os.path.join(dataset_folder,'*/*.png'))
image_files = pd.Series(image_files).str.rsplit('/',n=1,expand=True)
image_files.columns=['Folder','Image Index']

images = pd.read_csv(os.path.join(data_folder,'Data_Entry_2017.csv'))
images = image_files.merge(images.join(images['Finding Labels'].str.get_dummies()),on='Image Index',how='left')
dataset = images[['Folder', 'Image Index', 'Pneumonia']].copy()

dataset.rename(columns={'Pneumonia':'label'},inplace=True)

## --------------Model Parameters--------------

num_classes = dataset.label.nunique()

activation = 'relu'
epoch_toptrain = 20
total_epochs = 100
batchsize = 16

val_freq = 2
val_split = 0.2

imagesize=(256,256,3)

lr_stage1 = 1e-3
min_lr_stage1 = 1e-7
lr_stage2 = 1e-6
min_lr_stage2 = 1e-9

seed = 2345
shuffle = True
layer_map = {1:'grayscale',3:'rgb'}

## --------------Build DataLoaders--------------

train_ds = tf.keras.utils.image_dataset_from_directory(dataset_folder,
                                                       color_mode=layer_map[imagesize[2]],
                                                       image_size=imagesize[:2],
                                                       shuffle=shuffle,
                                                       label_mode='categorical',
                                                       validation_split=val_split,
                                                       batch_size=batchsize,
                                                       seed=seed,
                                                       subset='training')

train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

val_ds = tf.keras.utils.image_dataset_from_directory(dataset_folder,
                                                       color_mode=layer_map[imagesize[2]],
                                                       image_size=imagesize[:2],
                                                       shuffle=shuffle,
                                                       label_mode='categorical',
                                                       validation_split=val_split,
                                                       batch_size=batchsize,
                                                       seed=seed,
                                                       subset='validation')

val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

## --------------Base Model--------------

efficientnet = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top = False,
                                                             weights='imagenet',
                                                             input_shape=imagesize,
                                                             pooling='max')

efficientnet.trainable = False

## --------------Full Model--------------

inputs = tf.keras.Input(shape=imagesize, name='input')

x = efficientnet(inputs, training=False)
x = tf.keras.layers.Dense(1024, activation='relu', name='top_dense1')(x)
x = tf.keras.layers.Dropout(0.3, name='top_dropout1')(x)
x = tf.keras.layers.Dense(256, activation='relu', name='top_dense2')(x)
x = tf.keras.layers.Dropout(0.3, name='top_dropout2')(x)
predictions = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x)

net = tf.keras.Model(inputs = inputs, outputs = predictions, name=f'effnet.imagenet.{imagesize[0]}.ctl{epoch_toptrain}.ft{total_epochs}')

## --------------Compile Stage 1--------------

net.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_stage1),
    metrics=['accuracy'])

net.summary()

## --------------Define Callbacks--------------

checkpoint_filepath = f'checkpoints/effnet.imagenet.{imagesize[0]}.ctl{epoch_toptrain}.ft{total_epochs}.stage1.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filepath,
    save_weights_only = True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

logdir = f"logs/scalars/effnet{imagesize[0]}ftbsstage1"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                 mode='max',
                                                 factor=0.1,
                                                 patience=5,
                                                 min_lr=min_lr_stage1)

## --------------Train Stage 1--------------

history1 = net.fit(train_ds,
                   validation_data=val_ds,
                   epochs=epoch_toptrain,
                   validation_freq=val_freq,
                   callbacks=[model_checkpoint_callback, tensorboard_callback, reduce_lr],
                   verbose=2)

## --------------Load Best Model for Stage 2--------------

efficientnet.trainable = True

net.load_weights(checkpoint_filepath, skip_mismatch=False)

## --------------Compile Stage 2--------------

net.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_stage2),
    metrics=['accuracy'])

net.summary()

## --------------Define Callbacks Stage 2--------------

checkpoint_filepath = f'checkpoints/effnet.imagenet.{imagesize[0]}.ctl{epoch_toptrain}.ft{total_epochs}.stage2.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filepath,
    save_weights_only = True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

logdir = f"logs/scalars/effnet{imagesize[0]}ftbsstage2"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                 mode='max',
                                                 factor=0.1,
                                                 patience=5,
                                                 min_lr=min_lr_stage2)

## --------------Train Stage 2--------------

history2 = net.fit(train_ds,
                   validation_data=val_ds,
                   epochs=total_epochs-epoch_toptrain,
                   validation_freq=val_freq,
                   callbacks=[model_checkpoint_callback, tensorboard_callback, reduce_lr],
                   verbose=2)
