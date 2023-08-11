import numpy as np
from typing import Tuple
from scipy import special
from sklearn import metrics
import sklearn

from matplotlib import pyplot as plt

from glob import glob
import os
import shutil

import pandas as pd
from tqdm.notebook import tqdm

import cv2

import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# Set verbosity.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from sklearn.exceptions import ConvergenceWarning

import warnings
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

data_folder = 'datasets'

dataset_folder = os.path.join(data_folder,'training')
image_files = glob(os.path.join(dataset_folder,'*/*.png'))
image_files = pd.Series(image_files).str.rsplit('/',n=1,expand=True)
image_files.columns=['Folder','Image Index']

images = pd.read_csv(os.path.join(data_folder,'Data_Entry_2017.csv'))
images = image_files.merge(images.join(images['Finding Labels'].str.get_dummies()),on='Image Index',how='left')
dataset = images[['Folder', 'Image Index', 'Pneumonia']].copy()

dataset.rename(columns={'Pneumonia':'label'},inplace=True)
num_classes = dataset.label.nunique()

pd.read_csv(os.path.join(data_folder,'Data_Entry_2017.csv'))['Finding Labels'].str.get_dummies().sum().plot.bar()
plt.show()

dataset

activation = 'relu'
epoch_toptrain = 20
total_epochs = 100
batchsize = 16

val_freq = 2
val_split = 0.2

imagesize=(512,512,3)

lr_stage1 = 1e-3
lr_stage2 = 1e-5

seed = 123
shuffle = True
layer_map = {1:'grayscale',3:'rgb'}

train_ds = tf.keras.utils.image_dataset_from_directory(dataset_folder,
                                                       color_mode=layer_map[imagesize[2]],
                                                       image_size=imagesize[:2],
                                                       shuffle=shuffle,
                                                       label_mode='categorical',
                                                       validation_split=val_split,
                                                       batch_size=batchsize,
                                                       seed=seed,
                                                       subset='training')
val_ds = tf.keras.utils.image_dataset_from_directory(dataset_folder,
                                                       color_mode=layer_map[imagesize[2]],
                                                       image_size=imagesize[:2],
                                                       shuffle=shuffle,
                                                       label_mode='categorical',
                                                       validation_split=val_split,
                                                       batch_size=batchsize,
                                                       seed=seed,
                                                       subset='validation')

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomZoom(0.2, 0.2, fill_mode="constant", fill_value=0.0),
        tf.keras.layers.RandomRotation(0.1, fill_mode="constant", fill_value=0.0),
    ]
, name='augmentation')

efficientnet = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top = False,
                                                             weights='imagenet',
                                                             input_shape=imagesize,
                                                             pooling='max')

efficientnet.trainable = False

inputs = tf.keras.Input(shape=imagesize, name='input')
x = data_augmentation(inputs)

x = efficientnet(x, training=False)
x = tf.keras.layers.Dense(1024, activation='relu', name='top_dense')(x)
x = tf.keras.layers.Dropout(0.2, name='top_dropout')(x)
predictions = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x)

net = tf.keras.Model(inputs = inputs, outputs = predictions, name='effnet.aug.imagenet.512.ctl20.ft80')

net.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_stage1),
    metrics=['accuracy'])

net.summary()

checkpoint_filepath = 'checkpoints/effnet.aug.imagenet.512.ctl20.ft80.stage1'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filepath,
    save_weights_only = True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

net.fit(train_ds,
         validation_data=val_ds,
         epochs=epoch_toptrain,
         validation_freq=val_freq,
         callbacks=[model_checkpoint_callback],
         verbose=1)

efficientnet.trainable = True

net.summary()

net.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_stage2),
    metrics=['accuracy'])

checkpoint_filepath = 'checkpoints/effnet.aug.imagenet.512.ctl20.ft80.stage2'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filepath,
    save_weights_only = True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

net.fit(train_ds,
        validation_data=val_ds,
        epochs=total_epochs-epoch_toptrain,
        validation_freq=val_freq,
        callbacks=[model_checkpoint_callback],
        verbose=1)
