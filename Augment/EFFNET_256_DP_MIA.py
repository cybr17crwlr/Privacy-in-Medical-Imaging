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
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

## --------------Import TF Privacy--------------
import tensorflow_privacy
from tensorflow_privacy import compute_dp_sgd_privacy_statement
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackResultsCollection
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import PrivacyMetric
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import PrivacyReportMetadata
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import privacy_report

# --------------Set verbosity--------------

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from sklearn.exceptions import ConvergenceWarning

import warnings
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

print(tf.config.list_physical_devices('GPU'))
# mirrored_strategy = tf.distribute.MirroredStrategy()

## --------------MIA Attack Class--------------

class PrivacyMetrics(tf.keras.callbacks.Callback):
    def __init__(self, epochs_per_report, model_name):
        self.epochs_per_report = epochs_per_report
        self.model_name = model_name
        self.attack_results = []

    def on_epoch_end(self, epoch, logs=None):
        epoch = epoch+1

        if epoch % self.epochs_per_report != 0:
            return

        print(f'\nRunning privacy report for epoch: {epoch}\n')

        prediction = []
        labels = []
        for i in val_ds.as_numpy_iterator():
            prediction.append(self.model.predict(i[0]))
            labels.append(i[1])
        prob_test = np.concatenate(prediction)
        labels_test = np.concatenate(labels).astype(np.uint8)

        prediction = []
        labels = []
        for i in train_ds.as_numpy_iterator():
            prediction.append(self.model.predict(i[0]))
            labels.append(i[1])
        prob_train = np.concatenate(prediction)
        labels_train = np.concatenate(labels).astype(np.uint8)

        # Add metadata to generate a privacy report.
        privacy_report_metadata = PrivacyReportMetadata(
            # Show the validation accuracy on the plot
            # It's what you send to train_accuracy that gets plotted.
            accuracy_train=logs['val_accuracy'], 
            accuracy_test=logs['val_accuracy'],
            epoch_num=epoch,
            model_variant_label=self.model_name)

        attack_results = mia.run_attacks(
            AttackInputData(
                labels_train=labels_train[:,1],
                labels_test=labels_test[:,1],
                probs_train=prob_train,
                probs_test=prob_test),
            SlicingSpec(entire_dataset=True, by_class=True),
            attack_types=(AttackType.THRESHOLD_ATTACK,
                        AttackType.LOGISTIC_REGRESSION),
            privacy_report_metadata=privacy_report_metadata)

        self.attack_results.append(attack_results)

        results = AttackResultsCollection(self.attack_results)

        pickle.dump(results, open(f'Augment/privacy_reports/effnet.aug.imagenet.{imagesize[0]}.full{total_epochs}.dp{noise_multiplier}.pkl', 'wb'))

## --------------Import Dataset--------------

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
total_epochs = 200
batchsize = 16

val_freq = 2
val_split = 0.2

privacy_freq = 2

imagesize=(256,256,3)

lr_stage1 = 1e-3
min_lr = 1e-7

seed = 123
shuffle = True
layer_map = {1:'grayscale',3:'rgb'}

l2_norm_clip = 1.5
noise_multiplier = 0.1
num_microbatches = 8

if batchsize % num_microbatches != 0:
    raise ValueError('Batch size should be an integer multiple of the number of microbatches')

# with mirrored_strategy.scope():
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

## --------------Data Augmentation MiniModel--------------

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomZoom(0.2, 0.2, fill_mode="constant", fill_value=0.0),
        tf.keras.layers.RandomRotation(0.1, fill_mode="constant", fill_value=0.0),
    ]
, name='augmentation')

## --------------Base Model--------------

efficientnet = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top = False,
                                                            weights='imagenet',
                                                            input_shape=imagesize,
                                                            pooling='max')

## --------------Full Model--------------

inputs = tf.keras.Input(shape=imagesize, name='input')
x = data_augmentation(inputs)

x = efficientnet(x)
x = tf.keras.layers.Dense(1024, activation='relu', name='top_dense1')(x)
x = tf.keras.layers.Dropout(0.2, name='top_dropout1')(x)
x = tf.keras.layers.Dense(256, activation='relu', name='top_dense2')(x)
predictions = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x)

net = tf.keras.Model(inputs = inputs, outputs = predictions, name=f'effnet.aug.imagenet.{imagesize[0]}.full{total_epochs}.dp{noise_multiplier}')

## --------------Privacy Based Optimizer--------------

optimizer = tensorflow_privacy.VectorizedDPKerasAdamOptimizer(l2_norm_clip=l2_norm_clip,
                                                            noise_multiplier=noise_multiplier,
                                                            num_microbatches=num_microbatches,
                                                            learning_rate=lr_stage1)

loss = tf.keras.losses.CategoricalCrossentropy(
    from_logits=False, reduction=tf.losses.Reduction.NONE)

privacy_callback = PrivacyMetrics(privacy_freq, net.name)

## --------------Compile--------------

net.compile(loss=loss,
            optimizer=optimizer,
            metrics=['accuracy'])

net.summary()

## --------------Define Callbacks--------------

checkpoint_filepath = f'checkpoints/effnet.aug.imagenet.{imagesize[0]}.full{total_epochs}.dp{noise_multiplier}.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filepath,
    save_weights_only = True,
    monitor='val_accuracy',
    mode='max',
    save_best_only = True)

logdir = f"logs/scalars/effnet{imagesize[0]}augdpmia"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                 mode='max',
                                                 factor=0.5,
                                                 patience=5,
                                                 min_lr=min_lr)

## --------------Train--------------

if os.path.exists(checkpoint_filepath):
    net.load_weights(checkpoint_filepath)

history = net.fit(train_ds,
                  validation_data=val_ds,
                  epochs=total_epochs,
                  validation_freq=val_freq,
                  callbacks=[privacy_callback, model_checkpoint_callback, reduce_lr, tensorboard_callback],
                  verbose=1)

## --------------Privacy Statistics--------------

compute_dp_sgd_privacy_statement(n=len(train_ds),
                                 batch_size=batchsize,
                                 noise_multiplier=noise_multiplier,
                                 epochs=total_epochs,
                                 delta=3e-4)

# privacy_metrics = (PrivacyMetric.AUC, PrivacyMetric.ATTACKER_ADVANTAGE)
# epoch_plot = privacy_report.plot_by_epochs(results, 
#                                            privacy_metrics=privacy_metrics)