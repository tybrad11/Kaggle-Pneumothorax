# %% Setup
import time
import os
from glob import glob
from os.path import join

import GPUtil
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from natsort import natsorted
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from Datagen import PngClassDataGenerator, PngDataGenerator
from HelperFunctions import (RenameWeights, get_class_datagen, get_seg_datagen,
                             get_train_params, get_val_params)
from Losses import dice_coef_loss
from Models import BlockModel2D, BlockModel_Classifier, ConvertEncoderToCED

os.environ['HDF5_USE_FILE_LOCKING'] = 'false'


rng = np.random.RandomState(seed=1)

try:
    if not 'DEVICE_ID' in locals():
        DEVICE_ID = GPUtil.getFirstAvailable()[0]
        print('Using GPU', DEVICE_ID)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
except Exception as e:
    raise('No GPU available')


# ~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~ SETUP~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~

# Setup data
pre_train_datapath = '/data/Kaggle/nih-chest-dataset/images_resampled_sorted_into_categories/Pneumothorax_norm/'
pre_train_negative_datapath = '/data/Kaggle/nih-chest-dataset/images_resampled_sorted_into_categories/No_Finding_norm/'

train_pos_datapath = '/data/Kaggle/pos-norm-png'
train_neg_datapath = '/data/Kaggle/neg-norm-png'

pretrain_weights_filepath = 'Best_pretrain_class_weights.h5'
train_weight_filepath = 'Best_Kaggle_Classification_Weights_{}.h5'

# pre-train parameters
pre_im_dims = (512, 512)
pre_n_channels = 1
pre_batch_size = 16
pre_val_split = .15
pre_epochs = 2
pre_multi_process = False

# train parameters
im_dims = (512, 512)
n_channels = 1
batch_size = 4
learnRate = 1e-4
val_split = .2
epochs = 2  # epochs before and after unfreezing weights
full_epochs = 20 # epochs trained on 1024 data
multi_process = False

# datagen params
pre_train_params = get_train_params(
    pre_batch_size, pre_im_dims, pre_n_channels)
pre_val_params = get_val_params(pre_batch_size, pre_im_dims, pre_n_channels)
train_params = get_train_params(batch_size, im_dims, n_channels)
val_params = get_val_params(batch_size, im_dims, n_channels)
full_train_params = get_train_params(2, (1024, 1024), 1)
full_val_params = get_val_params(2, (1024, 1024), 1)

# %% ~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~Pre-training~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~

print('---------------------------------')
print('---- Setting up pre-training ----')
print('---------------------------------')

# Get datagens for pre-training
pre_train_gen, pre_val_gen, class_weights = get_class_datagen(
    pre_train_datapath, pre_train_negative_datapath, pre_train_params, pre_val_params, pre_val_split)

# Create model
pre_model = BlockModel_Classifier(input_shape=pre_im_dims+(pre_n_channels,),
                                  filt_num=filt_nums, numBlocks=num_blocks)

# Compile model
pre_model.compile(Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Create callbacks
cb_check = ModelCheckpoint(pretrain_weights_filepath, monitor='val_loss',
                           verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)

print('---------------------------------')
print('----- Starting pre-training -----')
print('---------------------------------')

# Train model
pre_history = pre_model.fit_generator(generator=pre_train_gen,
                                      epochs=pre_epochs, use_multiprocessing=pre_multi_process,
                                      workers=8, verbose=1, callbacks=[cb_check],
                                      class_weight=class_weights,
                                      validation_data=pre_val_gen)

# Load best weights
pre_model.load_weights(pretrain_weights_filepath)

# Calculate confusion matrix
print('Calculating classification confusion matrix...')
pre_val_gen.shuffle = False
preds = pre_model.predict_generator(pre_val_gen, verbose=1)
labels = [pre_val_gen.labels[f] for f in pre_val_gen.list_IDs]
y_pred = np.rint(preds)
totalNum = len(y_pred)
y_true = np.rint(labels)[:totalNum]
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print('--------------------------------------')
print('Classification Results on pre-training')
print('--------------------------------------')
print('True positives: {}'.format(tp))
print('True negatives: {}'.format(tn))
print('False positives: {}'.format(fp))
print('False negatives: {}'.format(fn))
print('% Positive: {:.02f}'.format(100*(tp+fp)/totalNum))
print('% Negative: {:.02f}'.format(100*(tn+fn)/totalNum))
print('% Accuracy: {:.02f}'.format(100*(tp+tn)/totalNum))
print('% Sensitivity: {:.02f}'.format(100*(tp)/(tp+fn)))
print('% Specificity: {:.02f}'.format(100*(tn)/(tn+fp)))
print('-----------------------')
