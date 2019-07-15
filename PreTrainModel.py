import os
from glob import glob
# Setup data
from os.path import join

import GPUtil
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from natsort import natsorted
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from Datagen import PngClassDataGenerator, PngDataGenerator
from Losses import dice_coef_loss
from Models import BlockModel2D, BlockModel_Classifier, ConvertEncoderToCED

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


pre_train_datapath = '/data/Kaggle/nih-chest-dataset/images_resampled_sorted_into_categories/Pneumothorax/'
pre_train_negative_datapath = '/data/Kaggle/nih-chest-dataset/images_resampled_sorted_into_categories/No_Finding/'

train_datapath = '/data/Kaggle/train-png'
train_mask_path = '/data/Kaggle/train-mask'

pretrain_weights_filepath = 'Pretrain_class_weights.h5'
weight_filepath = 'Kaggle_Weight_wpretrain_{epoch:02d}-{val_loss:.4f}.h5'

# pre-train parameters
pre_im_dims = (512, 512)
pre_n_channels = 1
pre_batch_size = 8
pre_val_split = .15
pre_epochs = 10
pre_multi_process = False

# train parameters
im_dims = (512, 512)
n_channels = 1
batch_size = 4
val_split = .2
epochs = [5, 20]  # epochs before and after unfreezing weights
multi_process = True

# datagen parameters
pre_train_params = {'batch_size': pre_batch_size,
                    'dim': pre_im_dims,
                    'n_channels': pre_n_channels,
                    'shuffle': True,
                    'rotation_range': 10,
                    'width_shift_range': 0.1,
                    'height_shift_range': 0.1,
                    'brightness_range': None,
                    'shear_range': 0.,
                    'zoom_range': 0.15,
                    'channel_shift_range': 0.,
                    'fill_mode': 'constant',
                    'cval': 0.,
                    'horizontal_flip': True,
                    'vertical_flip': False,
                    'rescale': None,
                    'preprocessing_function': None,
                    'interpolation_order': 1}

pre_val_params = {'batch_size': pre_batch_size,
                  'dim': pre_im_dims,
                  'n_channels': pre_n_channels,
                  'shuffle': True,
                  'rotation_range': 0,
                  'width_shift_range': 0.,
                  'height_shift_range': 0.,
                  'brightness_range': None,
                  'shear_range': 0.,
                  'zoom_range': 0.,
                  'channel_shift_range': 0.,
                  'fill_mode': 'constant',
                  'cval': 0.,
                  'horizontal_flip': False,
                  'vertical_flip': False,
                  'rescale': None,
                  'preprocessing_function': None,
                  'interpolation_order': 1}

train_params = {'batch_size': batch_size,
                'dim': im_dims,
                'n_channels': n_channels,
                'shuffle': True,
                'rotation_range': 10,
                'width_shift_range': 0.1,
                'height_shift_range': 0.1,
                'brightness_range': None,
                'shear_range': 0.,
                'zoom_range': 0.1,
                'channel_shift_range': 0.,
                'fill_mode': 'constant',
                'cval': 0.,
                'horizontal_flip': True,
                'vertical_flip': False,
                'rescale': None,
                'preprocessing_function': None,
                'interpolation_order': 1}

val_params = {'batch_size': batch_size,
              'dim': im_dims,
              'n_channels': n_channels,
              'shuffle': True,
              'rotation_range': 0,
              'width_shift_range': 0.,
              'height_shift_range': 0.,
              'brightness_range': None,
              'shear_range': 0.,
              'zoom_range': 0.,
              'channel_shift_range': 0.,
              'fill_mode': 'constant',
              'cval': 0.,
              'horizontal_flip': False,
              'vertical_flip': False,
              'rescale': None,
              'preprocessing_function': None,
              'interpolation_order': 1}

full_train_params = {'batch_size': 2,
                'dim': (1024,1024),
                'n_channels': n_channels,
                'shuffle': True,
                'rotation_range': 5,
                'width_shift_range': 0.05,
                'height_shift_range': 0.05,
                'brightness_range': None,
                'shear_range': 0.,
                'zoom_range': 0.05,
                'channel_shift_range': 0.,
                'fill_mode': 'constant',
                'cval': 0.,
                'horizontal_flip': True,
                'vertical_flip': False,
                'rescale': None,
                'preprocessing_function': None,
                'interpolation_order': 1}

full_val_params = {'batch_size': 2,
              'dim': (1024,1024),
              'n_channels': n_channels,
              'shuffle': True,
              'rotation_range': 0,
              'width_shift_range': 0.,
              'height_shift_range': 0.,
              'brightness_range': None,
              'shear_range': 0.,
              'zoom_range': 0.,
              'channel_shift_range': 0.,
              'fill_mode': 'constant',
              'cval': 0.,
              'horizontal_flip': False,
              'vertical_flip': False,
              'rescale': None,
              'preprocessing_function': None,
              'interpolation_order': 1}


# ~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~Pre-training~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~


# Get list of files
positive_img_files = natsorted(glob(join(pre_train_datapath, '*.png')))
print('Found {} positive files'.format(len(positive_img_files)))
negative_img_files = natsorted(
    glob(join(pre_train_negative_datapath, '*.png')))
print('Found {} negative files'.format(len(negative_img_files)))

# make labels
pos_labels = [1.]*len(positive_img_files)
neg_labels = [0.]*len(negative_img_files)
# combine
pretrain_img_files = positive_img_files + negative_img_files
pretrain_labels = pos_labels + neg_labels

# get class weights for  balancing
class_weights = class_weight.compute_class_weight(
    'balanced', np.unique(pretrain_labels), pretrain_labels)
class_weight_dict = dict(enumerate(class_weights))

# Split into test/validation sets
pre_trainX, pre_valX, pre_trainY, pre_valY = train_test_split(
    pretrain_img_files, pretrain_labels, test_size=val_split, random_state=rng, shuffle=True)

pre_train_dict = dict([(f, mf) for f, mf in zip(pre_trainX, pre_trainY)])
pre_val_dict = dict([(f, mf) for f, mf in zip(pre_valX, pre_valY)])

# Setup datagens
pre_train_gen = PngClassDataGenerator(pre_trainX,
                                      pre_train_dict,
                                      **pre_train_params)
pre_val_gen = PngClassDataGenerator(pre_valX,
                                    pre_val_dict,
                                    **pre_val_params)


# Create model
pre_model = BlockModel_Classifier(input_shape=pre_im_dims+(pre_n_channels,),
                                  filt_num=16, numBlocks=4)

# Compile model
pre_model.compile(Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Create callbacks
cb_check = ModelCheckpoint(pretrain_weights_filepath, monitor='val_loss',
                           verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
cb_plateau = ReduceLROnPlateau(
    monitor='val_loss', factor=.5, patience=3, verbose=1)


# Train model
pre_history = pre_model.fit_generator(generator=pre_train_gen,
                                      epochs=pre_epochs, use_multiprocessing=pre_multi_process,
                                      workers=8, verbose=1, callbacks=[cb_check, cb_plateau],
                                      class_weight=class_weight_dict,
                                      validation_data=pre_val_gen)

# Load best weights
pre_model.load_weights(pretrain_weights_filepath)


# ~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~ Training ~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~


# convert to segmentation model
model = ConvertEncoderToCED(pre_model)

# create segmentation datagens
train_img_files = natsorted(glob(join(train_datapath, '*.png')))
train_mask_files = natsorted(glob(join(train_mask_path, '*.png')))


# Split into test/validation sets
trainX, valX, trainY, valY = train_test_split(
    train_img_files, train_mask_files, test_size=val_split, random_state=rng, shuffle=True)

train_dict = dict([(f, mf) for f, mf in zip(trainX, trainY)])
val_dict = dict([(f, mf) for f, mf in zip(valX, valY)])

# Setup datagens
train_gen = PngDataGenerator(trainX,
                             train_dict,
                             **train_params)
val_gen = PngDataGenerator(valX,
                           val_dict,
                           **val_params)


# Create callbacks
cb_check = ModelCheckpoint(weight_filepath, monitor='val_loss',
                           verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
cb_plateau = ReduceLROnPlateau(
    monitor='val_loss', factor=.5, patience=3, verbose=1)

# Compile model
model.compile(Adam(), loss=dice_coef_loss)

history = model.fit_generator(generator=train_gen,
                              epochs=epochs[0], use_multiprocessing=multi_process,
                              workers=8, verbose=1, callbacks=[cb_check, cb_plateau],
                              validation_data=val_gen)

# make all layers trainable again
for layer in model.layers:
    layer.trainable = True

# Compile model
model.compile(Adam(), loss=dice_coef_loss)

history2 = model.fit_generator(generator=train_gen,
                               epochs=epochs[1], use_multiprocessing=multi_process,
                               workers=8, verbose=1, callbacks=[cb_check, cb_plateau],
                               validation_data=val_gen)

# make full-size model
full_model = BlockModel2D((1024,1024,n_channels),filt_num=16,numBlocks=4)
h5files = glob('*.h5')
load_file = max(h5files, key=os.path.getctime)
full_model.load_weights(load_file)

full_model.compile(Adam(lr=1e-3),loss=dice_coef_loss)
# Create callbacks
cb_check = ModelCheckpoint(weight_filepath, monitor='val_loss',
                           verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
cb_plateau = ReduceLROnPlateau(
    monitor='val_loss', factor=.5, patience=3, verbose=1)

# Setup full size datagens
train_gen = PngDataGenerator(trainX,
                             train_dict,
                             **full_train_params)
val_gen = PngDataGenerator(valX,
                           val_dict,
                           **full_val_params)
# train full size model
history_full = full_model.fit_generator(generator=train_gen,
                               epochs=2, use_multiprocessing=multi_process,
                               workers=8, verbose=1, callbacks=[cb_check, cb_plateau],
                               validation_data=val_gen)


# Rename best weights
h5files = glob('*.h5')
load_file = max(h5files, key=os.path.getctime)
os.rename(load_file,'Best_Kaggle_weights_wpretrainfull.h5')
