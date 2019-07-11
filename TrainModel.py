import os
from glob import glob
# Setup data
from os.path import join

import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from natsort import natsorted

from Datagen import PngDataGenerator
from Losses import dice_coef_loss
from Models import BlockModel2D
from sklearn.model_selection import train_test_split

rng = np.random.RandomState(seed=1)

import GPUtil
try:
    if not 'DEVICE_ID' in locals():
            DEVICE_ID = GPUtil.getFirstAvailable()[0]
            print('Using GPU',DEVICE_ID)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
except Exception as e:
    raise('No GPU available')


train_datapath = '/data/Kaggle/train-png'
train_mask_path = '/data/Kaggle/train-mask'
weight_filepath = 'Kaggle_Weights.{epoch:02d}-{val_loss:.4f}.h5'

pretrain_weights_filepath = 'Best_Kaggle_Weights.02-0.61.h5'
# pretrain_weights_filepath = None

# parameters
im_dims = (512, 512)
n_channels = 1
batch_size = 4
val_split = .2
epochs = 5
multi_process = True
train_params = {'batch_size': batch_size,
                'dim': im_dims,
                'n_channels': 1,
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
              'n_channels': 1,
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

# Get list of files
img_files = natsorted(glob(join(train_datapath, '*.png')))
mask_files = natsorted(glob(join(train_mask_path, '*.png')))
assert len(img_files) == len(mask_files)
# Split into test/validation sets
trainX, valX, trainY, valY = train_test_split(
    img_files, mask_files, test_size=val_split, random_state=rng, shuffle=True)

train_dict = dict([(f, mf) for f, mf in zip(trainX, trainY)])
val_dict = dict([(f, mf) for f, mf in zip(valX, valY)])

# Setup datagens
train_gen = PngDataGenerator(trainX,
                             train_dict,
                             **train_params)
val_gen = PngDataGenerator(valX,
                           val_dict,
                           **val_params)


# Create model
model = BlockModel2D(input_shape=im_dims+(n_channels,),
                     filt_num=16, numBlocks=4)

# Load pretrain weights, if provided
if pretrain_weights_filepath is not None:
    model.load_weights(pretrain_weights_filepath)
# Compile model
model.compile(Adam(), loss=dice_coef_loss)

# Create callbacks
cb_check = ModelCheckpoint(weight_filepath, monitor='val_loss',
                           verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
cb_plateau = ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=3,verbose=1)


# Train model
history = model.fit_generator(generator=train_gen,
                              epochs=epochs, use_multiprocessing=multi_process,
                              workers=8, verbose=1, callbacks=[cb_check, cb_plateau],
                              validation_data=val_gen)
