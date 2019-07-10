"""
Script for pre-training a lung segmentation model
"""
import os
from glob import glob
from os.path import join

import GPUtil
import numpy as np
from keras.optimizers import Adam
from Datagen import PngDataGenerator

from Losses import dice_coef_loss
from Models import BlockModel2D

try:
    if not 'DEVICE_ID' in locals():
        DEVICE_ID = GPUtil.getFirstAvailable()[0]
        print('Using GPU', DEVICE_ID)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
except RuntimeError as e:
    raise('No GPU available')


datapath = join('/', 'data', 'Kaggle-pneumothorax',
                'nih-chest-dataset', 'images_resampled_sorted_into_categories')
data_subdirs = glob(join(datapath, '*', ''))

# parameters
im_dims = (384, 384)
n_channels = 1
batch_size = 16
train_params = {'batch_size': batch_size,
                'dim': im_dims,
                'n_channels': 3,
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
                'vertical_flip': True,
                'rescale': None,
                'preprocessing_function': None,
                'interpolation_order': 1}

datagen = PngDataGenerator()


# Create model
model = BlockModel2D(input_shape=im_dims+(n_channels,), filt_num=16, numBlocks=4)
# Compile model
model.compile(Adam(), loss=dice_coef_loss)

# Make datagens

# Train model

