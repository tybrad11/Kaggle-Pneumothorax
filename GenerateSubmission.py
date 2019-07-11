from glob import glob
from os.path import join

import cv2
import numpy as np
from natsort import natsorted
from PIL import Image
import keras
from Models import BlockModel2D
from scipy.ndimage.measurements import label as scipy_label
import GPUtil
try:
    if not 'DEVICE_ID' in locals():
            DEVICE_ID = GPUtil.getFirstAvailable()[0]
            print('Using GPU',DEVICE_ID)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
except Exception as e:
    raise('No GPU available')

test_datapath = '/data/Kaggle/test-png'
weight_filepath = 'Best_Kaggle_Weights.h5'

# parameters
batch_size = 8
im_dims = (512, 512)
n_channels = 1

# Get list of files
img_files = natsorted(glob(join(test_datapath, '*.png')))

# load files into array
def LoadImg(f, dims):
    img = Image.open(f)
    img = cv2.resize(np.array(img), dims).astype(np.float)
    img -= img.mean()
    img /= img.std()
    return img

test_imgs = np.stack([LoadImg(f,im_dims) for f in img_files])[...,np.newaxis]

# Create model
model = BlockModel2D(input_shape=im_dims+(n_channels,),
                     filt_num=16, numBlocks=4)
# Load weights
model.load_weights(weight_filepath)

# Get predicted masks
masks = model.predict(test_imgs, batch_size=batch_size,verbose=1)

# process mask
cur_mask = masks[0,...,0]
cur_im = test_imgs[0,...,0]
cur_mask = (cur_mask>.5).astype(np.int)

lbl_mask,numObj = scipy_label(cur_mask)
processed_mask = np.zeros_like(cur_mask)
minimum_cc_sum = .002*np.prod(im_dims)
for label in range(1,numObj):
    if np.sum(lbl_mask == label) > minimum_cc_sum:
        processed_mask[lbl_mask == label] = 1
