import concurrent.futures
import csv
import os
import sys
from glob import glob
from os.path import join

import cv2
import GPUtil
import keras
import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted
from PIL import Image
from scipy.ndimage.measurements import label as scipy_label
from tqdm import tqdm

from mask_functions_pneumothorax import mask2rle, rle2mask
from Models import BlockModel2D, Inception_model
from ProcessMasks import CleanMask_v1
from VisTools import mask_viewer0

try:
    if not 'DEVICE_ID' in locals():
        DEVICE_ID = GPUtil.getFirstAvailable()[0]
        print('Using GPU', DEVICE_ID)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
except Exception as e:
    raise('No GPU available')


def splitfile(file):
    _, file = os.path.split(file)
    return os.path.splitext(file)[0]


test_datapath = '/data/Kaggle/test-norm-png-V2'
class_weights_filepath = 'Best_Kaggle_Classification_Weights_1024train.h5'
weight_filepath = 'Best_Kaggle_Weights_1024train.h5'
submission_filepath = 'Submission_v3.csv'

# parameters
batch_size = 4
im_dims = (1024, 1024)
n_channels = 1

# Get list of files
img_files = natsorted(glob(join(test_datapath, '*.png')))


def LoadImg(f, dims=(1024,1024)):
    img = Image.open(f)
    img = cv2.resize(np.array(img), dims).astype(np.float)
    img /= 255.
    return img

def GetSubData(file,label,mask):
    mask = mask[...,0]
    mask = (mask > .5).astype(np.int)
    fid = splitfile(file)

    if label == 0:
        return [fid,-1]

    processed_mask = CleanMask_v1(mask)
    lbl_mask, numObj = scipy_label(processed_mask)
    if numObj > 0:
        processed_mask[processed_mask > 0] = 255
        processed_mask = np.transpose(processed_mask)
        rle = mask2rle(processed_mask, 1024, 1024)
    else:
        rle = -1
    return [fid, rle]


# load files into array
tqdm.write('Loading images...')
img_list = list()
with concurrent.futures.ProcessPoolExecutor() as executor:
    for img_array in tqdm(executor.map(LoadImg, img_files),total=len(img_files)):
        # put results into correct output list
        img_list.append(img_array)
test_imgs = np.stack(img_list)[...,np.newaxis]


# Load classification model
tqdm.write('Loading classification model...')
class_model = Inception_model(input_shape=(1024,1024)+(n_channels,))
class_model.load_weights(class_weights_filepath)

# Get classification predictions
tqdm.write('Making classification predictions...')
pred_labels = class_model.predict(test_imgs,batch_size=4,verbose=1)
pred_labels = np.rint(pred_labels)[:,0]

# remove model
del class_model
tqdm.write('Finished with classification model')

# Create model
tqdm.write('Loading segmentation model...')
model = BlockModel2D(input_shape=im_dims+(n_channels,),
                     filt_num=16, numBlocks=4)
# Load weights
model.load_weights(weight_filepath)

# Get predicted masks
tqdm.write('Getting predicted masks...')
masks = model.predict(test_imgs, batch_size=batch_size, verbose=1)

# data to write to csv
submission_data = []
# process mask
tqdm.write('Processing masks...')
with concurrent.futures.ProcessPoolExecutor() as executor:
    for sub_data in tqdm(executor.map(GetSubData,img_files,pred_labels,masks),total=len(img_files)):
        # put results into correct output list
        submission_data.append(sub_data)

# write to csv
tqdm.write('Writing csv...')
with open(submission_filepath, mode='w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['ImageId', 'EncodedPixels'])
    for data in submission_data:
        writer.writerow(data)

# display some images
mask_viewer0(test_imgs[:100,...,0],masks[:100,...,0],labels=pred_labels[:100])

print('Done')
