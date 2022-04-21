import skimage.io as io
import numpy as np
import skimage as sk
import utils as util
import augmentation as aug
import collections
from skimage.transform import resize
import os
import csv

from skimage.filters.rank import mean
from skimage.morphology import square, opening
from skimage.util import img_as_int, img_as_ubyte, img_as_float
from collections import Counter
from skimage.transform import resize
from skimage.color import rgb2hsv, hsv2rgb

import skimage.morphology as morphology
from skimage.morphology import erosion, binary_erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola, try_all_threshold, median, unsharp_mask, threshold_multiotsu, sobel
from skimage.transform import rescale, resize

def get_eyeshape(image):
    image_resized = resize(image, (image.shape[0] // 8, image.shape[1] // 8),
                            anti_aliasing=True)
    image_padded = np.pad(image_resized, 1, mode='constant')
    med = median(image_padded, disk(8))
    clahe = sk.exposure.equalize_adapthist(med)
    gamma_corrected = sk.exposure.adjust_gamma(clahe, 0.3)
    thresh = threshold_otsu(gamma_corrected)
    binary = gamma_corrected > thresh
    binary_padded = np.pad(binary, 1, mode='constant')
    image_resized = resize(binary_padded, (image.shape[0], image.shape[1]),
                        anti_aliasing=True)
    result = image_resized > 0.5
    eroded = binary_erosion(result, disk(10))
    # util.show_images([med, clahe, gamma_corrected, eroded])
    return eroded.astype('bool')

def preprocess(img):
    i_bg = mean(img[:,:,1], square(64))
    i_open = opening(img[:,:,1])
    i_norm = i_open.astype(np.int16) - i_bg.astype(np.int16)
    (values,counts) = np.unique(i_norm,return_counts=True)
    ind=np.argmax(counts)
    print(values[ind])
    i_adjusted = values[ind] + 128 + 8*i_norm
    i_sc = i_adjusted.astype(np.uint8)
    
    valid_area = get_eyeshape(img[:,:,1])
    brightness = i_sc * valid_area
    hsv_img = rgb2hsv(img)

    hsv_img[:,:,2] = img_as_float(brightness)
    rgb_img = hsv2rgb(hsv_img)
    
    util.show_images([img, i_bg, i_open, i_norm, i_adjusted, i_sc, brightness, rgb_img])
    return i_norm


idrid_path = r'C:/Users/User/Desktop/Codes/copluk/datasets/dataset_1024x1024_bens_4ch/1. Original Images/a. Training Set/'
new_dataset_dir = r'C:/Users/User/Desktop/Codes/copluk/datasets/'

# aug.create_dataset(idrid_path, new_dataset_dir, bens = True, multi_channel = True, w=1024, h=1024)

# aug.patch_dataset(idrid_path,new_dataset_dir, window_size = 512, stride = 256)

entries = os.listdir(idrid_path)
i=int(0)
for entry in entries:
    img=io.imread(idrid_path + entry)
    print(entry)
    result = preprocess(img)
    util.show_images([img,result])
    i=i+1