import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import skimage as sk
from skimage.morphology import erosion, dilation, opening, closing, white_tophat, area_opening, area_closing, binary_erosion
from skimage.morphology import disk, square
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola, meijering, sato, frangi, hessian, try_all_threshold, threshold_triangle
from skimage.segmentation import flood, flood_fill
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage.filters.rank import median

import utils as ut
from opticDisk import OpticDiskExtract
from bloodVessel import extract_bv
import os

def paper_exudate(image):
    optic_image = OpticDiskExtract(image)
    disk_size_3 = 8
    disk_size_4 = 15
   
    window_size = 75
    disk_size = 45
    close_disk_size = 35
    alpha_bens = 0.35
    alpha = 0.3


    green = image[:,:,1]    #Choose green channel 
    green = (ut.delete_vessels(green, extract_bv(image))).astype('uint8')
    r1 = opening(green, disk(disk_size))
    tmp = green - r1
    bens = ut.bens_processing(tmp)
    thresh = (np.amax(bens))*alpha_bens
    binary = bens > thresh
    contours = sk.measure.find_contours(binary, 0.1)
    #ut.show_images([image, green, bens, bins[0],bins[1],bins[2],bins[3], bins[4]])


    '''# Display the image and plot all contours found
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(image)
    ax2.imshow(image)
    for n, contour in enumerate(contours):
        ax2.plot(contour[:, 1], contour[:, 0], linewidth=0.7)
    plt.show()'''
    #Second step
    r2 = closing(tmp, disk(close_disk_size))
    r3 = opening(r2, disk(25))
    r5 = r2 - r3
    bens_2 = ut.bens_processing(r5)
    thresh_2 = np.amax(bens_2)*alpha
    binary_2 = bens_2 > thresh_2
    #thresh_niblack = threshold_niblack(bens, window_size=window_size, k=-5)
    #thresh_niblack_2 = threshold_niblack(bens, window_size=window_size, k=3)
    #binary = bens > thresh_niblack

    ut.show_images([image, green, tmp, bens, r5, bens_2, binary, binary_2])


from skimage.transform import resize
entries = os.listdir('/home/vivente/Desktop/DR/Data/small_dataset/')
i=int(0)
for entry in entries:
    img=io.imread("/home/vivente/Desktop/DR/Data/small_dataset/" + entry)
    print(entry)
    result = paper_exudate(img)
    i=i+1
