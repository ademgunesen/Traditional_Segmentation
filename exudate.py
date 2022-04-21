import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import skimage.morphology as morphology
from skimage.morphology import erosion, binary_erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola, try_all_threshold, median, unsharp_mask, threshold_multiotsu, sobel
from skimage.transform import rescale, resize
import utils as util
from skimage.segmentation import flood, flood_fill
import bloodVessel as bv
import opticDisk as od
from skimage.color import label2rgb
from skimage.feature import canny

import time

########################
#FOR window_std Function
from scipy.ndimage.filters import uniform_filter
########################
#FOR TEST - DELETE
import os
import skimage.io as io
from skimage.transform import resize
########################

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
    eroded = binary_erosion(result, disk(50))
    # util.show_images([med, clahe, gamma_corrected, eroded])
    return eroded.astype('bool')

def delete_vessels(image, vessels):
    '''
    Takes gray image as input and binary image of vessels. Mask blood vessel regions with
    closing version of the original gray level image.
    Returns image with no vessels
    ''' 
    i_vessels = vessels < 0.5
    closed = closing(image, disk(15))
    hop = closed * vessels
    top = image * i_vessels
    no_vessel = hop + top
    return no_vessel
    
def get_exudate(image):
    '''
    Takes rgb image as input and convert it to gray level scale. Mask blood vessel regions with
    closing version of the original gray level image. Apply treshold to get exudates
    Return binary image of exudates
    ''' 
    tic = time.perf_counter()
    vessels = bv.extract_bv(image)
    toc = time.perf_counter()
    print(f"Damar bulma süresi: \t\t {toc - tic:0.4f} seconds")
    
    red = image[:,:,0]
    green = image[:,:,1]
    blue = image[:,:,2]
    
    tic = time.perf_counter()
    better = delete_vessels(green, vessels)
    toc = time.perf_counter()
    print(f"Damar silme süresi: \t\t {toc - tic:0.4f} seconds")
    
    tic = time.perf_counter()
    bensetti = util.bens_processing(better)
    toc = time.perf_counter()
    print(f"Bens_processing süresi: \t {toc - tic:0.4f} seconds")
    
    tic = time.perf_counter()
    #util.show_images([image, red, green, better, better_2, bensetti])
    w_tophat = white_tophat(bensetti, disk(31)) 
    toc = time.perf_counter()
    print(f"Blob bulma süresi: \t\t {toc - tic:0.4f} seconds")
    
    
    binary = w_tophat > 0.3
    #util.show_images([image, better, w_tophat, binary])
    
    tic = time.perf_counter()
    valid_area = get_eyeshape(green)
    toc = time.perf_counter()
    print(f"Göz şekli bulma süresi: \t {toc - tic:0.4f} seconds")
    
    tic = time.perf_counter()
    invalid_area = od.OpticDiskExtract(image).astype('bool')
    toc = time.perf_counter()
    print(f"Optik disk bulma süresi: \t {toc - tic:0.4f} seconds")
    util.show_images([image, vessels, better, bensetti, w_tophat, invalid_area])
    
    valid_area_2 = invalid_area < 0.5
    result = valid_area * valid_area_2 * binary
    
    tic = time.perf_counter()
    contours = sk.measure.find_contours(result, 0.1)
    toc = time.perf_counter()
    print(f"Contour bulma süresi: \t\t {toc - tic:0.4f} seconds")
    
    # Display the image and plot all contours found
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image)
    ax2.imshow(image)
    for n, contour in enumerate(contours):
        ax2.plot(contour[:, 1], contour[:, 0], linewidth=0.5, color = 'blue')
    plt.show()



def exudate_extract(image):
    '''
    Takes rgb image as input and convert it to gray level scale. Apply closing operation,
    local standard derivation.
    Return
    ''' 
    image_tmp = np.copy(image)
    image_tmp1 = image_tmp[:,:,0]
    #image_tmp2 = sk.filters.median(image_tmp1, morphology.disk(5)) #Median filter
    image_tmp2 = image_tmp1
    image_tmp3 = sk.exposure.equalize_adapthist(image_tmp2, img.shape[0]/4) #Apply CLAHE
    image_tmp4 = morphology.closing(image_tmp3, morphology.disk(6))
    image_tmp5 = window_stdev(image_tmp4, 7)
    thresh = sk.filters.threshold_otsu(image_tmp5)
    binary = image_tmp5 > thresh
    #dilated = morphology.binary_dilation(binary, morphology.disk(3))
    dilated=binary
    #filled = flood_fill(dilated, (0,0), 1)
    filled = binary

    
    util.show_images([image_tmp, image_tmp2, image_tmp3, image_tmp4, image_tmp5, binary, dilated, filled])


def window_stdev(X, window_size):
    '''
    Calculates local standard deviation given window size
    '''
    r,c = X.shape
    X+=np.random.rand(r,c)*1e-6
    c1 = uniform_filter(X, window_size, mode='reflect')
    c2 = uniform_filter(X*X, window_size, mode='reflect')
    return np.sqrt(c2 - c1*c1)

# path = r'C:/Users/User/Desktop/Codes/diabetic-retinopathy/FeatureExtraction/exudate_dataset/'    
# entries = os.listdir(path)
# i=int(0)
# for entry in entries:
    # img=io.imread(path+entry)
    # print(entry)
    # try:
        # result = get_exudate(img)
    # except:
        # print("This image is problematic")
    # i=i+1
img = io.imread("C:/Users/User/Desktop/Codes/datasets/APTOS/dataset_15&19/0dbaa09a458c.jpg")
result = get_exudate(img)