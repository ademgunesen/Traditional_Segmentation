import numpy as np
import matplotlib.pyplot as plt
from utils import rgb2gray
import utils as utils
import skimage as skimage
import skimage.morphology as morphology
from skimage.filters import threshold_otsu
import skimage.io as io
from skimage.transform import resize

##########
from skimage.feature import match_template
import matplotlib.patches as patches
import os
##########

########
from skimage import data, color, img_as_ubyte
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
########

def OpticDiskExtract(image):
    image_tmp = np.copy(image)
    gray = image_tmp[:,:,0]     #Red Channel
    tmp_filter = skimage.filters.median(gray, morphology.disk(image.shape[0]/35)) #Median filter
    tmp_clahe=skimage.exposure.equalize_adapthist(tmp_filter, image.shape[0]/4) #Apply CLAHE
    cy, cx, radii, edges = HoughEllipseTransfrom(tmp_clahe, (int(image.shape[0]/20), int(image.shape[0]/10)))
    
    #Choose the bes circle and create binary image for return
    ret_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.int)
    rr, cc = skimage.draw.circle(cy[0], cx[0], radii[0], shape=None)
    ret_image[rr, cc] = 1
    return ret_image
    
def HoughEllipseTransfrom(image, circle_radii=(20, 150)):
    '''
    TODO: Canny Edge work terrible, change with another edge finder.
    Apply Hough Transform to find circles radius are between given circle_radii.
    Return three circle best fit.
    '''
    image_tmp=np.copy(image)
    edges = canny(image_tmp, sigma=0.5)
    hough_radii = np.arange(circle_radii[0], circle_radii[1], 5)
    hough_res = hough_circle(edges, hough_radii)
    
    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=3)

    return cy, cx, radii, edges

def temp_match(image):
    '''
    Apply disk with size 15 template match to the given image rgb input. Return 
    x y coordinates of most similar point in the image.
    ''' 
    gray = rgb2gray(image)  #RGB to gray level
    tmp = skimage.filters.median(gray, morphology.disk(15)) #Median filter
    tmp = skimage.exposure.equalize_adapthist(tmp, image.shape[0]/4) #Apply CLAHE
    temp_filter = morphology.disk(20)
    result = match_template(tmp, temp_filter)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]
    print(x,y)
    rect = patches.Rectangle((x,y),40,40,linewidth=1,edgecolor='r',facecolor='none')
    return gray, rect

def optic_disc_extract_visualize(image):
    '''
    Same functions as OpticDiskExtract for debugging and visualization
    '''
    image_tmp = np.copy(image)
    gray = image_tmp[:,:,0]     #Red Channel
    tmp_filter = skimage.filters.median(gray, morphology.disk(image.shape[0]/35)) #Median filter
    tmp_clahe=skimage.exposure.equalize_adapthist(tmp_filter, img.shape[0]/4) #Apply CLAHE
    cy, cx, radii, edges = HoughEllipseTransfrom(tmp_clahe, (int(image.shape[0]/20), int(image.shape[0]/10)))
    #Visualization
    colorIndex=0
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius,
                                        shape=image_tmp.shape)
        image_tmp[circy, circx] = (80*(colorIndex%3), 80*(colorIndex%3), 80*(colorIndex%3))
        colorIndex=colorIndex+1
        print(colorIndex, radius)

    utils.show_images([img, tmp_filter, tmp_clahe, edges, image_tmp, image_tmp])

