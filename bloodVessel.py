import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import skimage as sk
from skimage.morphology import erosion, dilation, opening, closing, white_tophat, area_opening, area_closing, binary_erosion
from skimage.morphology import disk
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola, meijering, sato, frangi, hessian, try_all_threshold
import utils as util
import os


def extract_bv(image, disk_size = 3, criteria_const = 150, small_area = 100, lowt = 0.1, hight = 0.3):
    '''
    Takes rgb image as input and finds blood vessels by using morphological operations.
    By appling regions props eliminates fake vessels
    Return binay image of blood vessels
    ''' 
    green = image[:,:,1]

    clahe = sk.exposure.equalize_adapthist(image=green)
    clahe2 = sk.exposure.equalize_adapthist(image=clahe)
    
    r1 = opening(clahe2, disk(disk_size))
    R1 = closing(r1, disk(disk_size))
    r2 = opening(R1, disk(2*disk_size + 1))
    R2 = closing(r2, disk(2*disk_size + 1))
    r3 = opening(R2, disk(4*disk_size + 3))
    R3 = closing(r3, disk(4*disk_size + 3))
    f4 = R3 - clahe2
    f5 = sk.exposure.equalize_adapthist(image=f4)
    f6 = sk.filters.apply_hysteresis_threshold(f5, lowt, hight)
    f = f6.astype(np.int)
    
    opened = area_opening(f, small_area, connectivity=2)
    label_img = sk.measure.label(opened)
    regions = sk.measure.regionprops(label_img)
    labels = np.arange(np.max(label_img) + 1)
    for props in regions:
        criteria_val = (props.perimeter*props.perimeter) / props.area
        if (criteria_val > criteria_const):
            labels[props.label] = 1
        else:
            labels[props.label] = 0
    finalMask = labels[label_img]
    util.show_images([image, green, clahe, clahe2, r1, R1],["org","green","clahe","clahe2","r1","R1"])
    util.show_images([r2, R2, r3, R3, f4, f5],["r2", "R2", "r3", "R3", "f4", "clahe3"])
    util.show_images([image, f5, f6, opened, label_img, finalMask],["org", "clahe3", "hist_tresh","opened", "regionprops", "final"])
    return finalMask.astype('bool')

def easy_bv(image):
    '''
    Takes rgb image as input and finds blood vessels by using morphological operations.
    By appling ridge operators evaluate tubeness of vessels
    Return binay image of blood vessels (TODO)
    ''' 
    disk_size = 3
    lowt = 0.25
    hight = 0.3
    lowtb = 0.11
    hightb = 0.14
    green = image[:,:,1]
    clahe = sk.exposure.equalize_adapthist(image=green)
    clahe2 = sk.exposure.equalize_adapthist(image=clahe)
    
    r1 = opening(clahe2, disk(disk_size))
    R1 = closing(r1, disk(disk_size))
    r2 = opening(R1, disk(2*disk_size + 1))
    R2 = closing(r2, disk(2*disk_size + 1))
    r3 = opening(R2, disk(4*disk_size + 3))
    R3 = closing(r3, disk(4*disk_size + 3))
    f4 = R3 - clahe2
    f5 = sk.exposure.equalize_adapthist(image=f4)
    better = f5
    
    window_size = 75
    thresh_niblack = threshold_niblack(better, window_size=window_size, k=0.2)
    thresh_sauvola = threshold_sauvola(better, window_size=window_size)
    binary_niblack = better > thresh_niblack
    binary_sauvola = better > thresh_sauvola
    
    ustu_nib = better - thresh_niblack
    ustu_sav = better - thresh_sauvola

    #util.show_images([better, better, thresh_niblack,thresh_sauvola,binary_niblack,binary_sauvola,ustu_nib,ustu_sav])

    f56 = ustu_nib
    tubeness5 = meijering(f56, black_ridges = False)
    tubeness6 = sato(f56, black_ridges = False)
    hyst_5 = sk.filters.apply_hysteresis_threshold(tubeness5, lowt, hight)
    hyst_6 = sk.filters.apply_hysteresis_threshold(tubeness6, lowtb, hightb)
    #util.show_images([green, f56, tubeness5, tubeness6, hyst_5, hyst_6], "my_nibba      ")
    
    f57 = ustu_sav
    tubeness5b = meijering(f57, black_ridges = False)
    tubeness6b = sato(f57, black_ridges = False)

    hyst_5b = sk.filters.apply_hysteresis_threshold(tubeness5b, lowt, hight)
    hyst_6b = sk.filters.apply_hysteresis_threshold(tubeness6b, lowtb, hightb)

    # f16 = hyst_5
    # e1 = closing(f16, disk(disk_size))
    # E1 = opening(e1, disk(disk_size))

    # f16 = hyst_6
    # e1 = closing(f16, disk(disk_size))
    # E2 = opening(e1, disk(disk_size))

    # f16 = hyst_5b
    # e1 = closing(f16, disk(disk_size))
    # E3 = opening(e1, disk(disk_size))

    # f16 = hyst_6b
    # e1 = closing(f16, disk(disk_size))
    # E4 = opening(e1, disk(disk_size))
    #util.show_images([image, green, f56, f57, tubeness5, tubeness6, tubeness5b, tubeness6b, 
    #                hyst_5, hyst_6, hyst_5b, hyst_6b], "savolasin 12345678", rows = 3, columns = 4)

def dummy_predict(window_size, background_treshold = 0.9):
    band = (1 - background_treshold)/4
    ch1_thres = background_treshold + band
    ch2_thres = background_treshold + 2*band
    ch3_thres = background_treshold + 3*band
    ch4_thres = background_treshold + 4*band
    random_array = np.random.random((window_size,window_size))
    binary_back = random_array < background_treshold
    binary_ch1 = (random_array > background_treshold) *( random_array < ch1_thres)
    binary_ch2 = (random_array > ch1_thres) * ( random_array < ch2_thres)
    binary_ch3 = (random_array > ch2_thres) * ( random_array < ch3_thres)
    binary_ch4 = (random_array > ch3_thres) * ( random_array < ch4_thres)
    result_5ch = np.zeros([window_size, window_size, 5], dtype=bool)
    result_5ch[:,:,0] = binary_ch1
    result_5ch[:,:,1] = binary_ch2
    result_5ch[:,:,2] = binary_ch3
    result_5ch[:,:,3] = binary_ch4
    result_5ch[:,:,4] = binary_back
    return result_5ch
    
def colorize_binary_img(bin_img):
    red = [255,0,0]
    green = [0,255,0]
    blue = [0,0,255]
    yellow = [255,255,0]
    orange = [255,120,0]
    purple = [255,0,255]
    pink = [255,120,120]
    white = [255,255,255]
    color_list = [red, green, blue, yellow, white, orange, purple, pink]
    
    colored_img = np.zeros([bin_img.shape[0],bin_img.shape[1],3], dtype = np.uint8)
    for x in range (bin_img.shape[0]):
        for y in range (bin_img.shape[1]):
            for ch in range (bin_img.shape[2]):
                if (bin_img[x,y,ch] == 1) :
                    colored_img[x,y,:]=color_list[ch]
    return colored_img
    

def create_output_patches(img, window_size, stride, output_patch_path):
    #Get valid area to not include black parts of image
    for i in range(int(img.shape[0]/stride)):
        for j in range(int(img.shape[1]/stride)):
            x_start, y_start = i*stride, j*stride
            patch_in = img[x_start:x_start+window_size, y_start:y_start+window_size,:]
            index_x = str(i).zfill(3)
            index_y = str(j).zfill(3)
            if (not((patch_in.min(axis=0).min(axis=0)<[4,4,4]).all())):
                patch_5ch = dummy_predict(window_size, background_treshold = 0.97)
                # colored = colorize_binary_img(patch_5ch)
                # util.show_images([colored])
                util.save_var(patch_5ch, output_patch_path + "IDRiD_" + index_x + index_y + "_5CH")
            else:
                patch_5ch = np.zeros([window_size, window_size, 5], dtype=bool)
                util.save_var(patch_5ch, output_patch_path + "IDRiD_" + index_x + index_y + "_5CH")

def unify_patches(img, window_size, stride, output_patch_path):
    out_image = np.zeros([img.shape[0],img.shape[1],5], dtype = bool)
    for i in range(int(img.shape[0]/stride)):
        for j in range(int(img.shape[1]/stride)):
            x_start, y_start = i*stride, j*stride
            index_x = str(i).zfill(3)
            index_y = str(j).zfill(3)
            output_patch = util.read_var(output_patch_path + "IDRiD_" + index_x + index_y + "_5CH")
            out_image[x_start:x_start+window_size, y_start:y_start+window_size,:] = output_patch
    return out_image

    
#image = io.imread("C:/Users/User/Desktop/Codes/datasets/APTOS/dataset_15&19/00a8624548a9.jpg")
#sonuc = easy_bv(image)
#util.show_images([image,sonuc])
image = io.imread("C:/Users/User/Desktop/talya fotolar islenmis/adem anadolu.png")
red = image[:,:,0]
green = image[:,:,1]
blue = image[:,:,2]
util.show_images([image,red,green,blue])
image[:,:,2]=image[:,:,2]*0
util.show_images([image,red,green,blue])