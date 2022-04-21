import numpy as np
from PIL import Image
from matplotlib import image
import os
import skimage as sk
import skimage.io as io
import skimage.filters
import math
import matplotlib.pyplot as plt
from skimage.transform import resize, rescale, downscale_local_mean
from skimage.morphology import closing
from skimage.morphology import disk
import pickle
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel

def save_var(var, file_name):
    '''
    Saves any type of variable with the given filename(can be a path)
    '''
    out_file = open(file_name,'wb')
    pickle.dump(var,out_file)
    out_file.close()
    
def read_var(file_name):   
    infile = open(file_name,'rb')
    var = pickle.load(infile)
    infile.close()
    return var
 
def rgb2bool(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    bool_img = (r + g + b) > 0
    return bool_img

def read_image(filename):
    img = image.imread(filename)
    return img

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def rgb2green(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    #green=3*g / (r + g + b + 0.000001)
    total = (r + g + b + 0.000001)
    green = g
    result = 3*g/total
    print(total[500:550, 512])
    print(green[500:550, 512])
    print(result[500:550, 512])
    smallest = total.min(axis=0).min(axis=0)
    biggest = total.max(axis=0).max(axis=0)
    mean = np.mean(result, axis=(0, 1))
    print(smallest)
    print(biggest)
    print(mean)
    return result
    
def rgb2yellow(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    yellow=(g + r) / 2
    return yellow

def load_data(dir_name = 'DR_dataset', preprocessing = False):    
    '''
    Load images from the given directory. "DR_dataset" as default.
    Returns image list
    '''
    imgs = []
    for filename in os.listdir(dir_name):
        if os.path.isfile(dir_name + '/' + filename):
            img=io.imread(dir_name + '/' + filename)
            if(preprocessing == True):
                preProcessing(img)
            imgs.append(img)
    return imgs
    
# def addWeighted(src1, alpha, src2, beta, gamma):

    # gamma_im = sk.img_as_float(Image.new("RGB", (src1.shape[0],src1.shape[1]), (gamma,gamma,gamma)))
    # dst = sk.img_as_float((src1*alpha) + (src2*beta) + gamma_im)
    # return dst
    
# def bens_processing(img):
    # image=sk.img_as_float(img)
    # gaus_img = skimage.filters.gaussian(image, sigma=image.shape[0]/32, truncate= 6)
    # result = addWeighted(image, 4, gaus_img, -4, 128)
    # return result
    
def addWeighted(src1, alpha, src2, beta, gamma, colored=True):
    if colored:
        gamma_im = sk.img_as_float(Image.new("RGB", (src1.shape[1],src1.shape[0]), (gamma,gamma,gamma)))
    else:
        gamma_im = sk.img_as_float(Image.new("L", (src1.shape[1],src1.shape[0]), gamma))
        
    dst = sk.img_as_float((src1*alpha) + (src2*beta) + gamma_im)
    return dst
    
    
def bens_processing(img):
    image=sk.img_as_float(img)
    gaus_img = skimage.filters.gaussian(image, sigma=image.shape[0]/32, truncate=6)
    result = addWeighted(image, 4, gaus_img, -4, 128, colored=(img.ndim==3))
    #print('result min: ', np.amin(result), 'result max: ', np.amax(result))
    #rescaled_img = sk.exposure.rescale_intensity(result, in_range=(0.5,1.5), out_range=(0,1))
    rescaled_img = sk.exposure.rescale_intensity(result, in_range=(0,1), out_range=(0,1))
    return rescaled_img

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

def extract_exterior(image):
    from skimage.morphology import opening, closing
    #disk_size_half=int(image.shape[0]/20)
    #disk_size_quarter=int(image.shape[0]/40)
    disk_size_half=10
    disk_size_quarter=20
    #Convert gray TODO:try for red, green, blue channels
    gray= rgb2gray(image)
    #Opening seperately
    #opened_h = opening(gray, disk(disk_size_half))
    #opened_qu = opening(gray, disk(disk_size_quarter))
    #diff_o_h = gray - opened_h
    #diff_o_qu = gray- opened_qu
    #Closing seperately
    closed_h = closing(gray, disk(disk_size_half))
    closed_qu = closing(gray, disk(disk_size_quarter))
    edge_sobel_h = skimage.filters.sobel(closed_h)
    edge_sobel_q = skimage.filters.sobel(closed_qu)

    show_images([image, gray, closed_h, closed_qu, edge_sobel_h, edge_sobel_q])

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

def visualize(imgs, format=None, gray=False):
    plt.figure(figsize=(20, 40))
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1,2,0)
        plt_idx = i+1
        plt.subplot(5, 5, plt_idx)
        plt.imshow(img, format)
    plt.show()

def show_images(images: list, titles: list="Untitled    ", colorScale='gray', rows = 0, columns = 0) -> None:
    n: int = len(images)
    if rows == 0:
        rows=int(math.sqrt(n))
    if columns == 0:
        columns=(n/rows)
    f = plt.figure()
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(rows, columns, i + 1)
        plt.imshow(images[i], cmap=colorScale)
        plt.title(titles[i])
    plt.show(block=True)

    
def idrid_to_4channel(ch0_file, ch1_file, ch2_file):
    ch0= io.imread(ch0_file)   
    try:
      ch1= io.imread(ch1_file)
    except:
      print("ch1 image cannot read:")
      print(ch1_file)
      ch1=np.zeros([ch0.shape[0], ch0.shape[1], 3], dtype=bool)
    try:
      ch2= io.imread(ch2_file)
    except:
      print("ch2 image cannot read:")
      print(ch2_file)
      ch2=np.zeros([ch0.shape[0], ch0.shape[1], 3], dtype=bool)


    arr_4ch = np.zeros([ch0.shape[0], ch0.shape[1], 4], dtype=bool)
    arr_4ch[:,:,0] = rgb2bool(ch0)
    arr_4ch[:,:,1] = rgb2bool(ch1)
    arr_4ch[:,:,2] = rgb2bool(ch2)
    arr_4ch[:,:,3] = np.invert(arr_4ch[:,:,0] | arr_4ch[:,:,1] | arr_4ch[:,:,2])
    return arr_4ch

def create_4ch_dataset(path0, path1, path2, path_out, ind = 1):
    entries = sorted(os.listdir(path0))
    i=ind
    for entry in entries:
        print(entry)
        index = str(i).zfill(2)
        ch0_file = path0 + "IDRiD_" + index + "_MA.png"
        ch1_file = path1 + "IDRiD_" + index + "_HE.png"
        ch2_file = path2 + "IDRiD_" + index + "_EX.png"

        arr_4ch = idrid_to_4channel(ch0_file, ch1_file, ch2_file)
        save_var(arr_4ch, path_out + "IDRiD_" + index + "_4CH")
        i=i+1
        
        
        
def patch_4ch_dataset(path0, path1, path2, path_out):
    entries = sorted(os.listdir(path0))
    rows = []
    for entry in entries:
        name, img_formatt = entry.split("MA")
        inp_name = name + img_formatt
        ch0_name = name + "MA" + img_formatt
        ch1_name = name + "HE" + img_formatt
        ch2_name = name + "EX" + img_formatt
        ch4_name = name + "4CH"
        ch0_file = path0 + ch0_name
        ch1_file = path1 + ch1_name
        ch2_file = path2 + ch2_name

        arr_4ch = idrid_to_4channel(ch0_file, ch1_file, ch2_file)
        save_var(arr_4ch, path_out + ch4_name)
        row = [inp_name, ch0_name, ch1_name, ch2_name, ch4_name]
        rows.append(row)
    return rows

def create_1024_dataset(srcFileName, destFileName):
    for filename in os.listdir(srcFileName):
        if os.path.isfile(srcFileName + '/' + filename):
            print(filename)
            img=Image.open(srcFileName + '/' + filename)
            old_size = img.size
            if((old_size[0]<1024) & (old_size[1]<1024)):
                ratio=float(old_size[0])/float(old_size[1])
                if(old_size[0]<old_size[1]):
                    img = img.resize((int(1024*ratio),1024), Image.LANCZOS)
                else:
                    img = img.resize((1024,int(1024.0/ratio)), Image.LANCZOS)
            old_size = img.size
            new_size = (1024, 1024)
            new_im = Image.new("RGB", new_size)             #this is already black!
            new_im.paste(img, (int((new_size[0]-old_size[0])/2),
                                int((new_size[1]-old_size[1])/2)))
            new_im.save(destFileName + '/' + filename)

def adems_processing(img):
    image=sk.img_as_float(img)
    small_img = resize(img, (img.shape[0]/8,img.shape[1]/8))
    
    gaus_img = skimage.filters.gaussian(small_img, sigma=small_img.shape[0]/32, truncate=6)

    diff2mean = small_img - gaus_img
    distance2mean = np.sqrt(np.multiply(diff2mean[:,:,0],diff2mean[:,:,0]) + np.multiply(diff2mean[:,:,1],diff2mean[:,:,1]) + np.multiply(diff2mean[:,:,2],diff2mean[:,:,2]))
    treshold = 0.05

    small_img0 = small_img[:,:,0]
    small_img1 = small_img[:,:,1]
    small_img2 = small_img[:,:,2]
    small_img0[distance2mean > treshold] = np.nan
    small_img1[distance2mean > treshold] = np.nan
    small_img2[distance2mean > treshold] = np.nan
    small_img[:,:,0] = small_img0
    small_img[:,:,1] = small_img1
    small_img[:,:,2] = small_img2

    kernel = Gaussian2DKernel(x_stddev=int(small_img.shape[0]/32))

    astropy_conv = np.ones_like(small_img)
    astropy_conv[:,:,0] = convolve(small_img[:,:,0], kernel)
    astropy_conv[:,:,1] = convolve(small_img[:,:,1], kernel)
    astropy_conv[:,:,2] = convolve(small_img[:,:,2], kernel)

    astropy_conv_big = resize(astropy_conv, (img.shape[0],img.shape[1]))
    result = addWeighted(image, 4, astropy_conv_big, -4, 128, colored=True)
    rescaled_img = sk.exposure.rescale_intensity(result, in_range=(0,1), out_range=(0,1))
    return rescaled_img
    
# from skimage.transform import resize
# entries = os.listdir('/home/vivente/Desktop/DR/Data/small_dataset/')
# i=int(0)
# for entry in entries:
    # img=io.imread("/home/vivente/Desktop/DR/Data/small_dataset/" + entry)
    # print(entry)
    # extract_exterior(img)
    # i=i+1