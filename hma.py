import os
from skimage import io
import numpy as np
import utils as util
from skimage.filters import median
from skimage.filters.rank import modal
from skimage.morphology import square
import skimage.morphology as morphology
#***************************************
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, UpSampling2D, Reshape
from tensorflow.keras.optimizers import Adam

def xml_to_df(xml_file, df_cols):
    import xml.etree.ElementTree as et
    
    '''
    Parse the input XML file and store the result in a pandas 
    DataFrame with the given columns. 
    
    The first element of df_cols is supposed to be the identifier 
    variable, which is an attribute of each node element in the 
    XML data; other features will be parsed from the text content 
    of each sub-element. 
    '''

    xtree = et.parse(xml_file)
    xroot = xtree.getroot()
    rows = []
    
    for node in xroot: 
        res = []
        res.append(node.attrib.get(df_cols[0]))
        for el in df_cols[1:]: 
            if node is not None and node.find(el) is not None:
                res.append(node.find(el).text)
            else: 
                res.append(None)
        rows.append({df_cols[i]: res[i] 
                     for i, _ in enumerate(df_cols)})
    
    out_df = pd.DataFrame(rows, columns=df_cols)
        
    return out_df

def split_data(train_df, dest_path, ratio=0.2, seed=0):
    '''
    Takes dataframe as input and split ratio. Split dataframe to train and valdiation set.
    Save csv file to destination path with additional label row
    Return given input with split-name rows
    '''
    from sklearn.model_selection import train_test_split
    #Train - Validation split 20%
    train, validation = train_test_split(train_df, test_size=0.2, random_state=seed)

    train['set'] = 'train'
    validation['set'] = 'validation'
    train_complete = train.append(validation)
    train_complete.to_csv(dest_path, index=False)

    return train_complete

def data_gen(dataframe, img_folder, mask_folder, batch_size):
    import random
    c = 0
    #n = os.listdir(img_folder) #List of training images
    #random.shuffle(n)
    #index=np.arrange(len(dataframe.index))
    dataframe.sample(frac=1)  #Shuffle dataset using sample method
  
    while (True):
        img = np.zeros((batch_size, 128, 128, 3))
        mask = np.zeros((batch_size, 128,128,4))

        for i in range(c, c+batch_size):

            train_img = io.imread(img_folder+dataframe.loc[i,'x'])/255.
            if(train_img.shape==(128, 128, 3)): #check for image array size
                img[i-c] = train_img #add to array - img[0], img[1], and so on.

            train_mask = util.read_var(mask_folder+dataframe.loc[i,'y'])/255.                    
            if(train_mask.shape==(128, 128, 4)): #check for mask array size
                mask[i-c] = train_mask

        c+=batch_size
        if(c+batch_size>=len(dataframe.index)):
            c=0
            dataframe.sample(frac=1)
                  # print "randomizing again"
        yield img, mask

def model_SegNet(input_shape=(128,128,3), classes=4):
    input_tensor = Input(shape=input_shape)
    #Encoder
    model = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=input_shape)(input_tensor)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(model)

    model = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(model)

    model = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(model)

    #Decoder
    '''
    Here, there are two ways to implement upsampling. One is Upsampling2D which is upsampling your feature map. Like
    [[1, 2], [3, 4]] -> [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]. Then Conv2DTranspose which with 
    stride=1 and padding='same' is equivalent to Conv2D can be used.
    Another approach is un(max)pooling feature map. Like,
    [[1, 2], [3, 4]] -> [[1, 0, 2, 0], [0, 0, 0, 0], [3, 0, 4, 0], [0, 0, 0, 0]]. Then use  Conv2DTranspose or Conv2D.
    More reasonable explanation of difference can be found in 
    Link: https://stackoverflow.com/questions/48226783/what-is-the-difference-between-performing-upsampling-together-with-strided-trans
    '''
    model = UpSampling2D(size=(2, 2), data_format=None, interpolation="nearest")(model)
    model = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)


    model = UpSampling2D(size=(2, 2), data_format=None, interpolation="nearest")(model)
    model = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    
    model = UpSampling2D(size=(2, 2), data_format=None, interpolation="nearest")(model)
    model = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(4, kernel_size=(3, 3), strides=(1, 1), padding='same')(model) #Add softmax layer
    model = BatchNormalization()(model)
    out = Activation('softmax')(model)

    my_model = Model(input_tensor, out)

    return my_model

def train_hma():
    
    IMG_HEIGHT, IMG_WIDTH = 128, 128
    BATCH_SIZE = 64
    EPOCH = 30
    
    #Read data
    x_path = '/home/vivente/Downloads/IDRID/A. Segmentation/Patches/Input/'
    y_path = '/home/vivente/Downloads/IDRID/A. Segmentation/Patches/Output/'
    df_path = '/home/vivente/Desktop/DR/patch_dataframe_fold.csv'

    seed = 0
    df = pd.read_csv(df_path)
    train_df = df[df['set'] == 'train']
    valid_df = df[df['set'] == 'validation']
    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    #Create data generator
    train_gen = data_gen(dataframe=train_df, img_folder=x_path, mask_folder=y_path, batch_size=BATCH_SIZE)
    val_gen = data_gen(dataframe=valid_df, img_folder=x_path, mask_folder=y_path, batch_size=BATCH_SIZE)

    #Define model
    model = model_SegNet()

    model.compile(loss='categorical_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])

    model.summary()

    #Train model
    history = model.fit_generator(train_gen,
                        steps_per_epoch=len(train_df.index) // (BATCH_SIZE),
                        validation_data=val_gen,
                        validation_steps=len(valid_df.index) // (BATCH_SIZE),
                        epochs=EPOCH).history

    
    #Visualize history
    history = {'loss': history['loss'], 'val_loss': history['val_loss'], 
            'accuracy': history['accuracy'], 'val_accuracy': history['val_accuracy']}

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(20, 18))

    ax1.plot(history['loss'], label='Train loss')
    ax1.plot(history['val_loss'], label='Validation loss')
    ax1.legend(loc='best')
    ax1.set_title('Loss')

    ax2.plot(history['accuracy'], label='Train accuracy')
    ax2.plot(history['val_accuracy'], label='Validation accuracy')
    ax2.legend(loc='best')
    ax2.set_title('Accuracy')

    plt.xlabel('Epochs')
    plt.show()

def get_mode(image):
    '''
    Get images can be rgb or gray. Returns max occurence pixel value except 0.
    If 0 is the mode, then return second most value
    '''
    if(len(image.shape)==2):
        #Gray image
        value,count = np.unique(image.reshape(-1,1), axis=0, return_counts=True)
    else:
        #RGB image
        value,count = np.unique(image.reshape(-1,img.shape[-1]), axis=0, return_counts=True)
    #Delete zero
    count[count.argmax()]=0
    #value, count = value[:], count[1:]
    return value[count.argmax()]

def preprocessing_hma(image):
    '''
     Takes 1 channel 2D image array.Takes rgb image
    '''
    import matplotlib.pyplot as plt
    from skimage.exposure import rescale_intensity
    '''
    #RGB histogram
    _ = plt.hist(image.ravel(), bins = 256, color = 'orange', )
    _ = plt.hist(image[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)
    _ = plt.hist(image[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)
    _ = plt.hist(image[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)
    _ = plt.xlabel('Intensity Value')
    _ = plt.ylabel('Count')
    _ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
    plt.show()

    #GRAY histogram
    gray = ut.rgb2gray(image)
    ax = plt.hist(gray.ravel(), bins = 256)
    plt.show()
    '''

    '''Remanining Code'''
    hsv_img = rgb2hsv(image)
    value = hsv_img[:,:,2]

    #ax = plt.hist(value.ravel(), bins = 500)
    #plt.show()

    img_background = median(value, square(120))
    #image_opened = morphology.opening(value, morphology.disk(5))
    image_opened = value
    value_img = image_opened.astype(np.float) - img_background.astype(np.float)
    subtract_img = image_opened.astype(np.float) - img_background.astype(np.float)
    value_img = rescale_intensity(value_img, out_range=(0, 1))
    #np.bincount(ret_img).argmax()

    ax = plt.hist(value_img.ravel(), bins = 500)
    plt.show()

    count_zero = 0
    mode_of_img = (get_mode(value_img))[0]
    for i in range(value_img.shape[0]):
        for j in range(value_img.shape[1]):
            if((value_img[i,j]<0)):
                value_img[i,j] = 0
            elif((value_img[i,j]<1)):
                value_img[i,j] = value_img[i,j] + 0.5 - mode_of_img
            else:
                value_img[i,j] = 1
                
    value_img = value_img.clip(max=1, min=0)
    hsv_img[:,:,2] = value_img  
    ret_img = hsv2rgb(hsv_img)
    ut.show_images([image, img_background, subtract_img, value_img, value_img, ret_img])
    
def rgb2hsv(image):
    from skimage.color import rgb2hsv
    hsv = rgb2hsv(image)
    
    return hsv

def hsv2rgb(image):
    from skimage.color import hsv2rgb
    rgb = hsv2rgb(image)
    
    return rgb

def delete_background_shade(image):
    from skimage import exposure
    '''
    Takes rgb image delete background light or texture changes. Remove small gradients.
    Return rgb image.
    '''
    '''
    hsv = rgb2hsv(image)
    v_value = hsv[:,:,2]
    correction_value=[] 
    for i in range(v_value.shape[0]):
        for j in range (v_value.shape[1]):
            correction_value[i,j] = sqrt(1 - (v_value[i,j] - 1)^2)
    '''
    gamma_corrected = exposure.adjust_gamma(image, 1.2)
    return gamma_corrected

def remove_background(image):
    '''
    Takes 1 channel 2D image array. Apply median filter and extract img- median_image. The maps high 
    intensity(positive values) to 0 to get rid of exudates.
    '''
    img_background = median(image, square(25))
    ret_img = image.astype(np.int) - img_background.astype(np.int)
    ret_img = ret_img.clip(max=0)
    return ret_img

def create_joined_segmented_dataset(src_path="", dest_path=""):

    path_ma = '/home/vivente/Downloads/IDRID/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/1. Microaneurysms/'    
    path_hm = '/home/vivente/Downloads/IDRID/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/2. Haemorrhages/'   
    path_he = '/home/vivente/Downloads/IDRID/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/3. Hard Exudates/'   
    path_se = '/home/vivente/Downloads/IDRID/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/4. Soft Exudates/'   
    dest_path = '/home/vivente/Downloads/IDRID/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/0. All/'
    patch_path = '/home/vivente/Downloads/IDRID/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/Patches/'

    util.create_4ch_dataset(path_ma, path_hm, path_he, path_se, dest_path)

def create_patches(src_path_input, src_path_output, dest_path_input, dest_path_output):
    window_size = 128
    stride = 64
    df = pd.DataFrame()

    for entry in range(1,55):
        print(entry)
        img = io.imread(src_path_input + 'IDRiD_'+str("%02d" % entry)+'.jpg')
        var = util.read_var(src_path_output + 'IDRiD_'+str("%02d" % entry)+'_4CH')

        #Get valid area to not include black parts of image
        for i in range(int(img.shape[0]/stride)):
            for j in range(int(img.shape[1]/stride)):
                x_start, y_start = i*stride, j*stride
                patch_in = img[x_start:x_start+window_size, y_start:y_start+window_size,:]
                patch_out = var[x_start:x_start+window_size, y_start:y_start+window_size,:]
                if (not((patch_in.min(axis=0).min(axis=0)<[4,4,4]).all())):
                    io.imsave(dest_path_input+'IDRiD_'+str("%02d" % entry)+'_'+str(i)+'_'+str(j)+'.jpg', patch_in)
                    util.save_var(patch_out, dest_path_output+'IDRiD_'+str("%02d" % entry)+'_'+str(i)+'_'+str(j)+'_4CH')
                    
                    df = df.append({'x': 'IDRiD_'+str("%02d" % entry)+'_'+str(i)+'_'+str(j)+'.jpg', 
                                    'y':'IDRiD_'+str("%02d" % entry)+'_'+str(i)+'_'+str(j)+'_4CH'}, ignore_index=True)
    df.to_csv('patch_dataframe.csv')

'''
from skimage.transform import resize
entries = os.listdir('/home/vivente/Desktop/DR/Data/small_dataset/')
i=int(0)
for entry in entries:
    img=io.imread("/home/vivente/Desktop/DR/Data/small_dataset/" + entry)
    print(entry)
    preprocessing_hma(img)
    i=i+1
'''
#First create 4 channel segmentation mask set. Since dataset has individiual mask for exudate, hemoraji and microaneurisma, I merge them to (width,height,4) 
#array and save that variable using pickle library. Then patches which are in size 128*128 are generated and saved to folder Patches. Also Dataframe which includes
#file names for input image and output masks is generated too. Then dataframe is splited to train and validation folds.
#create_joined_segmented_dataset() 
#output_4c_path = '/home/vivente/Downloads/IDRID/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/0. All/'
#output_patch_path = '/home/vivente/Downloads/IDRID/A. Segmentation/Patches/Output/'

#input_4c_path='/home/vivente/Downloads/IDRID/A. Segmentation/1. Original Images/a. Training Set/'
#input_patch_path= '/home/vivente/Downloads/IDRID/A. Segmentation/Patches/Input/'

#create_patches(input_4c_path, output_4c_path, input_patch_path, output_patch_path)
#df=pd.read_csv('/home/vivente/Desktop/DR/patch_dataframe.csv')
#split_data(train_df=df, dest_path='/home/vivente/Desktop/DR/patch_dataframe_fold.csv')

train_hma()