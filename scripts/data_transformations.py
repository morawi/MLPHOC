from __future__ import print_function, division

#import glob
#import os
## from scripts.Word2PHOC import build_phoc as PHOC
#from utils import globals


import numpy as np
from torchvision import transforms
from skimage.morphology import thin as skimage_thinner
import Augmentor


#  Method to compute the padding odf the input image to the max image size
def get_padding(image, output_size):
    output_max_width = output_size[0]
    output_max_height = output_size[1]
    w, h = image.size

    pad_width = output_max_width - w
    if pad_width<0 : 
        print('MAX_IMAGE_WIDTH is smaller than expected, config_file'); 
        exit(0)
    
    if pad_width < 2:
        pad_left = pad_width
        pad_right = 0
    else:
        if pad_width % 2 == 0:
            pad_left = int(pad_width / 2)
            pad_right = pad_left
        else:
            pad_left = int(pad_width / 2) + 1
            pad_right = pad_left - 1
    
    pad_height = output_max_height - h    
    if pad_height<0 : print('MAX_IMAGE_WIDTH is smaller than expected, see config_file'); exit(0)
    if pad_height < 2:
        pad_top = pad_height
        pad_bottom = 0
    else:
        if pad_height % 2 == 0:
            pad_top = int(pad_height / 2)
            pad_bottom = pad_top
        else:
            pad_top = int(pad_height / 2) + 1
            pad_bottom = pad_top - 1

    padding = (pad_left, pad_top, pad_right, pad_bottom)
    return padding



# Class to perform the padding
class PadImage(object):
    """Pad the image in a sample to the max size

    Args:
        output_size (tuple or int): Desired output size.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_max_width = output_size[0]
        self.output_max_height = output_size[1]

    def __call__(self, image):
        padding = get_padding(image, (self.output_max_width, self.output_max_height))
        tsfm = transforms.Pad(padding)        
        image = tsfm(image)
        return image

# Class to perform the padding
class NoneTransform(object):
    """ Does nothing to the image, to be used instead of None
    
    Args:
        image in, image out, nothing is done
    """
        
    def __call__(self, image):       
        return image
    


def image_thinning(img, p):
    # input image as PIL, output image as PIL
    img= np.array(img).squeeze()
    thin_iter_step  = 1   
    img_max_orig = img.max()
    for i in range(25): 
        sum_img = img.sum()/(img.size* img.max())
        if sum_img>p:
            img = skimage_thinner(img, max_iter= thin_iter_step)
            ''' remove below maybe '''
        else: 
            # img = (img/img.max())
            break
    
    img = img.reshape(img.shape[0], img.shape[1], 1)
    img = img*img_max_orig   # Now, bringing the normalization back to all images
    img = img.astype('float32')
    tsfm = transforms.ToPILImage()
    img = tsfm(img)   
    
    return img 

class ImageThinning(object):
    """ Thin the image input as PIL and output a PIL
        To be used as part of  torchvision.transforms
    Args: p, a threshold value to determine the thinning
        
    """
    def __init__(self, p = 0.2):
       #  assert isinstance(output_size, (int, tuple))
        self.p = p                  
        
    def __call__(self, image):
        # image =self.image_thinning(image, self.p)                      
        image = image_thinning(image, self.p) 
        
        return image

class TheAugmentor(object):
    """ 
        Using the Augmentor module to distorte the text, see https://github.com/mdbloice/Augmentor
        
    """
    def __init__(self, probability=.5, grid_width=4, grid_height=4, magnitude=8):
       self.p = Augmentor.Pipeline()
       # self.p.random_distortion(probability, grid_width, grid_height, magnitude)
       self.p.shear(probability=.7, max_shear_left=5, max_shear_right=5)
       self.transform = transforms.Compose([self.p.torch_transform()])
        
    def __call__(self, image):        
        
        image = self.transform(image)
        if type(image)==list:
            image = image[0]
        image = np.array(image) #, dtype='float32') #for some reason augmentor returns a list  
        image = image > image.mean()
        image = image.astype('float32')
        image = image.reshape(image.shape[0], image.shape[1], 1)
        tsfm = transforms.ToPILImage()
        image = tsfm(image)   
           
        return image
