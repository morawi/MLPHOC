''' 
Main @author: malrawi 

'''


from __future__ import print_function, division

#import glob
#import os
## from scripts.Word2PHOC import build_phoc as PHOC
#from utils import globals


import numpy as np
from torchvision import transforms
from skimage.morphology import thin as skimage_thinner
import Augmentor
from PIL import Image # ImageChops,  ImageOps, ImageStat
import torchvision

#  Method to compute the padding odf the input image to the max image size
def get_padding(image, output_size):
    output_max_width = output_size[0]
    output_max_height = output_size[1]
    w, h = image.size

    pad_width = output_max_width - w
    if pad_width<0:
        print(output_max_width, w) 
        print('MAX_IMAGE_WIDTH is smaller than expected, check config_file'); 
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
    if pad_height<0 : 
        print(output_max_height, h) 
        print('MAX_IMAGE_WIDTH is smaller than expected, see config_file'); exit(0)
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
    thin_iter_step  = 1   
    max_no_of_thin_iterations = 25 # the algorithm will mostly used 2 or 3, as shown in our tests    
    img_max_orig = img.getextrema()[1] # we need this to normalize the image back to the max value
    img= np.array(img).squeeze() # the img max will be 1 now, which is suitable for our thinning algorithm
    sum_img = img.sum()/(img.size* img.max())   
    for i in range(max_no_of_thin_iterations): 
        sum_img = img.sum()/(img.size* img.max())
        if sum_img>p:
            img = skimage_thinner(img, max_iter= thin_iter_step)            
        else:           
            break        
    
    img = img.reshape(img.shape[0], img.shape[1], 1)    
    img = img*img_max_orig   # Now, bringing the normalization back to all images    
    img = img.astype('float32')
    tsfm = transforms.ToPILImage()
    img = tsfm(img)   
    img = img.convert('1')  # image will have max value (even 255) after this conversion, looks crazey but it had to be done!!
        
    return img  # the value 1 is used as a flag that there was thinning

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
#        if x == 0:
#            image = y
#            
       
        
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


class OverlayImage(object):
    """ 
        Use an image from dataset as background to the handwritting image (STL10 dataset is used)
        
    """
    def __init__(self, cf):
       self.cf=cf     
       self.dataset = self.get_the_data('STL10')
       
       
    def get_the_data(self, data_set_name):
        # data_set_name = 'STL10'
        print(data_set_name, ' ', end='')
        # other split flags: ‘train’, 'test' ‘train+unlabeled’
        unlabeled_set = torchvision.datasets.STL10(root = self.cf.stl100_path, split= 'unlabeled', download=True, transform=None, target_transform = None )
        return unlabeled_set

    def stitch_images(self, hand_wrt_img):        
                
        intended_w, intded_h = hand_wrt_img.size
        w, h = self.dataset[0][0].size# .resize([128, 128], Image.ANTIALIAS) # rerurns a tuple, image at idx 0, and label at idx 1
        no_of_images_to_stich = intended_w // w
        if no_of_images_to_stich ==0:  no_of_images_to_stich = 1
        imgs_idx = np.random.randint(0, len(self.dataset), no_of_images_to_stich) # idx of selected images
        stiched_image = Image.new("RGB", (no_of_images_to_stich*w, h))        
        for index in range(no_of_images_to_stich):  
          img = self.dataset[imgs_idx[index]][0] # .resize([128, 128], Image.ANTIALIAS)  
          x = index * w    
          stiched_image.paste(img, (x , 0, x + w , h))        
        stiched_image = stiched_image.resize([intended_w,intded_h], Image.ANTIALIAS) 
        my_mask = hand_wrt_img.convert('1')
        hand_wrt_img = hand_wrt_img.convert('RGB')
        if self.cf.change_hand_wrt_color:
            rr, gg, bb = hand_wrt_img.split()
            rr = rr.point(lambda p: 0 if p==0 else np.random.randint(256) )
            gg = gg.point(lambda p: 0 if p==0 else np.random.randint(256) )
            bb = bb.point(lambda p: 0 if p==0 else np.random.randint(256) )
            hand_wrt_img = Image.merge("RGB", (rr, gg, bb))  
            my_mask = hand_wrt_img.convert('1') # comment this to remove gaps from the handwriting
        else:
            # keep it white
            hand_wrt_img = hand_wrt_img.point(lambda p: 0 if p==0 else 255)
        
        stiched_image.paste(hand_wrt_img , box=None, mask = my_mask)
                
        
        return stiched_image
    # mean_val = sum(ImageStat.Stat(stiched_image).mean)/3
    # stiched_image.paste(ImageOps.invert(hand_wrt_img), box=None, mask=hand_wrt_img.convert('1'))
    
    def __call__(self, image): 
        image = self.stitch_images(image)
       
        return image
    
    
