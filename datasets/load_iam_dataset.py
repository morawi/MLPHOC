"""
Created on Thu Jul  7 12:08:22 2018

@author: malrawi

"""

from torch.utils.data import Dataset
import numpy as np
# from scripts.Word2PHOC import build_phoc as PHOC
from PIL import Image, ImageOps
from utils.some_functions import remove_non_words


# from torchvision import transforms
# import Augmentor

VGG_NORMAL = True
RM_BACKGROUND = True
FLIP = False # flip the image
OUTPUT_MAX_LEN = 23 # max-word length is 21  This value should be larger than 21+2 (<GO>+groundtruth+<END>)
IMG_WIDTH = 1011 # m01-084-07-00 max_length
IMG_HEIGHT = 64

#IMG_WIDTH = 256 # img_width < 256: padding   img_width > 256: resize to 256
''' IAM has 46945 train data; 7554 alid data; and 20304 test data '''
    
def get_iam_file_label(cf, mode):         
        subname = 'word' 
        
        if mode=='train':
            gt_tr = 'RWTH.iam_'+subname+'_gt_final.train.thresh'
            with open(cf.gt_path_IAM + gt_tr, 'r') as f_tr:
                data_tr = f_tr.readlines()
                file_label = [i[:-1].split(' ') for i in data_tr]
                
        elif mode == 'validate':
            gt_va = 'RWTH.iam_'+subname+'_gt_final.valid.thresh'
            with open(cf.gt_path_IAM +  gt_va, 'r') as f_va:
                data_va = f_va.readlines()
                file_label = [i[:-1].split(' ') for i in data_va]
                
        elif mode == 'test':
            gt_te = 'RWTH.iam_'+subname+'_gt_final.test.thresh'    
            with open(cf.gt_path_IAM + gt_te, 'r') as f_te:
                data_te = f_te.readlines()
                file_label = [i[:-1].split(' ') for i in data_te]
                
        file_label = remove_non_words(file_label)
        
        return file_label


    
def get_the_image(file_name, transform, cf):
   
    file_name, thresh = file_name.split(',')        
    thresh = int(thresh)    
    img_name = cf.dataset_path_IAM + file_name + '.png'        
    data = Image.open(img_name)     # data.show()
    data = data.point(lambda p: 255 if int(p < thresh) else 0 )   # thresholding   and inversion  
    data = data.convert('1')    # This is necessary to have all datasets in the same mode, it has to be befor the lambda function, for some reasone!
    # I have settled on using 255 as the max, even after convert(), max is 255, to convert to max 1, use: data = data.point(lambda p: 1 if p == 255  else 0 )  # even after convert-'1', 255 values are still there
    
    return data
    



class IAM_words(Dataset):
    def __init__(self, cf, mode='train', transform = None):
        # mode: 'train', 'validate', or 'test'        
        self.cf = cf
        self.mode = mode
        self.file_label = get_iam_file_label(self.cf, self.mode)        
        self.transform = transform
        self.len_phoc = len( self.cf.PHOC(word='abcd', cf = self.cf) ) # passing an arbitrary string to get the phoc lenght
        self.weights = np.ones( len(self.file_label) , dtype = 'uint8' )
               
        
    def __getitem__(self, index):
        word = self.file_label[index]  
        word_str = word[1].lower() # word_str = word[1].lower(); # to only keep lower-case       
        img = get_the_image(word[0], self.transform, self.cf)         
        if not(self.cf.H_iam_scale ==0): # resizing just the height            
            new_w = int(img.size[0]*self.cf.H_iam_scale/img.size[1])
            if new_w>self.cf.MAX_IMAGE_WIDTH: 
                new_w = self.cf.MAX_IMAGE_WIDTH
            img = img.resize( (new_w, self.cf.H_iam_scale), Image.ANTIALIAS)
        
        if self.cf.encoder=='label':
            target = cf.English_label # 1: lable for Arabic script
        else:
            # target = self.phoc_word[idx]
            target = self.cf.PHOC(word_str, cf = self.cf)
            target = self.cf.PHOC(word_str, cf = self.cf)  
        
        if self.transform:
            img = self.transform(img)    
        
        return img, target, word_str, self.weights[index]

    def __len__(self):
        return len(self.file_label)
     
    def num_classes(self):
        return self.len_phoc

   
     

#    # label, label_mask = label_padding(' '.join(word[1:]), self.output_max_len) iside the class
#    
#    ''' Auxliary functions '''
#def label_padding(labels, output_max_len):
#    new_label_len = []
#    the_labels = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
#    letter2index = {label: n for n, label in enumerate(the_labels)}        
#    tokens = {'GO_TOKEN': 0, 'END_TOKEN': 1, 'PAD_TOKEN': 2}
#    num_tokens = len(tokens.keys())
#    ll = [letter2index[i] for i in labels]
#    num = output_max_len - len(ll) - 2
#    new_label_len.append(len(ll)+2)
#    ll = np.array(ll) + num_tokens
#    ll = list(ll)
#    ll = [tokens['GO_TOKEN']] + ll + [tokens['END_TOKEN']]
#    if not num == 0:
#        ll.extend([tokens['PAD_TOKEN']] * num) # replace PAD_TOKEN
#
#    def make_weights(seq_lens, output_max_len):
#        new_out = []
#        for i in seq_lens:
#            ele = [1]*i + [0]*(output_max_len -i)
#            new_out.append(ele)
#        return new_out
#    return ll, make_weights(new_label_len, output_max_len)
#
