"""
Created on Thu Jul  7 12:08:22 2018

@author: malrawi

"""

from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import time
import re

# from torchvision import transforms
# import Augmentor


OUTPUT_MAX_LEN = 23 # max-word length is 21  This value should be larger than 21+2 (<GO>+groundtruth+<END>)

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
                
        # entries of file_lable are: ['c04-160-06-05,171', 'viewers'], ['c04-160-06-06,171', 'complained'], ['c04-160-06-07,171', 'to']]
        x_file = []
        for item in file_label:
            word = item[1]            
            if len(word)>2 or re.search('[a-zA-Z]', word) != None:  #  removing less than size 2 words,      has_alphabets = re.search('[a-zA-Z]', word_string) # if at leas has one alphabet
                x_file.append(item)
            
        return x_file


    
def get_the_image(file_name, cf):
   
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
        self.weights = np.ones( len(self.file_label) , dtype = 'uint8' )
        self.PHOC_vector = torch.empty(len(self.file_label), len(self.cf.PHOC('dump', self.cf)) , dtype=torch.float)
        print('\n Storing PHOCs for ', mode  ); # print(end='')  
        time.sleep(1)
        pbar = tqdm(total=len(self.file_label));  
        time.sleep(1)
        for i in range(len(self.file_label)):
            word = self.file_label[i]  
            word_str = word[1].lower() 
            self.PHOC_vector[i] = torch.from_numpy(self.cf.PHOC(word_str, self.cf))
            pbar.update(1)        
        pbar.close();   del pbar 
               
        
    def __getitem__(self, index):
        word = self.file_label[index]  
        word_str = word[1].lower() # word_str = word[1].lower(); # to only keep lower-case       
        img = get_the_image(word[0], self.cf)         
        if not(self.cf.H_iam_scale ==0): # resizing just the height             
            new_w = int(img.size[0]*self.cf.H_iam_scale/img.size[1])
            if new_w>self.cf.MAX_IMAGE_WIDTH: 
                new_w = self.cf.MAX_IMAGE_WIDTH
            img = img.resize( (new_w, self.cf.H_iam_scale), Image.ANTIALIAS)
        
        if self.cf.task_type=='script_identification':
            # target = self.cf.English_label # 1: lable for Arabic script
            target = self.cf.PHOC('English'.lower()+ 
                                  self.cf.language_hash_code['English'], self.cf, 
                                  mode = 'hand-writing') # language_name + hashcode
        else:            
            target = self.PHOC_vector[index]
                    
        if self.transform:
            img = self.transform(img)    
        
        return img, target, word_str, self.weights[index]

    def __len__(self):
        return len(self.file_label)
     
    def num_classes(self):
        if self.cf.task_type=='script_identification':
            # return len(self.cf.English_label)
            return len(self.cf.PHOC('dump', self.cf)) # pasing 'dump' word to get the length
        else:
            return len(self.cf.PHOC('dump', self.cf)) # pasing 'dump' word to get the length



   
     

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
