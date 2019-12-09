#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:53:48 2019

@author: malrawi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:07:33 2019

@author: malrawi
"""
import os
from PIL import Image
from torch.utils.data import Dataset


class Instagram_images(Dataset):
    def __init__(self, cf, transform=None):       
            
        self.cf = cf       
        self.folder_of_data = cf.dataset_path_InstagramHL                    
        self.transform = transform                                                 
        self.img_names = os.listdir(cf.dataset_path_InstagramHL)   
        self.weights = 0
     
        
    def __getitem__(self, index):
        img_name = self.img_names[index]
        word_str = self.img_names[index][:-4].lower() # removing .jpg from the name, the name is the text inside the image
        word_str = word_str.replace(" ", "")
        data = Image.open(self.cf.dataset_path_InstagramHL+img_name)                     
        
        if not(self.cf.H_Instagram_scale ==0): # resizing just the height            
            new_w = int(data.size[0]*self.cf.H_Instagram_scale/data.size[1])
            if new_w>self.cf.MAX_IMAGE_WIDTH: 
                new_w = self.cf.MAX_IMAGE_WIDTH
            data = data.resize( (new_w, self.cf.H_Instagram_scale), Image.ANTIALIAS)                                
                
        if self.transform is not None:
            data = self.transform(data)
        
        # target = self.cf.PHOC(word_str, cf = self.cf)    
        if self.cf.encoder_type=='script_identification':            
            target = self.cf.PHOC('English'.lower()+ 
                                  self.cf.language_hash_code['English'], self.cf) # language_name + hashcode
        else:            
            target = self.cf.PHOC(word_str, cf = self.cf)    
    
        return data, target, word_str, self.weights
    
    def num_classes(self):          
       return len(self.cf.PHOC('abcd', self.cf)) # pasing 'dump' word to get the length
        
    
    def __len__(self):
        return len(self.img_names)
    
#configuration = Configuration(config_path='config/config_file_wg.py', test_name='') # test_name: Optional name of the sub folder where to store the results
#cf = configuration.load(print_vals=True)
#x = Instagram_images(cf)
#id = np.random.randint(500)
#x[id][0].show()