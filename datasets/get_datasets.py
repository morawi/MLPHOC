#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 22:40:10 2018

@author: malrawi
"""
from datasets.load_instagramHL_dataset import Instagram_images
from datasets.load_driver_dataset import SafeDriverDataset
from datasets.load_MLT_dataset import MLT_words
from datasets.load_washington_dataset import WashingtonDataset
from datasets.load_ifnenit_dataset import IfnEnitDataset
from datasets.load_WG_IFN_dataset import WG_IFN_Dataset
from datasets.load_iam_dataset import IAM_words
from datasets.load_IAM_IFN_dataset import IAM_IFN_Dataset
from datasets.load_tf_speech_recognition_dataset import TfSpeechDataset
from datasets.load_cifar100_dataset import Cifar100Dataset
from datasets.load_IFN_from_folders import IFN_XVAL_Dataset
from datasets.load_imdb_dataset import IMDB_dataset
from datasets.load_cub2011_dataset import Cub2011
from scripts.data_transformations import PadImage, ImageThinning, NoneTransform, OverlayImage
from sys import exit as sys_exit
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torch



''' The reason for having more than two transforms is that handwring images come with one channel'''
def get_transforms(cf):
    image_transform = {}
    mu = ((0.5),) * 3
    std = ((0.25),) * 3
    
    RGB_img = cf.overlay_handwritting_on_STL_img # one channel images will be replicated, but not RGBs
    
    # to be used with Handwriting data
    image_transform['image_transform_hdr'] = transforms.Compose([
            ImageThinning(p = cf.thinning_threshold) if cf.thinning_threshold < 1 else NoneTransform(),            
            OverlayImage(cf) if cf.overlay_handwritting_on_STL_img else NoneTransform(), # Add random image background here, to mimic scenetext, or, let's call it scenehandwritten            
            PadImage((cf.MAX_IMAGE_WIDTH, cf.MAX_IMAGE_HEIGHT)) if cf.pad_images else NoneTransform(),            
            transforms.Resize(cf.input_size) if cf.resize_images else NoneTransform(),            
            transforms.ToTensor(),            
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))  if not(RGB_img)  else NoneTransform(), # this is becuase the overlay, and Cifar100 produces an RGB image            
            transforms.Normalize( mu, std ) if cf.normalize_images else NoneTransform(),                                   
            ])    
        
    image_transform['image_transform_spch'] = transforms.Compose([            
            PadImage((cf.MAX_IMAGE_WIDTH, cf.MAX_IMAGE_HEIGHT)) if cf.pad_images else NoneTransform(),            
            transforms.Resize(cf.input_size) if cf.resize_images else NoneTransform(),            
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize( ((0.0),) * 3, (8, 8, 8) ) if cf.normalize_images else NoneTransform(),                                   
            # mu might need to be replaced, as 0.25 is not a correct mean!
            ])
   
    image_transform['image_transform_imdb'] = transforms.Compose([            
            PadImage((cf.MAX_IMAGE_WIDTH, cf.MAX_IMAGE_HEIGHT)) if cf.pad_images else NoneTransform(),            
            transforms.Resize(cf.input_size) if cf.resize_images else NoneTransform(),            
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),            
            transforms.Normalize( ((0.0),) * 3, ((.5),) * 3 ) if cf.normalize_images else NoneTransform(),                                   
            ])    

    image_transform['image_transform_scn'] = transforms.Compose([            
            PadImage((cf.MAX_IMAGE_WIDTH, cf.MAX_IMAGE_HEIGHT)) if cf.pad_images else NoneTransform(),            
            transforms.Resize(cf.input_size) if cf.resize_images else NoneTransform(),            
            transforms.ToTensor(),
            transforms.Normalize( mu, std ) if cf.normalize_images else NoneTransform(),                                   
            ])

     
    
    
    
    return image_transform
    # sheer_tsfm = transforms.RandomAffine(0, shear=(-30, 10) ) 
    # random_sheer = transforms.RandomApply([sheer_tsfm], p=0.7) # will only be used if cf.use_distortion_augmentor is True
    # can be used in the transform random_sheer if cf.use_distortion_augmentor else NoneTransform(),                       
    

def get_cub2011(cf, image_transform):
    
    train_set = Cub2011(cf, root='/home/malrawi/Desktop/My Programs/all_data/', train=True, 
                        transform=image_transform['image_transform_scn'])
    test_set = Cub2011(cf, root='/home/malrawi/Desktop/My Programs/all_data/', train=False,
                       transform=image_transform['image_transform_scn'])
    return train_set, test_set

def get_mlt(cf, image_transform):
    print('...................Loading MLT dataset...................')        
    
    ''' randomly split training and testing according to split percentage  '''
    train_set = MLT_words(cf, train=True, transform=image_transform['image_transform_scn'])
    test_set = MLT_words(cf, train=False, transform=image_transform['image_transform_scn'], 
                        data_idx = train_set.data_idx)      
        
    return train_set, test_set


def get_safe_driver(cf, image_transform):
     print('...................Safe Driver dataset...................')
     train_set = SafeDriverDataset(cf, transform = image_transform['image_transform_scn'])
     test_set = SafeDriverDataset(cf, train=False, data_idx = train_set.data_idx, 
                                  transform = image_transform['image_transform_scn'])
     
     return train_set, test_set
     
def get_imdb(cf, image_transform):
     print('...Loading................Large Image Movie Dataset (IMDB)...................')
     train_set = IMDB_dataset(cf, mode='train', transform = image_transform['image_transform_imdb']) # should use mode as 'train'
     test_set = IMDB_dataset(cf, mode='test', transform = image_transform['image_transform_imdb']) # should use mode as 'test'
     print('.....Loading .. IMDB done')
     
     return train_set, test_set

def get_cifar100(cf, image_transform):
    print('...................Cifar100 dataset...................')
    train_set = Cifar100Dataset(cf, 'train', transform=image_transform['image_transform_scn']) 
    test_set = Cifar100Dataset(cf, 'test', transform=image_transform['image_transform_scn'])  
    return train_set, test_set
    
def get_tf_speech(cf, image_transform):
    print('...................Loading TensorFlow Speech Recog dataset...................')
    train_set = TfSpeechDataset(cf, train=True, transform=image_transform['image_transform_spch'])
    test_set = TfSpeechDataset(cf, train=False, transform=image_transform['image_transform_spch'])
#    test_set = TfSpeechDataset(cf, train=False, transform=image_transform['image_transform_spch'], 
#                        data_idx =train_set.data_idx, complement_idx = True)        
#    test_set_from_folder = TfSpeechDataset(cf, train=False, transform=image_transform['image_transform_spch']) # we need to set train=true, so that split does not reduce the data we are using    
    # combining test_set from the split and test_set_from_folder
#    test_set =  torch.utils.data.ConcatDataset( [test_set,  test_set_from_folder] )
    
    return train_set, test_set, test_set # as this is 

def get_gw(cf, image_transform):
    print('...................Loading WG dataset...................')
    train_set = WashingtonDataset(cf, train=True, transform=image_transform['image_transform_hdr'])
    test_set = WashingtonDataset(cf, train=False, transform=image_transform['image_transform_hdr'], 
                        data_idx = train_set.data_idx, complement_idx = True)
    
    return train_set, test_set



def get_ifn(cf, image_transform):
    print('...................Loading IFN dataset...................')        
    if not(cf.IFN_based_on_folds_experiment):
        ''' randomly split training and testing according to split percentage 
        the folder left for tesing is the cf.IFN_test '''
        train_set = IfnEnitDataset(cf, train=True, transform=image_transform['image_transform_hdr'])
        test_set = IfnEnitDataset(cf, train=False, transform=image_transform['image_transform_hdr'], 
                            data_idx = train_set.data_idx, complement_idx = True)            
    else:
        ''' leave one folder out of 'abcde' folders '''
        train_set = IFN_XVAL_Dataset(cf, train=True, transform = image_transform['image_transform_hdr'])
        test_set = IfnEnitDataset(cf, train=False, transform = image_transform['image_transform_hdr'])
    return train_set, test_set

def get_wg_ifn(cf, image_transform):
    print('...................IFN & WG datasets ---- The multi-lingual PHOCNET')        
    train_set_gw, test_set_gw = get_gw(cf, image_transform)
    train_set_ifn, test_set_ifn = get_ifn(cf, image_transform)  
    train_set = torch.utils.data.ConcatDataset( [train_set_gw, train_set_ifn])
    test_set = torch.utils.data.ConcatDataset( [test_set_gw, test_set_ifn])
    return train_set, test_set, test_set_gw, test_set_ifn



def get_iam(cf, image_transform):
    print('...................Loading IAM dataset...................') 
    validate_set = IAM_words(cf, mode='validate', transform = image_transform['image_transform_hdr'])        
    train_set = IAM_words(cf, mode='train', transform = image_transform['image_transform_hdr'])    
    print('\n Concatnating IAM validate and train sets')
    train_set = torch.utils.data.ConcatDataset( [train_set, validate_set])
    del validate_set
    # train_set = iam_train_valid_combined_dataset(cf, train=True, transform = image_transform['image_transform_hdr']) # this merges train and validate into one
    test_set = IAM_words(cf, mode='test', transform = image_transform['image_transform_hdr']) 
    return train_set, test_set




def get_iam_ifn(cf, image_transform):
    print('................... IAM & IFN datasets ---- The multi-lingual PHOCNET')    

    train_set_ifn, test_set_ifn = get_ifn(cf, image_transform)    
    train_set_iam, test_set_iam = get_iam(cf, image_transform)  
    train_set = torch.utils.data.ConcatDataset( [train_set_iam, train_set_ifn])
    test_set = torch.utils.data.ConcatDataset( [test_set_iam, test_set_ifn])
    
    return train_set, test_set, test_set_iam, test_set_ifn




def get_datasets(cf, image_transform):
        
    test_per_data = {}
    
    if cf.dataset_name == 'Cifar100+TFSPCH+IAM+IFN+safe-driver+imdb+cub2011':
        train_set_cifar100, test_per_data['test_set_cifar100'] = get_cifar100(cf, image_transform)      
        train_set_tfspch, test_set_tfspch, test_per_data['test_set_TFSPCH'] = get_tf_speech(cf, image_transform)
        train_set_iam_ifn, test_set_iam_ifn, test_per_data['test_set_iam'], test_per_data['test_set_ifn'] = get_iam_ifn(cf, image_transform)
        train_set_sf_drive, test_per_data['test_set_safe_driver'] = get_safe_driver(cf, image_transform)
        
        train_set_imdb, test_per_data['test_set_imdb']  = get_imdb(cf, image_transform)
        train_set_cub2011, test_per_data['cub2011'] = get_cub2011(cf, image_transform)         
        
        train_set = torch.utils.data.ConcatDataset( [train_set_cifar100, 
                                                     train_set_imdb,
                                                     train_set_tfspch, 
                                                     train_set_iam_ifn, 
                                                     train_set_sf_drive, 
                                                     train_set_cub2011] )
        test_set = torch.utils.data.ConcatDataset( [test_per_data['test_set_cifar100'], 
                                                    test_per_data['test_set_imdb'],
                                                    test_set_tfspch, 
                                                    test_set_iam_ifn, 
                                                    test_per_data['test_set_safe_driver'],
                                                    test_per_data['cub2011']
                                                    ] )
    
        del train_set_cifar100, train_set_tfspch, test_set_tfspch, 
        train_set_iam_ifn, test_set_iam_ifn, train_set_sf_drive, 
        train_set_imdb, train_set_cub2011
        
   
    elif cf.dataset_name == 'Cifar100+TFSPCH+IAM+IFN+safe-driver+imdb+cub2011+MLT':
        train_set_cifar100, test_per_data['test_set_cifar100'] = get_cifar100(cf, image_transform)      
        train_set_tfspch, test_set_tfspch, test_per_data['test_set_TFSPCH'] = get_tf_speech(cf, image_transform)
        train_set_iam_ifn, test_set_iam_ifn, test_per_data['test_set_iam'], test_per_data['test_set_ifn'] = get_iam_ifn(cf, image_transform)
        train_set_sf_drive, test_per_data['test_set_safe_driver'] = get_safe_driver(cf, image_transform)
        
        train_set_imdb, test_per_data['test_set_imdb']  = get_imdb(cf, image_transform)
        train_set_cub2011, test_per_data['cub2011'] = get_cub2011(cf, image_transform)  
        train_set_MLT, test_per_data['test_set_MLT'] = get_mlt(cf, image_transform)  
        
        train_set = torch.utils.data.ConcatDataset( [train_set_cifar100, 
                                                     train_set_imdb,
                                                     train_set_tfspch, 
                                                     train_set_iam_ifn, 
                                                     train_set_sf_drive, 
                                                     train_set_cub2011, 
                                                     train_set_MLT] )
        test_set = torch.utils.data.ConcatDataset( [test_per_data['test_set_cifar100'], 
                                                    test_per_data['test_set_imdb'],
                                                    test_set_tfspch, 
                                                    test_set_iam_ifn, 
                                                    test_per_data['test_set_safe_driver'],
                                                    test_per_data['cub2011'],
                                                    test_per_data['test_set_MLT']
                                                    ] )
       
        del train_set_cifar100, train_set_tfspch, test_set_tfspch, 
        train_set_iam_ifn, test_set_iam_ifn, train_set_sf_drive, 
        train_set_imdb, train_set_cub2011, train_set_MLT     
    
    elif cf.dataset_name == 'MLT':
        train_set, test_set = get_mlt(cf, image_transform)           
        test_per_data['test_set_MLT'] = test_set   
        
        if 'Instagram_test' in cf.MLT_lang:            
            train_set_gw, test_set_gw = get_gw(cf, image_transform)    
            test_set_iam = IAM_words(cf, mode='test', transform = image_transform['image_transform_hdr']) 
            test_per_data['test_Instagram']  = Instagram_images(cf, image_transform['image_transform_scn'])
            test_per_data['test_set_gw'] = test_set_gw
            test_per_data['test_set_iam'] = test_set_iam
            train_set = torch.utils.data.ConcatDataset( [train_set, train_set_gw] )
            test_set = torch.utils.data.ConcatDataset( [test_set_gw] ) # only GW will be used for the overall QbS QbE as the target is instagram
                                  
    
        
    elif cf.dataset_name == 'Cifar100+TFSPCH+IAM+IFN+safe-driver+imdb':
        train_set_cifar100, test_per_data['test_set_cifar100'] = get_cifar100(cf, image_transform)      
        train_set_tfspch, test_set_tfspch, test_per_data['test_set_TFSPCH'] = get_tf_speech(cf, image_transform)
        train_set_iam_ifn, test_set_iam_ifn, test_per_data['test_set_iam'], test_per_data['test_set_ifn'] = get_iam_ifn(cf, image_transform)
        train_set_sf_drive, test_per_data['test_set_safe_driver'] = get_safe_driver(cf, image_transform)
        
        train_set_imdb, test_per_data['test_set_imdb']  = get_imdb(cf, image_transform)
         
        
        train_set = torch.utils.data.ConcatDataset( [train_set_cifar100, train_set_imdb,
                                                     train_set_tfspch, train_set_iam_ifn, train_set_sf_drive] )
        test_set = torch.utils.data.ConcatDataset( [test_per_data['test_set_cifar100'], test_per_data['test_set_imdb'],
                                                    test_set_tfspch, test_set_iam_ifn, test_per_data['test_set_safe_driver']] )
     # Delete all but keep train_set, test_set, test_per_data

    
    elif cf.dataset_name == 'Cifar100+TFSPCH+IAM+IFN+safe-driver':
        train_set_cifar100, test_per_data['test_set_cifar100'] = get_cifar100(cf, image_transform)      
        train_set_tfspch, test_set_tfspch, test_per_data['test_set_TFSPCH'] = get_tf_speech(cf, image_transform)
        train_set_iam_ifn, test_set_iam_ifn, test_per_data['test_set_iam'], test_per_data['test_set_ifn'] = get_iam_ifn(cf, image_transform)
        train_set_sf_drive, test_per_data['test_set_safe_driver'] = get_safe_driver(cf, image_transform)
        
        train_set = torch.utils.data.ConcatDataset( [train_set_cifar100, 
                                                     train_set_tfspch, train_set_iam_ifn, train_set_sf_drive] )
        test_set = torch.utils.data.ConcatDataset( [test_per_data['test_set_cifar100'], 
                                                    test_set_tfspch, test_set_iam_ifn, test_per_data['test_set_safe_driver']] )
    
    elif cf.dataset_name == 'safe_driver':
        train_set, test_set = get_safe_driver(cf, image_transform)        
        test_per_data['test_set_safe_driver'] = test_set
                
    elif cf.dataset_name == 'Cifar100+TFSPCH+IAM+IFN':
        train_set_cifar100, test_per_data['test_set_cifar100'] = get_cifar100(cf, image_transform)      
        train_set_tfspch, test_set_tfspch, test_per_data['test_set_TFSPCH'] = get_tf_speech(cf, image_transform)
        train_set_iam_ifn, test_set_iam_ifn, test_per_data['test_set_iam'], test_per_data['test_set_ifn'] = get_iam_ifn(cf, image_transform)
        
        train_set = torch.utils.data.ConcatDataset( [train_set_cifar100, train_set_tfspch, train_set_iam_ifn] )
        test_set = torch.utils.data.ConcatDataset( [test_per_data['test_set_cifar100'], test_set_tfspch, test_set_iam_ifn] )
    
    elif cf.dataset_name == 'Cifar100+TFSPCH+GW+IFN':
        train_set_cifar100, test_per_data['test_set_cifar100'] = get_cifar100(cf, image_transform)      
        train_set_tfspch, test_set_tfspch, test_per_data['test_set_TFSPCH'] = get_tf_speech(cf, image_transform)
        train_set_gw_ifn, test_set_gw_ifn, test_per_data['test_set_wg'], test_per_data['test_set_ifn'] = get_wg_ifn(cf, image_transform)
        
        train_set = torch.utils.data.ConcatDataset( [train_set_cifar100, train_set_tfspch, train_set_gw_ifn] )
        test_set = torch.utils.data.ConcatDataset( [test_per_data['test_set_cifar100'], test_set_tfspch, test_set_gw_ifn] )
               
        
    elif cf.dataset_name == 'Cifar100':
        train_set, test_set = get_cifar100(cf, image_transform)  
        test_per_data['test_set_cifar100'] = test_set 
    
    elif cf.dataset_name == 'TFSPCH':
        train_set, test_set, test_per_data['test_set_TFSPCH'] = get_tf_speech(cf, image_transform)
    
    elif cf.dataset_name == 'WG':
        train_set, test_set = get_gw(cf, image_transform)    
        test_per_data['test_set_gw'] = test_set 
        
    elif cf.dataset_name =='IAM':
        train_set, test_set = get_iam(cf, image_transform) 
        test_per_data['test_set_iam'] = test_set 

    elif cf.dataset_name == 'IFN':
        train_set, test_set = get_ifn(cf, image_transform)        
        test_per_data['test_set_ifn'] = test_set 
                    
    elif cf.dataset_name =='WG+IFN': 
        train_set, test_set, test_per_data['test_set_wg'], test_per_data['test_set_ifn'] = get_wg_ifn(cf, image_transform)
               
        
    elif cf.dataset_name =='IAM+IFN': 
        train_set, test_set, test_per_data['test_set_iam'], test_per_data['test_set_ifn'] = get_iam_ifn(cf, image_transform)
    
    elif cf.dataset_name == 'cub2011':
        train_set, test_set = get_cub2011(cf, image_transform)    
        test_per_data['test_set_cub2011'] = test_set 
        
    elif cf.dataset_name == 'imdb_movie':
        train_set, test_set  = get_imdb(cf, image_transform)
        test_per_data['test_set_imdb'] = test_set     
        
    else:
        print('Please select correct dataset name, one of dataset_name in config_file_wg.py')
        sys_exit('Incorrect dataset name')
    
    display_images(train_set, test_set)
       
    
    return train_set, test_set, test_per_data

def display_img(img, str_v):
    to_pil = transforms.ToPILImage() 
    img1  = to_pil(img) # we can also use test_set[1121][0].numpy()    
    plt.axis('off')
    plt.imshow(np.array(img1).squeeze()); plt.show()
    print('one ', str_v, ' image: has min-max vals', img.min(), img.max())  


def display_images(train_set, test_set):
    ''' Diagnostics: displaying images for overlay diagnostics, or see if they are correctly formated '''
    display_img(train_set[len(train_set)//3][0], 'train') # 41, some trivial index     
    display_img(test_set[len(test_set)-10][0], 'test')    # 1121, some trivial index
    display_img(train_set[len(train_set)//4][0], 'train') # 41, some trivial index     
    display_img(test_set[len(test_set)//2][0], 'test')    # 1121, some trivial index    
    print(train_set[41][0].shape)
    
def get_sampled_loader(cf, test_set):
        no_of_samples  = len(test_set)
        sample_idx = np.random.permutation(np.arange(0, no_of_samples))[:cf.no_of_sampled_data]                     
        if len(sample_idx) ==0:  
            exit('exiting function get_the_sampler(), sample_idx size is 0')    
        my_sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_idx)  
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=cf.batch_size_test,
                          shuffle= False, num_workers=cf.num_workers, sampler=my_sampler)
        return test_loader
    

def get_dataloaders(cf, train_set, test_set, test_per_data):
        
    # the main loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cf.batch_size_train,
                                  shuffle = cf.shuffle, num_workers=cf.num_workers)
    
    if not cf.sampled_testing: # chose this when RAM (memory) is limited
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=cf.batch_size_test,
                                  shuffle = False, num_workers=cf.num_workers)
    else:
        test_loader = get_sampled_loader(cf, test_set)
   
    # the per-script/dataset test loaders
    per_data_loader= {}
    for item in test_per_data.items():
        per_data_loader[item[0]] = torch.utils.data.DataLoader(test_per_data[item[0]], 
                            batch_size=cf.batch_size_test, shuffle = False, num_workers=cf.num_workers)
    
    return train_loader, test_loader, per_data_loader



