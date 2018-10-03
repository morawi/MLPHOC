#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 14:10:07 2018

@author: malrawi
"""

"""
Structure Based on:
- https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
- https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from cnn_finetune import make_model

from datasets.load_WG_IFN_dataset import WG_IFN_Dataset
from scripts.data_transformations import PadImage, ImageThinning, NoneTransform, TheAugmentor



def script_recognition(cf):
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
#    the_augmentor = TheAugmentor(probability=.5, grid_width=3, 
#                                 grid_height=3, magnitude=8)

    sheer_tsfm = transforms.RandomAffine(0, shear=(-30, 10) ) 
    random_sheer = transforms.RandomApply([sheer_tsfm], p=0.7) # will only be used if cf.use_distortion_augmentor is True

    
    image_transform = transforms.Compose([
            ImageThinning(p = cf.thinning_threshold) if cf.thinning_threshold < 1 else NoneTransform(),            
            random_sheer if cf.use_distortion_augmentor else NoneTransform(),            
            PadImage((cf.MAX_IMAGE_WIDTH, cf.MAX_IMAGE_HEIGHT)) if cf.pad_images else NoneTransform(),
            transforms.Scale(cf.input_size) if cf.resize_images else NoneTransform(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize( (0.5, 0.5, 0.5), (0.25, 0.25 , 0.25) ) if cf.normalize_images else NoneTransform(),
            ])
#        
                    
    if cf.dataset_name =='WG+IFN': 
        print('...................IFN & WG datasets ---- The multi-lingual PHOCNET')        
        
        train_set = WG_IFN_Dataset(cf, train=True, transform = image_transform)
        test_set = WG_IFN_Dataset(cf, train=False, transform = image_transform, 
                                  data_idx_WG = train_set.data_idx_WG, 
                                  data_idx_IFN = train_set.data_idx_IFN, 
                                        complement_idx = True)
    else: 
        exit('only works for WG+IFN script recognition')
            
        
        # plt.imshow(train_set[29][0], cmap='gray'); plt.show()
               
    if cf.use_weight_to_balance_data: 
        print('Adding weights to balance the data')
        # train_set = add_weights_of_words(train_set)
        train_set.add_weights_of_words()
        
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cf.batch_size_train,
                                  shuffle = cf.shuffle, num_workers=cf.num_workers)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=cf.batch_size_test,
                                  shuffle = False, num_workers=cf.num_workers)
   

    model = make_model(
        cf.model_name,
        pretrained = cf.pretrained,
        num_classes = train_set.num_classes(),
        input_size = cf.input_size, 
        dropout_p = cf.dropout_probability,
    )
    model = model.to(device)
   
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD( model.parameters(), 
                          lr = cf.learning_rate, 
                          momentum = cf.momentum,
                          nesterov = cf.use_nestrov_moment,
                          weight_decay = cf.weight_decay,
                          dampening = cf.damp_moment if not(cf.use_nestrov_moment) else 0
                          )    

    def train(epoch):
        total_loss = 0
        total_size = 0
        model.train()
        for batch_idx, (data, target, word_str, weight) in enumerate(train_loader):
            data, target = data.to(device), target.to(device) 
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            total_size += data.size(0)
            loss.backward()
            optimizer.step()
            if batch_idx % cf.batch_log == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.7f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), total_loss / total_size))


    def test(test_loader):               
        
        model.eval()        
        test_loss = 0
        correct = 0
        pred_all = torch.tensor([], dtype=torch.float32, device=device)
        target_all = torch.tensor([], dtype=torch.float32, device=device)
        word_str_all = ()
        with torch.no_grad():
            for data, target, word_str, weight in test_loader: # weight is trivial here
                data, target = data.to(device), target.to(device)
                output = model(data)
                ''' loss = criterion(output.float(), target.float())                
                loss = criterion(output, target )
                test_loss += loss.item() 
                '''
                output = F.sigmoid(output)
                target = target.type(torch.cuda.LongTensor)
                pred = output.data.max(1, keepdim=True)[1]
                # pred = pred.type(torch.cuda.DoubleTensor)
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()                
               
                
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset)*pred.size()[1],
            100. * correct / (len(test_loader.dataset)*pred.size()[1] )))   
        
        return 0 # to be used in case we want to try different distances later
        
                
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cf.lr_milestones , gamma= cf.lr_gamma) 
    # print('PHOC length', train_set.)
    print('Chance level performance \n');  
    test(test_loader) # nice to know the performance prior to training
    for epoch in range(1, cf.epochs + 1):
        scheduler.step();  
        print("lr = ", scheduler.get_lr(), end="") # to be used with MultiStepLR
        train(epoch)           
        if not(epoch % cf.testing_print_frequency):
            test(test_loader)
            
    result = test(test_loader)
    
    return result, train_set, test_set, train_loader, test_loader # returned to be checked from command console, this is provisary
    
   
