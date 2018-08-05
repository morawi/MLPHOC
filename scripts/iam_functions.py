#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 23:31:51 2018

@author: malrawi
"""

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
        
        # np.random.shuffle(file_label_tr)
        
        return file_label


''' Auxliary functions '''
def label_padding(labels, output_max_len):
    new_label_len = []
    the_labels = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    letter2index = {label: n for n, label in enumerate(the_labels)}        
    tokens = {'GO_TOKEN': 0, 'END_TOKEN': 1, 'PAD_TOKEN': 2}
    num_tokens = len(tokens.keys())
    ll = [letter2index[i] for i in labels]
    num = output_max_len - len(ll) - 2
    new_label_len.append(len(ll)+2)
    ll = np.array(ll) + num_tokens
    ll = list(ll)
    ll = [tokens['GO_TOKEN']] + ll + [tokens['END_TOKEN']]
    if not num == 0:
        ll.extend([tokens['PAD_TOKEN']] * num) # replace PAD_TOKEN

    def make_weights(seq_lens, output_max_len):
        new_out = []
        for i in seq_lens:
            ele = [1]*i + [0]*(output_max_len -i)
            new_out.append(ele)
        return new_out
    return ll, make_weights(new_label_len, output_max_len)



def readImage_keepRatio(file_name, flip, augmentation, transformer, cf):
    if RM_BACKGROUND:
        file_name, thresh = file_name.split(',')
        thresh = int(thresh)        
        # subdir = 'words/'
    url = cf.dataset_path_IAM + file_name + '.png'
    img = cv2.imread(url, 0)
    if RM_BACKGROUND:
        img[img>thresh] = 255
    
    rate = float(IMG_HEIGHT) / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1]*rate)+1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC) # INTER_AREA con error
    # c04-066-01-08.png 4*3, for too small images do not augment
    if augmentation: # augmentation for training data
        img_new = transformer(img)
        if img_new.shape[0] != 0 and img_new.shape[1] != 0:
            rate = float(IMG_HEIGHT) / img_new.shape[0]
            img = cv2.resize(img_new, (int(img_new.shape[1]*rate)+1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC) # INTER_AREA con error
        else:
            img = 255 - img
    else:
        img = 255 - img

    img_width = img.shape[-1]

    if flip: # because of using pack_padded_sequence, first flip, then pad it
        img = np.flip(img, 1)

    if img_width > IMG_WIDTH:
        outImg = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        #outImg = img[:, :IMG_WIDTH]
        img_width = IMG_WIDTH
    else:
        outImg = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='uint8')
        outImg[:, :img_width] = img
    outImg = outImg/255. #float64
    outImg = outImg.astype('float32')
    if VGG_NORMAL:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        outImgFinal = np.zeros([3, *outImg.shape])
        for i in range(3):
            outImgFinal[i] = (outImg - mean[i]) / std[i]
        return outImgFinal, img_width

    outImg = np.vstack([np.expand_dims(outImg, 0)] * 3) # GRAY->RGB
    return outImg, img_width


