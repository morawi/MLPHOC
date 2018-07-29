from torch.utils.data import Dataset
import numpy as np
from scripts.Word2PHOC import build_phoc as PHOC
from PIL import Image, ImageOps


# from torchvision import transforms
# import Augmentor

VGG_NORMAL = True
RM_BACKGROUND = True
FLIP = False # flip the image
OUTPUT_MAX_LEN = 23 # max-word length is 21  This value should be larger than 21+2 (<GO>+groundtruth+<END>)
IMG_WIDTH = 1011 # m01-084-07-00 max_length
IMG_HEIGHT = 64

#IMG_WIDTH = 256 # img_width < 256: padding   img_width > 256: resize to 256
''' IAM has 46945 train data; 6445 valid data; and 13752 test data '''
    
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


def get_the_image(file_name, transform, cf):
   
    file_name, thresh = file_name.split(',')        
    thresh = int(thresh)    
    img_name = cf.dataset_path_IAM + file_name + '.png'        
    data = Image.open(img_name)    
    data = data.point(lambda p: p > thresh and 255) # threshold the image
    data = ImageOps.invert(data)    # Invert the input image 
    # data.show()
    # data = data.convert('L') 
    # Convert data to numpy array, so that we use it as input to transform    
    data = np.array(data.getdata(),
                np.uint8).reshape(data.size[1], data.size[0], 1)
    
    if transform:
        data = transform(data)
           
    return data
    


class IAM_words(Dataset):
    def __init__(self, cf, mode='train', transform = None, augmentation=True ):
        # mode: 'train', 'validate', or 'test'
        #def __init__(self, cf, train=True, transform=None, data_idx = np.arange(1), complement_idx=False):
        self.cf = cf
        self.mode = mode
        self.file_label = get_iam_file_label(self.cf, self.mode)
        self.output_max_len = OUTPUT_MAX_LEN
        self.augmentation = augmentation        
        self.transform = transform
        self.len_phoc = len( PHOC(word='abcd', cf = self.cf) ) # passing an arbitrary string to get the phoc lenght

    def __getitem__(self, index):
        word = self.file_label[index]                
        label, label_mask = label_padding(' '.join(word[1:]), self.output_max_len)
        word_str = word[1] # word_str = word[1].lower(); # to only keep lower-case
        # img, img_width = readImage_keepRatio(word[0], FLIP, self.augmentation, self.transformer, self.cf)
        # in the origianl implementation (img[0]) has the word img[1]) has the image        
        # return img[1], target, word_str

        img = get_the_image(word[0], self.transform, self.cf) 
        target = PHOC(word_str, self.cf)        
        return img, target, word_str

    def __len__(self):
        return len(self.file_label)
     
    def num_classes(self):
        return self.len_phoc

        

    