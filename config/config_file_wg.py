''' 
Main @author: malrawi 
'''

import time   # used to create a seed for the randomizers


dataset_name    = 'IAM'#  'WG+IFN' , 'IAM+IFN'     # Dataset name: ['WG', 'IFN', 'WG+IFN', IAM]
encoder         = 'phoc' # ['label', 'rawhoc', 'phoc', 'pro_hoc']  label is used for script recognition only
folder_of_data              = '/home/malrawi/Desktop/My Programs/all_data/'
redirect_std_to_file   = False  # The output 'll be stored in a file if True 
normalize_images       = False
overlay_handwritting_on_STL_img = True
if overlay_handwritting_on_STL_img:
    normalize_images = True # have not used it in the analysis, yet
    

phoc_levels = [2 ,3, 4, 5 ]
phoc_tolerance = 0 # if above 0,  it will perturbate the phoc/rawhoc by tolerance 0=< phoc_tolerance <<1
if encoder =='phoc':
    from scripts.Word2PHOC import build_phoc as PHOC
    unigram_levels               = phoc_levels  # # PHOC levels                                            

elif encoder == 'rawhoc' :
    from scripts.Word2RAWHOC import build_rawhoc as PHOC
    rawhoc_repeates = 2
    max_word_len = 24
    
elif encoder == 'pro_hoc': 
    from scripts.Word2RAWHOC import build_pro_hoc as PHOC
    unigram_levels               = phoc_levels  # # PHOC levels   
    rawhoc_repeates = 2
    max_word_len = 24   
else: 
    print('wrong encoder name: one of; phoc, rawhoc, pro_hoc')      
del phoc_levels                                

        

# Dataset max W and H
if dataset_name ==  'WG': # 645 x 120
    MAX_IMAGE_WIDTH  = 645
    MAX_IMAGE_HEIGHT = 120

elif dataset_name == 'IFN': # 1069 x 226
    MAX_IMAGE_WIDTH  = 1069 # (set_a: h226, w1035); (set_b: h214, w1069); (set_c: w211, h1028); (set_d: h195, w1041);  (set_e: h-197, w-977)
    MAX_IMAGE_HEIGHT = 226  
    H_ifn_scale      = 0  # to skip scaling the height, use 0
    
elif dataset_name == 'IAM': # 1087 x 241
    MAX_IMAGE_WIDTH  = 1087
    # MAX_IMAGE_HEIGHT = 241
    
    '''Testing latest normalization'''
    MAX_IMAGE_HEIGHT = 120
    
    H_iam_scale      = 120 
    # In IAM
    #Max Image Height is 241 n02-049-03-02 (182, 241) test set
    #Max Image Width is 1087 c06-103-00-01 (1087, 199) train set
    
    ''' for mix language, we have to scale IFN to WG size'''
elif dataset_name == 'WG+IFN':      
    MAX_IMAGE_WIDTH  = 1069 # 
    MAX_IMAGE_HEIGHT = 120    # maybe this should be 120, as GW and IFN are 120 after h_ifn_scale 
    H_ifn_scale      = 120 # to skip scaling the height, use 0, pr, use WG_IMAGE_HEIGHT = 120
    
elif dataset_name == 'IAM+IFN':
    MAX_IMAGE_WIDTH  = 1087
    MAX_IMAGE_HEIGHT = 241
    H_iam_scale      = 120 
    H_ifn_scale      = 120 # to skip scaling the height, use 0, pr, use WG_IMAGE_HEIGHT = 120


# Input Images
use_weight_to_balance_data      = False
use_distortion_augmentor        = False
thinning_threshold              = .35 #  1   no thinning  # This value should be decided upon investigating                          # the histogram of text to background, see the function hist_of_text_to_background_ratio in test_a_loader.py # use 1 to indicate no thinning, could only be used with IAM, as part of the transform

pad_images                   = True         # Pad the input images to a fixed size [576, 226]
resize_images                = True         # Resize the dataset images to a fixed size
if resize_images:
    input_size               = (120, 600) # [60, 150]   # Input size of the dataset images [HeightxWidth], images will be re-scaled to this size
else: 
    input_size = ( MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH )
   

# Model parameters
model_name                   = 'resnet152' # 'resnet152' #'resnet152' #'resnet50' #'resnet152' # 'vgg16_bn'#  'resnet50' # ['resnet', 'PHOCNet', ...]
pretrained                   = True # When true, ImageNet weigths will be loaded to the DCNN
momentum                     = 0.9
weight_decay                 = 1*10e-14
learning_rate                = 0.1 #10e-4
lr_milestones                = [ 40, 80, 150 ]  # it is better to move this in the config
lr_gamma                     = 0.1 # learning rate decay calue
use_nestrov_moment           = True 
damp_moment                  = 0 # Nestrove will toggle off dampening moment
dropout_probability          = 0
testing_print_frequency      = 11 # prime number, how frequent to test/print during training
batch_log                    = 2000  # how often to report/print the training loss
binarizing_thresh            = 0.5 # threshold to be used to binarize the net sigmoid output, 

epochs                       = 100
batch_size_train             = 2 
if dataset_name=='IAM' or dataset_name == 'IAM+IFN':
    batch_size_train             = 6 #  value of 2 gives better results

batch_size_test              = 100  # Higher values may trigger memory problems
shuffle                      = True # shuffle the training set
num_workers                  = 4
loss                         = 'BCEWithLogitsLoss' # ['BCEWithLogitsLoss', 'MSELoss', 'CrossEntropyLoss']
mAP_dist_metric              = 'cosine' # See options below
rnd_seed_value               = int(time.time()) # 1533323200 #int(time.time()) #  #0 # int(time.time())  #  0 # time.time() should be used later

if encoder == 'label':
    loss == 'CrossEntropyLoss'
    batch_size_train         = 10  # Prev works used 10 .....  a value of 2 gives better results
    model_name               = 'resnet18'
    testing_print_frequency  = 2 # prime number, how frequent to test/print during training
    dataset_name    = 'WG+IFN' # or 'IAM+IFN'


IFN_test = 'set_a'
IFN_all_data_grouped_in_one_folder = True

if IFN_all_data_grouped_in_one_folder:
    IFN_based_on_folds_experiment  = False
    del IFN_test 
    # all the data in one folder
    dataset_path_IFN              = folder_of_data + 'ifnenit_v2.0p1e/all_folders/bmp/' # path to IFN images
    gt_path_IFN                   = folder_of_data + 'ifnenit_v2.0p1e/all_folders/tru/' # path to IFN ground_truth
else:
    
    dataset_path_IFN             = folder_of_data + 'ifnenit_v2.0p1e/data/'+ IFN_test +'/bmp/' # path to IFN images
    gt_path_IFN                  = folder_of_data + 'ifnenit_v2.0p1e/data/'+ IFN_test + '/tru/' # path to IFN ground_truth
    # For IFN, there are other folers/sets, b, c, d, e ;  sets are stored in {a, b, c, d ,e}

dataset_path_WG              = folder_of_data + 'washingtondb-v1.0/data/word_images_normalized'    # path to WG images
gt_path_WG                   = folder_of_data + 'washingtondb-v1.0/ground_truth/word_labels.txt'   # path to WG ground_truth

dataset_path_IAM             = folder_of_data + 'IAM-V3/iam-images/'    # path to IAM images
gt_path_IAM                  = folder_of_data + 'IAM-V3/iam-ground-truth/'   # path to IAM ground_truth


IFN_based_on_folds_experiment  = False
train_split                    = True # When True, this is the training set 
if train_split: 
    split_percentage         = .75  # 80% will be used to build the PHOC_net, and 20% will be used for tesging it, randomly selected 
if IFN_based_on_folds_experiment==True and dataset_name=='IFN': 
    train_split              = False # no split will be applied 
    split_percentage         = 1
    folders_to_use = 'abcde'   # 'eabcd' or 'abcd' in the publihsed papers, only abcd are used, donno why!?


''' I need to remove these keep flags'''
keep_non_alphabet_of_GW_in_analysis       = True  # if True, it will be used in the analysis, else, it will be skipped from the phoc, even if has been loaded  
keep_non_alphabet_of_GW_in_loaded_data    = True 


''' Language / script dataset to use '''       
iam_char = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', 
            '/', ':', ';', '?', '_',  
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 
            'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 
            'w', 'x', 'y', 'z'] # upper case removed
iam_char = ''.join(map(str, iam_char))
ifn_char = "0123456789أءابجدهوزطحيكلمنسعفصقرشتثخذضظغةى.ئإآ\'ّ''"
gw_char =  ".0123456789abcdefghijklmnopqrstuvwxyz,-;':()£|"

if dataset_name == 'WG':   
    phoc_unigrams = gw_char    

elif dataset_name =='IFN':
    phoc_unigrams = ifn_char    

elif dataset_name == 'WG+IFN':    
    phoc_unigrams =''.join(sorted( set(ifn_char + gw_char) ))        

elif dataset_name == 'IAM':    
    phoc_unigrams = ''.join(map(str, iam_char))
    
elif dataset_name == 'IAM+IFN':                 
    phoc_unigrams = ''.join(sorted(set(iam_char + ifn_char)))    

else: 
    exit("Datasets to use: 'WG', 'IFN', 'IAM', 'WG+IAM', or 'IAM+IFN' ")
            
del iam_char, ifn_char, gw_char

# Save results
save_results           = False                            # Save Log file
results_path           = 'datasets/washingtondb-v1.0/results'  # Output folder to save the results of the test




'''
 of model_name : 
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
    'resnext101_32x4d', 'resnext101_64x4d', 'nasnetalarge',   
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'xception',        
    'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn131', 'dpn107',
    'densenet121', 'densenet169', 'densenet201', 'densenet161',       
    'squeezenet1_0', 'squeezenet1_1', 
    'alexnet', 
    'resnext101_32x4d' , 'resnext101_64x4d'
    'nasnetalarge' , 'nasnetamobile',
    'inceptionresnetv2', 'inception_v3', 'inception_v4'
'''

'''  
list of mAP_dist_metric:  
    'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 
    'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 
    'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 
    'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    
'''


