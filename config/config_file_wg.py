import time   # used to create a seed for the randomizers

encoder = 'phoc' # ['rawhoc', 'phoc', 'pro_hoc']
if encoder =='phoc':
    from scripts.Word2PHOC import build_phoc as PHOC
    unigram_levels               = [2, 3, 4, 5 ]  # # PHOC levels                                            

elif encoder == 'rawhoc' :
    from scripts.Word2RAWHOC import build_rawhoc as PHOC
    rawhoc_repeates = 2
    max_word_len = 24
elif encoder == 'pro_hoc': 
    from scripts.Word2RAWHOC import build_pro_hoc as PHOC
    unigram_levels               = [2, 3, 4, 5 ]  # # PHOC levels   
    rawhoc_repeates = 2
    max_word_len = 24   
    
else: 
    print('wrong encoder name: one of; phoc, rawhoc, pro_hoc')                                      

phoc_tolerance = 0 # if above 0, it will perturbate the phoc/rawhoc by tolerance

# Dataset
dataset_name            = 'IAM'#  'WG+IFN'      # Dataset name: ['WG', 'IFN', 'WG+IFN', IAM]

if dataset_name ==  'WG': # 645 x 120
    MAX_IMAGE_WIDTH  = 645
    MAX_IMAGE_HEIGHT = 120

elif dataset_name == 'IFN': # 1069 x 226
    MAX_IMAGE_WIDTH  = 1069 # (set_a: h226, w1035); (set_b: h214, w1069); (set_c: w211, h1028); (set_d: h195, w1041);  (set_e: h-197, w-977)
    MAX_IMAGE_HEIGHT = 226  
    H_ifn_scale      = 0  # to skip scaling the height, use 0
    
elif dataset_name == 'IAM': # 1087 x 241
    MAX_IMAGE_WIDTH  = 1087
    MAX_IMAGE_HEIGHT = 241
    
    ''' for mix language, we have to scale IFN to WG size'''
elif dataset_name == 'WG+IFN':  
    H_ifn_scale      = 120 # to skip scaling the height, use 0, pr, use WG_IMAGE_HEIGHT = 120
    MAX_IMAGE_WIDTH  = 1069 # 
    MAX_IMAGE_HEIGHT = 120    # maybe this should be 120, as GW and IFN are 120 after h_ifn_scale 
    

elif dataset_name == 'IAM+IFN':
    MAX_IMAGE_WIDTH  = 1087
    MAX_IMAGE_HEIGHT = 241
    H_ifn_scale = 0

    
    
''' Path to  Data '''
folder_of_data              = '/home/malrawi/Desktop/My Programs/all_data/'



dataset_path_IFN             = folder_of_data + 'ifnenit_v2.0p1e/data/set_e/bmp/' # path to IFN images
gt_path_IFN                  = folder_of_data + 'ifnenit_v2.0p1e/data/set_e/tru/' # path to IFN ground_truth
# For IFN, there are other folers/sets, b, c, d, e ;  sets are stored in {a, b, c, d ,e}
'''
# all the dataa in one folder
dataset_path_IFN              = folder_of_data + 'ifnenit_v2.0p1e/all_folders/bmp/' # path to IFN images
gt_path_IFN                   = folder_of_data + 'ifnenit_v2.0p1e/all_folders/tru/' # path to IFN ground_truth
'''

dataset_path_WG              = folder_of_data + 'washingtondb-v1.0/data/word_images_normalized'    # path to WG images
gt_path_WG                   = folder_of_data + 'washingtondb-v1.0/ground_truth/word_labels.txt'   # path to WG ground_truth

dataset_path_IAM             = folder_of_data + 'IAM-V3/iam-images/'    # path to IAM images
gt_path_IAM                  = folder_of_data + 'IAM-V3/iam-ground-truth/'   # path to IAM ground_truth
       
del folder_of_data # not needed anymore


use_weight_to_balance_data   = False
keep_non_alphabet_of_GW_in_analysis       = True  # if True, it will be used in the analysis, else, it will be skipped from the phoc, even if has been loaded  
keep_non_alphabet_of_GW_in_loaded_data    = True 

split_percentage             = .8  # 80% will be used to build the PHOC_net, and 20% will be used for tesging it, randomly selected 
rnd_seed_value               =  1533323200 #0 # int(time.time())  #  0 # time.time() should be used later
train_split                  = True # When True, this is the training set 

# Input Images
normalize_images             = False
pad_images                   = True         # Pad the input images to a fixed size [576, 226]
resize_images                = True         # Resize the dataset images to a fixed size
input_size                   = (120, 600) # [60, 150]   # Input size of the dataset images [HeightxWidth], images will be re-scaled to this size
   
                                         # H= 40, then, W = (576/226)*40 ~= 100
# Dataloader
batch_size_train             = 4  # Prev works say the less the better, 10 is best?!
batch_size_test              = 100  # Higher values may trigger memory problems
shuffle                      = True # shuffle the training set
num_workers                  = 4
thinning_threshold           = 0 # 0.4 # This value should be decided upon investigating 
                                    # the histogram of text to background, see the function hist_of_text_to_background_ratio in test_a_loader.py
                                    # use 0 to indicate no thinning, could only be used with IAM, as part of the transform

# Model parameters
model_name                   = 'resnet152' #'resnet152' #'resnet50' #'resnet152' # 'vgg16_bn'#  'resnet50' # ['resnet', 'PHOCNet', ...]
epochs                       = 100 
momentum                     = 0.9
weight_decay                 = 1*10e-14
learning_rate                = 0.1 #10e-4
lr_milestones                = [ 40, 80, 150 ]  # it is better to move this in the config
lr_gamma                     = 0.1 # learning rate decay calue
use_nestrov_moment           = True 
damp_moment                  = 0 # Nestrove will toggle off dampening moment
dropout_probability          = 0


pretrained                   = True # When true, ImageNet weigths will be loaded to the DCNN
testing_print_frequency      = 11 # prime number, how frequent to test/print during training
batch_log                    = 1000  # how often to report/print the training loss
binarizing_thresh            = 0.2 # threshold to be used to binarize the net sigmoid output, 
                                # every val > threshold will be converted to 1
                              # threshold 0.5 will run round() function
loss                         = 'BCEWithLogitsLoss' # ['BCEWithLogitsLoss', 'MSELoss', 'CrossEntropyLoss']
mAP_dist_metric              = 'cosine' # See options below


    

''' Language / dataset to use '''
if dataset_name == 'WG':
    if keep_non_alphabet_of_GW_in_analysis == True:
        phoc_unigrams =".0123456789abcdefghijklmnopqrstuvwxyz,-;':()£|"
    else:
        phoc_unigrams ='abcdefghijklmnopqrstuvwxyz0123456789'
elif dataset_name =='IFN':
    phoc_unigrams ="0123456789أءابجدهوزطحيكلمنسعفصقرشتثخذضظغةى.ئإآ\'ّ''"
elif dataset_name == 'WG+IFN':
    if keep_non_alphabet_of_GW_in_analysis==True:
        phoc_unigrams ="abcdefghijklmnopqrstuvwxyz,-;':()£|0123456789أءابجدهوزطحيكلمنسعفصقرشتثخذضظغةى.ئإآ\'ّ''"
    else:
        phoc_unigrams ="abcdefghijklmnopqrstuvwxyz0123456789أءابجدهوزطحيكلمنسعفصقرشتثخذضظغةى.ئإآ\'ّ''"
        
elif dataset_name == 'IAM':
    # all symbol    # x = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    # juse just lower case
    x = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    phoc_unigrams = ''.join(map(str, x))
else: 
    exit("Datasets to use: 'WG', 'IFN', 'IAM', or 'WG+IAM' ")
    
# Save results
save_results                 = False                            # Save Log file
results_path                 = 'datasets/washingtondb-v1.0/results'  # Output folder to save the results of the test
save_plots                   = True                                                     # Save the plots to disk
    



'''
list of model_name : 
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


