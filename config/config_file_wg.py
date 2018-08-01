import time   # used to create a seed for the randomizers

'''

3- add thininig preprocessing
4- add scaling preprocessing, for IAM only, to scale if the image has top and down char parts 

'''
 

# Dataset
dataset_name                 = 'WG+IFN'       # Dataset name: ['WG', 'IFN', 'WG+IFN', IAM]

if dataset_name ==  'WG': # 645 x 120
    MAX_IMAGE_WIDTH = 645
    MAX_IMAGE_HEIGHT = 120

elif dataset_name == 'IFN': # 1069 x 226
    MAX_IMAGE_WIDTH = 1069
    MAX_IMAGE_HEIGHT = 226  
    H_ifn_scale = 120  # to skip scaling the height, use 0
    
elif dataset_name == 'IAM': # 1087 x 241
    MAX_IMAGE_WIDTH = 1087
    MAX_IMAGE_HEIGHT = 241
    
    ''' for mix language, we have to scale IFN to WG size'''
elif dataset_name == 'WG+IFN':   
    MAX_IMAGE_WIDTH = 1069 # 
    MAX_IMAGE_HEIGHT = 226     
    H_ifn_scale = 0 # to skip scaling the height, use 0, pr, use WG_IMAGE_HEIGHT = 120

   


keep_non_alphabet_in_GW      = True  # This option can be used to include non_alphabet (if true), only for GW dataset
split_percentage             = .8  # 80% will be used to build the PHOC_net, and 20% will be used for tesging it, randomly selected 
rnd_seed_value               = 0 #  int(time.time())  #  0 # time.time() should be used later
train_split                  = True # When True, this is the training set 

# Input Images
pad_images                   = True         # Pad the input images to a fixed size [576, 226]
resize_images                = True         # Resize the dataset images to a fixed size
input_size                   = [120, 400] # [60, 150]   # Input size of the dataset images [HeightxWidth], images will be re-scaled to this size
                                            # H= 40, then, W = (576/226)*40 ~= 100
# Dataloader
batch_size_train             = 16  # Prev works say the less the better, 10 is best?!
batch_size_test              = 300  # Higher values may trigger memory problems
shuffle                      = True # shuffle the training set
num_workers                  = 4
thinning_threshold           = 0.2 # This value should be decided upon investigating 
                                    # the histogram of text to background, see the function hist_of_text_to_background_ratio in test_a_loader.py
                                    # use 0 ti indicate no thinning
''' Path to  Data '''
folder_of_data              = '/home/malrawi/Desktop/My Programs/all_data/'


dataset_path_IFN             = folder_of_data + 'ifnenit_v2.0p1e/data/set_a/bmp/' # path to IFN images
gt_path_IFN                  = folder_of_data + 'ifnenit_v2.0p1e/data/set_a/tru/' # path to IFN ground_truth
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
       
# PHOC levels                                            
unigram_levels               = [2, 3, 4, 5]

# Model parameters
model_name                   = 'resnet152' # 'vgg16_bn'#  'resnet50' # ['resnet', 'PHOCNet', ...]
epochs                       = 60 # 300 #300
momentum                     = 0.9
weight_decay                 = 5*10e-5
learning_rate                = 0.1 #10e-4
dropout_probability          = 0.25
loss                         = 'BCEWithLogitsLoss' # ['BCEWithLogitsLoss', 'CrossEntropyLoss']
mAP_dist_metric              = 'cosine' # See options below
pretrained                   = True # When true, ImageNet weigths will be loaded to the DCNN
lr_milestones                = [10, 30, 50, 100, 200 ]  # it is better to move this in the config
lr_gamma                     = 0.1 # learning rate decay calue
testing_print_frequency      = 11 # prime number, how frequent to test/print during training
batch_log                    = 1000  # how often to report/print the training loss
binarizing_thresh            = 0.3 # threshold to be used to binarize the net sigmoid output, 
                                # every val > threshold will be converted to 1
                              # threshold 0.5 will run round() function
    
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


