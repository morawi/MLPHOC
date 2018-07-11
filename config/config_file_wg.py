# Dataset
dataset_name                 = 'WG+IFN' # Dataset name: ['WG', 'IFN', 'WG+IFN']

"""Select the aplabet of the dataset. The value 'alphabet will be used in the PHOC function'
to map the string to the phoc according to the chars of the language """
if dataset_name=='IFN':
    alphabet                     = 'arabic'    
elif dataset_name=='WG': 
    alphabet                     = 'english'    
elif dataset_name== 'WG+IFN': 
    alphabet                     = 'multiple'


train_split                  = True # When True, this is the training set 
non_alphabet                 = False  # This option can be used to include non_alphabet (if true)
split_percentage             = 0.80  # 80% will be used to build the PHOC_net, and 20% will be used for tesging it, randomly selected 
rnd_seed_value               = 0 # time.time() should be used later

# Input Images
pad_images                   = True         # Pad the input images to a fixed size [576, 226]
resize_images                = True         # Resize the dataset images to a fixed size
input_size                   = [60, 150]   # Input size of the dataset images [HeightxWidth], images will be re-scaled to this size
                                            # H= 40, then, W = (576/226)*40 ~= 100
# Dataloader
batch_size_train             = 16  # Prev works say the less the better, 10 is best?!
batch_size_test              = 100  # Higher values may trigger memory problems
shuffle                      = True # shuffle the training set
num_workers                  = 4


dataset_path_IFN              = 'datasets/ifnenit_v2.0p1e/data/set_a/bmp/' # path to IFN images
gt_path_IFN                   = 'datasets/ifnenit_v2.0p1e/data/set_a/tru/' # path to IFN ground_truth
# For IFN, there are other folers/sets, b, c, d, e ;  sets are stored in {a, b, c, d ,e}

dataset_path_WG              = 'datasets/washingtondb-v1.0/data/word_images_normalized'    # path to WG images
gt_path_WG                   = 'datasets/washingtondb-v1.0/ground_truth/word_labels.txt'   # path to WG ground_truth

       
# PHOC levels                                            
unigram_levels               = [2, 3, 4, 5]

# Model parameters
model_name                   = 'vgg19_bn' # 'vgg16_bn'#  'resnet50' # ['resnet', 'PHOCNet', ...]
epochs                       = 200 #300
momentum                     = 0.9
weight_decay                 = 5*10e-5
learning_rate                = 0.1 #10e-4
dropout_probability          = 0.1
loss                         = 'BCEWithLogitsLoss' # ['BCEWithLogitsLoss', 'CrossEntropyLoss']
mAP_dist_metric              = 'cosine' # See options below
pretrained                   = True # When true, ImageNet weigths will be loaded to the DCNN

lr_milestones                = [10, 30, 80, 200 ]  # it is better to move this in the config
lr_gamma                     = 0.1 # learning rate decay calue
testing_print_frequency      = 11 # prime number, how frequent to test/print during training
batch_log                    = 1000  # how often to report/print the training loss

    
# Save results
save_results                 = True                                   # Save Log file
results_path                 = 'datasets/washingtondb-v1.0/results'  # Output folder to save the results of the test
save_plots = True                                                     # Save the plots to disk

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


