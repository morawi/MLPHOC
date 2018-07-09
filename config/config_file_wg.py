# Dataset
dataset_name                 = 'WG' # Dataset name: ['WG', 'IFN', 'WG+IFN']
dataset_path                 = 'datasets/washingtondb-v1.0/data/word_images_normalized'    # Dataset images path
gt_path                      = 'datasets/washingtondb-v1.0/ground_truth/word_labels.txt'   # Ground truth path
train_split                  = True
non_alphabet                 = False

# Input Images
pad_images                   = True         # Pad the input images to a fixed size [576, 226]
resize_images                = True         # Resize the dataset images to a fixed size
input_size                   = [120, 300]   # Input size of the dataset images [HeightxWidth], images will be re-scaled to this size
                                            # H= 40, then, W = (576/226)*40 ~= 100
# Dataloader
batch_size_train             = 16
batch_size_test              = 100  # In fact, the max test size should be used
shuffle                      = True
num_workers                  = 4

# PHOC
alphabet                     = 'english'    # ['english', 'arabic', 'multiple'] # Eiher use alphabet or dataset_name, 
                                            # then, using if statement to determine the other
unigram_levels               = [2, 3, 4, 5]

# Model parameters
model_name                   = 'vgg19_bn' # 'vgg16_bn'#  'resnet50' # ['resnet', 'PHOCNet', ...]
epochs                       = 300 #300
momentum                     = 0.9
weight_decay                 = 5*10e-5
learning_rate                = 0.1 #10e-4
dropout_probability          = 0.1
loss                         = 'BCEWithLogitsLoss' # ['BCEWithLogitsLoss', 'CrossEntropyLoss']
mAP_dist_metric              = 'cosine' # See options below
pretrained                   = True # False

lr_milestones                = [10, 30, 80, 200 ]  # it is better to move this in the config
lr_gamma                     = 0.1 # learning rate decay calue
    
# Save results
save_results                 = True                                   # Save Log file
results_path                 = 'datasets/washingtondb-v1.0/results'  # Output folder to save the results of the test
save_plots = True                                                     # Save the plots to disk
testing_print_frequency      = 11 # prime number, how frequent to print during testing

batch_log                    = 200  # how often to report the training loss

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


