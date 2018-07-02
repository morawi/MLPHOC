# Dataset
dataset_name                 = 'IFN' # Dataset name: ['WG', 'IFN']
dataset_path                 = '../datasets/ifnenit_v2.0p1e/data/set_a/bmp/'   # Dataset images path
gt_path                      = '../datasets/ifnenit_v2.0p1e/data/set_a/tru/'   # Ground truth path
train_split                  = True
non_alphabet                 = False

# Input Images
pad_images                   = True         # Pad the input images to a fixed size [576, 226]
resize_images                = True         # Resize the dataset images to a fixed size
input_size                   = [256, 256]   # Input size of the dataset images

# Dataloader
batch_size_train             = 10
batch_size_test              = 1
shuffle                      = True
num_workers                  = 8

# PHOC
alphabet                     = 'arabic'    # ['english', 'arabic', 'multiple']
unigram_levels               = [2, 3, 4, 5]

# Model parameters
model_name                   = 'resnet50' # ['resnet', 'PHOCNet', ...]
epochs                       = 100 #80000
solver_type                  = 'Adam' # 'SGD'
momentum                     = 0.9
weight_decay                 = 5*10e-5
learning_rate                = 0.01 #10e-4
dropout_probability          = 0.2
loss                         = 'BCEWithLogitsLoss' # ['BCEWithLogitsLoss', 'CrossEntropyLoss']
pretrained                   = False

# Save results
save_results                 = True                                   # Save Log file
results_path                 = 'datasets/ifnenit_v2.0p1e/results'     # Output folder to save the results of the test
save_plots = True                                                     # Save the plots to disk