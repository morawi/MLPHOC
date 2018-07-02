# Dataset
dataset_name                 = 'WG' # Dataset name: ['WG', 'IFN']
dataset_path                 = 'datasets/washingtondb-v1.0/data/word_images_normalized'    # Dataset images path
gt_path                      = 'datasets/washingtondb-v1.0/ground_truth/word_labels.txt'   # Ground truth path
train_split                  = True
non_alphabet                 = False

# Input Images
pad_images                   = True         # Pad the input images to a fixed size [576, 226]
resize_images                = True         # Resize the dataset images to a fixed size
input_size                   = [256, 256]   # Input size of the dataset images

# Dataloader
batch_size_train             = 12
batch_size_test              = 12
shuffle                      = True
num_workers                  = 4

# PHOC
alphabet                     = 'english'    # ['english', 'arabic', 'multiple']
unigram_levels               = [2, 3, 4, 5]

# Model parameters
model_name                   = 'resnet50' # ['resnet', 'PHOCNet', ...]
epochs                       = 1 #80000
momentum                     = 0.9
weight_decay                 = 5*10e-5
learning_rate                = 0.01 #10e-4
dropout_probability          = 0.2
loss                         = 'BCEWithLogitsLoss' # ['BCEWithLogitsLoss', 'CrossEntropyLoss']
pretrained                   = False

# Save results
save_results                 = True                                   # Save Log file
results_path                 = 'datasets/washingtondb-v1.0/results'  # Output folder to save the results of the test
save_plots = True                                                     # Save the plots to disk