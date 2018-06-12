# Dataset
dataset_name                 = 'WG' # Dataset name: ['WG', 'IFN']
dataset_path                 = 'datasets/washingtondb-v1.0/data/word_images_normalized'    # Dataset images path
gt_path                      = 'datasets/washingtondb-v1.0/ground_truth/word_labels.txt'   # Ground truth path
train_split                  = True
non_alphabet                 = False

# Dataloader
batch_size                   = 4
shuffle                      = True
num_workers                  = 4

# Input Images
pad_images                   = True         # Pad the input images to a fixed size [576, 226]
resize_images                = True         # Resize the dataset images to a fixed size
input_size                   = [256, 256]   # Input size of the dataset images

# PHOC
alphabet                     = 'english'    # ['english', 'arabic', 'multiple']
unigram_levels               = [2, 3, 4, 5]

# Model parameters
model                        = 'PHOCNet' # ['resnet', 'PHOCNet', ...]
num_iterations               = 80000
batch_size                   = 10
momentum                     = 0.9
weight_decay                 = 5*10e-5
learning_rate                = 10e-4

# Save results
save_results                 = True                                   # Save Log file
results_path                 = 'datasets/washingtondb-v1.0/results'  # Output folder to save the results of the test
save_plots = True                                                     # Save the plots to disk