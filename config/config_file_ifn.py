# Dataset
dataset_name                 = 'IFN' # Dataset name: ['WG', 'IFN']
dataset_path                 = 'datasets/ifnenit_v2.0p1e/data/set_a/bmp/'   # Dataset images path
gt_path                      = 'datasets/ifnenit_v2.0p1e/data/set_a/tru/'   # Ground truth path
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
alphabet                     = 'arabic'    # ['english', 'arabic', 'multiple']
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
results_path                 = 'datasets/ifnenit_v2.0p1e/results'     # Output folder to save the results of the test
save_plots = True                                                     # Save the plots to disk