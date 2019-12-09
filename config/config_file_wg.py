''' 
Main @author: malrawi 

'''


''' torch distanc for retrieval
https://pytorch.org/docs/stable/_modules/torch/nn/modules/distance.html
'''

import time   # used to create a seed for the randomizers
folder_of_data         = '/home/malrawi/MyPrograms-old/' #'/home/malrawi/Desktop/My Programs/'

use_hashing = True
print_accuracy = True
normalize_images     =  False
phoc_levels = [ 2, 3, 4, 5, 6, 8]  # [2,3,4,5,6,7,8] be used with varphoc
encoder   = 'phoc' # 'phonetic_vec'#   'phoc' # 'chars2vec' 'rohoc' #   ['rohoc', 'rawhoc', 'phoc', 'pro_hoc']  
task_type = 'word_spotting'  # 'script_identification' #  'word_spotting' # 'script_identification' 
resize_images        = True        # Resize the dataset images to a fixed size
pad_images           = False         # Pad the input images to a fixed size [576, 226]
redirect_std_to_file   = False  # The output 'll be stored in a file if True 

def get_dataset_name(data_set_id  = 0):    
    all_datasets = ['Cifar100+TFSPCH+IAM+IFN',  # 0
                    'Cifar100+TFSPCH+GW+IFN',   # 1
                    'Cifar100+TFSPCH+IAM+IFN+safe-driver', # 2
                    'WG+IFN' ,  # 3 
                    'IAM+IFN', # 4
                    'WG', # 5                
                    'IFN', # 6 
                    'IAM', # 7
                    'Cifar100', # 8
                    'TFSPCH', # 9
                    'safe_driver', # 10
                    'imdb_movie', # 11
                     'Cifar100+TFSPCH+IAM+IFN+safe-driver+imdb', # 12
                     'Cifar100+TFSPCH+IAM+IFN+safe-driver+imdb+cub2011', # 13                 
                     'Cifar100+TFSPCH+IAM+IFN+safe-driver+imdb+cub2011+MLT', #14
                     'MLT', # 15   
                     'cub2011', #16     
                     
                    ]
    
    dataset_name    = all_datasets[data_set_id]
    return dataset_name

dataset_name = get_dataset_name(13)


overlay_handwritting_on_STL_img = False
if overlay_handwritting_on_STL_img == True:
    change_hand_wrt_color = True    
    overlay_handwritting_on_STL_img = True


if 'imdb' in dataset_name: 
    word_corpus_4_text_understanding = 'Google_news' #'Custom', 'CharNGram' (char2vec) # 'Fasttext' # 'Google_news' # else, will use 'Brown' 'Fasttext', 'Glove'
    imdb_min_occurane = 10  # minimum number of words to appear in building the custom W2V model


sampled_testing = True # to be used if the testing set is larger than 30K, due to limited RAM memory
if sampled_testing: no_of_sampled_data = 25000 


if encoder =='phonetic_vec':    
    from scripts.Word2Phonetic import Word2Phonetic 
    Word2PhoneticObject = Word2Phonetic()
    PHOC = Word2PhoneticObject.getvec
    loss =  'BCEWithLogitsLoss' # ['BCEWithLogitsLoss', 'CrossEntropyLoss', 'MSELoss', ]
           
    
elif encoder =='chars2vec':    
    from scripts.Word2CharsVec import Chars2Vec 
    Chars2VecObject = Chars2Vec()
    PHOC = Chars2VecObject.getvec
    loss =  'BCEWithLogitsLoss'# 'BCELoss'# 'BCEWithLogitsLoss' # ['BCEWithLogitsLoss', 'CrossEntropyLoss', 'MSELoss', ]
    
elif encoder =='varphoc':
    from scripts.Word2VarPhoc import var_phoc as PHOC
    unigram_levels   = phoc_levels  # # PHOC levels 
    loss = 'BCEWithLogitsLoss' # ['BCEWithLogitsLoss', 'MSELoss', ]    
    
elif encoder =='phoc':
    from scripts.Word2PHOC import build_phoc as PHOC
    unigram_levels   = phoc_levels  # # PHOC levels 
    loss = 'BCEWithLogitsLoss' # ['BCEWithLogitsLoss', 'MSELoss', ]                                           

elif encoder == 'rawhoc' :
    from scripts.Word2RAWHOC import build_rawhoc as PHOC
    rawhoc_repeates = 2
    max_word_len = 24
    loss = 'BCEWithLogitsLoss' # ['BCEWithLogitsLoss', 'MSELoss', ]                                           
    
elif encoder == 'pro_hoc': 
    from scripts.Word2RAWHOC import build_pro_hoc as PHOC
    PHOC('') # trivial call to get rid of a warnning
    unigram_levels               = phoc_levels  # # PHOC levels   
    rawhoc_repeates = 2
    max_word_len = 24   
    loss = 'BCEWithLogitsLoss' # ['BCEWithLogitsLoss', 'MSELoss', ]                                           
elif encoder == 'rohoc':
    no_word_rotations = 5 
    from scripts.Word2RotatedHOC import rotated_hoc as PHOC  
    loss = 'BCEWithLogitsLoss' # ['BCEWithLogitsLoss', 'MSELoss', ]                                           
    unigram_levels   = [10] # this will, more or less, be useful in decoding 20-letter words   

else: 
    print('wrong encoder name: one of; phoc, rawhoc, pro_hoc')      
del phoc_levels                                


  
# Dataset max W and H
universal_H = 200  # 120 used in CVPR paper, H=Heigh. 

H_Instagram_scale= 200

if dataset_name  == 'MLT':
    H_MLT_scale = 128# 200
    MAX_IMAGE_WIDTH  = 256# 600 #  267 # 640
    MAX_IMAGE_HEIGHT = H_MLT_scale # 300 # 200 # 480
    H_Instagram_scale = H_MLT_scale
    H_gw_scale = H_MLT_scale    # to be used wiht MLT instagram if needed
    H_iam_scale = H_MLT_scale

elif dataset_name  == 'cub2011':
    MAX_IMAGE_WIDTH  = 400 #  267 # 640
    MAX_IMAGE_HEIGHT = 300 # 200 # 480
    H_cub2011_scale = universal_H  # use 0 for no scale
    

elif dataset_name  == 'safe_driver':
    MAX_IMAGE_WIDTH  = 400 #  267 # 640
    MAX_IMAGE_HEIGHT = 300 # 200 # 480
    H_sfDrive_scale = 300#  universal_H
    
    
elif dataset_name =='Cifar100+TFSPCH+IAM+IFN' or dataset_name == 'Cifar100+TFSPCH+GW+IFN' or dataset_name == 'Cifar100+TFSPCH+IAM+IFN+safe-driver':
    MAX_IMAGE_WIDTH  = 600 # Adding H-GW_scale to GW load file 
    MAX_IMAGE_HEIGHT = universal_H 
    w_new_size_cifar100 = universal_H # these are used to up-scale Cifar100 dataset 
    h_new_size_cifar100 = universal_H
    H_ifn_scale  = universal_H
    H_iam_scale  = universal_H
    H_gw_scale = universal_H
    H_sfDrive_scale = universal_H
    H_TFSPCH_scale = 0 # do not scale speech data
    resize_images = False # override in case resize_images is True
    ''' Better to set resize_images to False in this case
    IFN and IAM will be rescaled to this MAX_IMAGE_WIDTH if their width is larget than MAX_IMAGE_WIDTH, 
    Cifar100 will have w_new_size, TSFPCH will be the same  '''

elif dataset_name == 'Cifar100+TFSPCH+IAM+IFN+safe-driver+imdb+cub2011' or \
   dataset_name == 'Cifar100+TFSPCH+IAM+IFN+safe-driver+imdb+cub2011+MLT':
                
    MAX_IMAGE_WIDTH  = 600 # Adding H-GW_scale to GW load file 
    MAX_IMAGE_HEIGHT = universal_H 
    if word_corpus_4_text_understanding=='Google_news':  
        MAX_IMAGE_HEIGHT = 300
        universal_H = 300
    
    resize_images = True # override in case resize_images is True    
    
    w_new_size_cifar100 = universal_H # these are used to up-scale Cifar100 dataset 
    h_new_size_cifar100 = universal_H
    H_ifn_scale  = 120 # images will be scaled down to match MAX_IMAGE_WIDTH, if the new width is larger than MAX_IMAGE_WIDTH
    H_iam_scale  = 120 # images will be scaled down to match MAX_IMAGE_WIDTH, if the new width is larger than MAX_IMAGE_WIDTH
    H_sfDrive_scale = universal_H
    H_TFSPCH_scale = 0 # do not scale speech data    
    W_imdb_width  = 600
    H_imdb_scale = 0 # universal_H   
    H_cub2011_scale = universal_H  # use 0 for no scale
    H_MLT_scale = universal_H
    
    ''' Better to set resize_images to False in this case
    IFN and IAM will be rescaled to this MAX_IMAGE_WIDTH if their width is larget than MAX_IMAGE_WIDTH, 
    Cifar100 will have w_new_size, TSFPCH will be the same  '''
    
elif dataset_name == 'Cifar100+TFSPCH+IAM+IFN+safe-driver+imdb':
    MAX_IMAGE_WIDTH  = 600 # Adding H-GW_scale to GW load file 
    MAX_IMAGE_HEIGHT = universal_H 
    if word_corpus_4_text_understanding=='Google_news':  
        MAX_IMAGE_HEIGHT = 300
        universal_H = 300
       
    w_new_size_cifar100 = universal_H # these are used to up-scale Cifar100 dataset 
    h_new_size_cifar100 = universal_H
    H_ifn_scale  = 120 # images will be scaled down to match MAX_IMAGE_WIDTH, if the new width is larger than MAX_IMAGE_WIDTH
    H_iam_scale  = 120 # images will be scaled down to match MAX_IMAGE_WIDTH, if the new width is larger than MAX_IMAGE_WIDTH
    H_sfDrive_scale = universal_H
    H_TFSPCH_scale = 0 # do not scale speech data
    resize_images = False # override in case resize_images is True
    W_imdb_width  = 600
    H_imdb_scale = 0 # universal_H   
    
    ''' Better to set resize_images to False in this case
    IFN and IAM will be rescaled to this MAX_IMAGE_WIDTH if their width is larget than MAX_IMAGE_WIDTH, 
    Cifar100 will have w_new_size, TSFPCH will be the same  '''

elif dataset_name ==  'imdb_movie': # W x H; depends on the parameters we pass to the sepectogram function
    MAX_IMAGE_WIDTH  = 600 # should be >=  W_imdb_width    
    MAX_IMAGE_HEIGHT = 100 # universal_H # to get a better max width, one must load/train the gensim model at this stage
    if word_corpus_4_text_understanding=='Google_news' or word_corpus_4_text_understanding=='Custom':  
        MAX_IMAGE_HEIGHT = 300
    W_imdb_width  = 600
    H_imdb_scale = 0 # universal_H
      
elif dataset_name == 'Cifar100':
    MAX_IMAGE_WIDTH  = 32 # 256
    MAX_IMAGE_HEIGHT = 32 # 256
    w_new_size_cifar100 = 32 # 256
    h_new_size_cifar100 = 32 # 256

elif dataset_name ==  'TFSPCH': # W x H; depends on the parameters we pass to the sepectogram function
    MAX_IMAGE_WIDTH  =  260 # 162
    MAX_IMAGE_HEIGHT = 260 # 100
    H_TFSPCH_scale = 0


elif dataset_name ==  'WG': # 645 x 120 (largest size in GW dataset)
    MAX_IMAGE_WIDTH  =  645
    MAX_IMAGE_HEIGHT = 120
    H_gw_scale = 0 #  MAX_IMAGE_HEIGHT


elif dataset_name == 'IFN': # 1069 x 226
    MAX_IMAGE_WIDTH  = 1069 # (set_a: h226, w1035); (set_b: h214, w1069); (set_c: w211, h1028); (set_d: h195, w1041);  (set_e: h-197, w-977)
    MAX_IMAGE_HEIGHT = universal_H  
    H_ifn_scale      = universal_H  # to skip scaling the height, use 0
            
elif dataset_name == 'IAM': # 1087 x 241 (largest size in IAM datasaet)
    MAX_IMAGE_WIDTH  = 1087 # 256 # 1087    
    MAX_IMAGE_HEIGHT = 241 # 241 # universal_H              
    H_iam_scale      = 0#  MAX_IMAGE_HEIGHT # universal_H
    ''' In IAM max image height is 241 n02-049-03-02 (182, 241) test set   Max Image Width is 1087 c06-103-00-01 (1087, 199) train set '''
    
    
    ''' To merge WG and IFN sets, we have to scale IFN to match WG height(120) '''
elif dataset_name == 'WG+IFN':  # max is 226x1069    
    MAX_IMAGE_WIDTH  = 1069 # 
    MAX_IMAGE_HEIGHT = universal_H    # maybe this should be 120, as GW and IFN are 120 after h_ifn_scale 
    H_ifn_scale      = universal_H #0# universal_H # to skip scaling the height, use 0, else use WG_IMAGE_HEIGHT = 120
    H_gw_scale = universal_H # 0
    
elif dataset_name == 'IAM+IFN': # 241 x 1087
    MAX_IMAGE_WIDTH  = 1087 
    MAX_IMAGE_HEIGHT = universal_H   # originaloly 241, changing it to universal_H
    H_iam_scale      = universal_H 
    H_ifn_scale      = universal_H # to skip scaling the height, use 0, else use WG_IMAGE_HEIGHT = 120
      

if resize_images:
    input_size       =  (128, 256) # [60, 150]   # Input size of the dataset images [HeightxWidth], images will be re-scaled to this size
else: 
    input_size = ( MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH )
  

# Model parameters
model_name                   = 'resnet18'# 'resnet18' #      #  #'resnet50' #'resnet152' # 'vgg16_bn'#  
learning_rate                = 0.1 # 10e-4 # 0.1


thinning_threshold              = 1# .35 #  1   no thinning  # This value should be decided upon investigating                          # the histogram of text to background, see the function hist_of_text_to_background_ratio in test_a_loader.py # use 1 to indicate no thinning, could only be used with IAM, as part of the transform
pretrained                   = True # When true, ImageNet weigths will be loaded to the DCNN
momentum                     = 0.9
weight_decay                 = 1*10e-14

lr_milestones                = [ 40, 80, 150 ]  # it is better to move this in the config
lr_gamma                     = 0.1 # learning rate decay calue
use_nestrov_moment           = True 
damp_moment                  = 0 # Nestrove will toggle off dampening moment

dropout_probability          = 0.05 # 0.25 #  0.25


epochs                       =  300 # 60# 10 # 60
testing_print_frequency      =  11 # prime number, how frequent to test/print during training
batch_log                    = 2000  # how often to report/print the training loss
binarizing_thresh            = 0.5 # threshold to be used to binarize the net sigmoid output, 
split_MLT     =.75
split_percentage           = .75  # 80% will be used to build the PHOC_net, and 20% will be used for tesging it, randomly selected 
split_percentage_TFSPCH    = 1 # we can use a different percentage for speech data, has not effect on testing now, as there is a test set on a separate folder 
                                # If not 1, say 0.90 this means the 90% of the data will be used for training. 
batch_size_test              = 100  # Higher values may trigger memory problems
shuffle                      = True # shuffle the training set
num_workers                  = 4
mAP_dist_metric              = 'cosine' #'correlation' # 'cosine' # See options below
rnd_seed_value  =  int(time.time()) # 1552938612 1533323200 #int(time.time()) #  #0 # int(time.time())  #  0 # time.time() should be used later
batch_size_train             =  16

''' Folders of data: STL is embedded  '''
dataset_path_IFN              = folder_of_data + 'all_data/ifnenit_v2.0p1e/all_folders/bmp/' # path to IFN images
gt_path_IFN                   = folder_of_data + 'all_data/ifnenit_v2.0p1e/all_folders/tru/' # path to IFN ground_truth
    
dataset_path_WG              = folder_of_data + 'all_data/washingtondb-v1.0/data/word_images_normalized'    # path to WG images
gt_path_WG                   = folder_of_data + 'all_data/washingtondb-v1.0/ground_truth/word_labels.txt'   # path to WG ground_truth

dataset_path_IAM             = folder_of_data + 'all_data/IAM-V3/iam-images/'    # path to IAM images
gt_path_IAM                  = folder_of_data + 'all_data/IAM-V3/iam-ground-truth/'   # path to IAM ground_truth

dataset_path_TF_SPEECH_train = folder_of_data + 'all_data/tf_speech_recognition_v1/train/audio/' # or v2, which is tf_speech_recognition_v2/train/audio/'
dataset_path_TF_SPEECH_test = folder_of_data + 'all_data/tf_speech_recognition_v1/test/audio/' # or v2, which is tf_speech_recognition_v2/train/audio/'
cifar100_path = folder_of_data + 'all_data//dataCifar100/'
stl100_path = folder_of_data +'all_data/dataSTL10'   
safe_driver_path = folder_of_data + 'all_data/safe_driver/train/'
dataset_path_MLT = folder_of_data+'all_data/MLT2017/'
dataset_path_InstagramHL=folder_of_data + 'all_data/InstagramHL/'

''' Language / script dataset to use '''       
iam_char = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', 
            '/', ':', ';', '?', '_',  
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 
            'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 
            'w', 'x', 'y', 'z'] # upper case removed
iam_char = ''.join(map(str, iam_char))
ifn_char = "0123456789أءابجدهوزطحيكلمنسعفصقرشتثخذضظغةى.ئإآ\'ّ''"
gw_char = ".0123456789abcdefghijklmnopqrstuvwxyz,-;':()£|"
iam_ifn_char = ''.join(sorted(set(iam_char + ifn_char))) 
wg_ifn_char = ''.join(sorted( set(ifn_char + gw_char) )) 

language_hash_code = {'Arabic': '1234', 'Bangla': 'govu9',  'English': '9872',  'French': 'zxyw3',
                  'German': '5609',  'Italian': 'pqkso'}  # this is only used for script identification, the purpose of the hashcode is to introduce variance in the PHOC representation
MLT_lang = 'Latin+Arabic+Bangla' # 'Latin' #  'Eng+Ara+Bang'# 'Latin+Arabic+Bangla' # 'Bangla' # 'Latin+Arabic' # 'Bangla' # 'Latin' #'Eng+Ara+Bang' # 'Bangla' #'English', 'Arabic', 'Bangla', 'Arabic+English'
MLT_latin_script_vs_others = False

if MLT_lang=='MLT_English+Instagram_test':
    MLT_languages = ['English']
    extra_MLT = "‘°\٬%$]€>@·[ؤ=٠×—،~ـ“ِ '"    
    mlt_char = ''.join(sorted( set(iam_char + extra_MLT) ))
    mlt_char = ''.join(sorted( set(gw_char + mlt_char) ))
    testing_print_frequency      = 11

elif MLT_lang=='Arabic':
    MLT_languages = ['Arabic']
    extra_MLT = "‘٬٫%$]€>:+/#!()@·°[ؤ=٠×،ـ١٦٧٨٩٤٥٢٣ڥ出ڤ'ٌ' ”“ِ 'ً'ڭڨ ُ َ~— ْ"
    mlt_char = ''.join(sorted( set(ifn_char + extra_MLT) ))

elif MLT_lang=='English':
    MLT_languages = ['English']
    extra_MLT = "‘°\٬%$]€>@·[ؤ=٠×—،~ـ“ِ '"    
    mlt_char = ''.join(sorted( set(iam_char + extra_MLT) ))

elif MLT_lang == 'Arabic+English':  
    extra_MLT = "‘٬٫%$]€>@·°[ؤ=٠×،—ـ١٦٧٨٩٤٥٢٣ڥڤ'ٌ' ”“ِ 'ً'ڭڨ ُ َ~— ْ"
    MLT_languages = ['English', 'Arabic']
    mlt_char = ''.join(sorted( set(iam_ifn_char + extra_MLT) ))
elif MLT_lang=='Bangla':
    mlt_char = ' !"%\'()*-./0123456789:?achilmnprstuy`~।ঁংঃঅআইঈউএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ািীুূৃেৈোৌ্ৎড়য়০১২৩৪৫৬৭৮৯\u200c'
    MLT_languages = ['Bangla']
elif MLT_lang == 'Eng+Ara+Bang':
    extra_MLT = "‘٬٫%$]€>@·°[ؤ=٠×،—ـ١٦٧٨٩٤٥٢٣ڥڤ'ٌ' ”“ِ 'ً'ڭڨ ُ َ~— ْ"
    extra_MLT = ''.join(sorted( set(ifn_char + extra_MLT) ))
    bangla_char = ' !"%\'()*-./0123456789:?achilmnprstuy`~।ঁংঃঅআইঈউএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ািীুূৃেৈোৌ্ৎড়য়০১২৩৪৫৬৭৮৯\u200c'
    extra_MLT = ''.join(sorted( set(bangla_char + extra_MLT) ))
    latin_char = '!"#$%&\'()*+-./0123456789:;<=>?€β@[\\]—¥_`abcdefghijklmnopqrstuvwxyz'
    mlt_char = ''.join(sorted( set(extra_MLT + latin_char) ))
  
    MLT_languages = ['English', 'Arabic', 'Bangla']

elif MLT_lang == 'Latin':
    mlt_char = '!"#$%&\'()*+-./0123456789:;<=>?€β@[\\]—¥_`abcdefghijklmnopqrstuvwxyz~°²·×ßàáâãäçèéêëìîòóôöøùúûüÿōœšʒ'
    MLT_languages = ['English', 'French','German','Italian']

elif MLT_lang == 'Latin+Arabic':
    extra_MLT = "‘٬٫%$]€>@·°[ؤ=٠×،—ـ١٦٧٨٩٤٥٢٣ڥڤ'ٌ' ”“ِ 'ً'ڭڨ ُ َ~— ْ"
    extra_MLT = ''.join(sorted( set(ifn_char + extra_MLT) ))
    latin_char = '!"#$%&\'()*+-./0123456789:;<=>?€β@[\\]—¥_`abcdefghijklmnopqrstuvwxyz~°²·×ßàáâãäçèéêëìîòóôöøùúûüÿōœšʒ'
    mlt_char = ''.join(sorted( set(extra_MLT + latin_char) ))
    MLT_languages = ['English', 'French','German','Italian', 'Arabic']
elif MLT_lang == 'Latin+Arabic+Bangla':
    extra_MLT = "‘٬٫%$]€>@·°[ؤ=٠×،—ـ١٦٧٨٩٤٥٢٣ڥڤ'ٌ' ”“ِ 'ً'ڭڨ ُ َ~— ْ"
    extra_MLT = ''.join(sorted( set(ifn_char + extra_MLT) ))
    bangla_char = ' !"%\'()*-./0123456789:?achilmnprstuy`~।ঁংঃঅআইঈউএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ািীুূৃেৈোৌ্ৎড়য়০১২৩৪৫৬৭৮৯\u200c'
    extra_MLT = ''.join(sorted( set(bangla_char + extra_MLT) ))
    latin_char = '!"#$%&\'()*+-./0123456789:;<=>?€β@[\\]—¥_`abcdefghijklmnopqrstuvwxyz~°²·×ßàáâãäçèéêëìîòóôöøùúûüÿōœšʒ'
    mlt_char = ''.join(sorted( set(extra_MLT + latin_char) ))
    MLT_languages = ['English', 'French','German','Italian', 'Arabic', 'Bangla']
    MLT_latin_script_vs_others = True

    
    
if dataset_name == 'safe_driver' or dataset_name == 'Cifar100' or dataset_name =='imdb_movie' or dataset_name == 'WG' or dataset_name == 'TFSPCH' or dataset_name == 'cub2011': 
    phoc_unigrams = gw_char      # this depends on the alphabets used to name the classes, gw is English so that's fine 

elif dataset_name=='Cifar100+TFSPCH+GW+IFN':
    phoc_unigrams = wg_ifn_char       

elif dataset_name == 'Cifar100+TFSPCH+IAM+IFN' \
   or dataset_name == 'Cifar100+TFSPCH+IAM+IFN+safe-driver' \
   or dataset_name ==  'Cifar100+TFSPCH+IAM+IFN+safe-driver+imdb' \
   or dataset_name ==  'Cifar100+TFSPCH+IAM+IFN+safe-driver+imdb+cub2011' :         
   phoc_unigrams = iam_ifn_char    

elif dataset_name == 'Cifar100+TFSPCH+IAM+IFN+safe-driver+imdb+cub2011+MLT' :
    phoc_unigrams = ''.join(sorted( set(mlt_char + iam_ifn_char) ))    
   
elif dataset_name =='IFN':
    phoc_unigrams = ifn_char    

elif dataset_name == 'WG+IFN':    
    phoc_unigrams = wg_ifn_char       

elif dataset_name == 'IAM':    
    phoc_unigrams = iam_char
    
elif dataset_name == 'IAM+IFN':                 
    phoc_unigrams = iam_ifn_char
elif dataset_name=='MLT':
    phoc_unigrams = mlt_char    

else: 
    exit("Datasets to use: 'WG', 'IFN', 'IAM', 'WG+IAM', 'IAM+IFN', 'imdb_movie', 'TFSPCH' ")
            
del iam_char, ifn_char, gw_char, iam_ifn_char, wg_ifn_char, mlt_char


if task_type == 'script_identification': # label used for script identification/separation
    encoder = 'phoc'
    batch_size_train        = 2 #  = 10  # Prev works used 10 .....  a value of 2 gives better results
    model_name               = 'resnet18'
    testing_print_frequency  = 2 # 3 # prime number, how frequent to test/print during training
        
    unigram_levels = [2,4]  # we can rduce it here, probably no effect of using large levels
    
    batch_size_test   = 50  # Higher values may trigger memory problems
 

use_distortion_augmentor        = False

''' I need to remove these keep_flags.....later !!'''
keep_non_alphabet_of_GW_in_analysis       = True  # if True, it will be used in the analysis, else, it will be skipped from the phoc, even if has been loaded  
keep_non_alphabet_of_GW_in_loaded_data    = True 

# Save results
save_results           = False                            # Save Log file
results_path           = 'datasets/washingtondb-v1.0/results'  # Output folder to save the results of the test
IFN_based_on_folds_experiment  = False



'''
# testing based on folder sets, depriciated as no difference between our random split and this

IFN_test = 'set_a'
IFN_all_data_grouped_in_one_folder = True
IFN_based_on_folds_experiment  = False
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

if IFN_based_on_folds_experiment==True and dataset_name=='IFN':     
    split_percentage         = 1
    folders_to_use = 'abcde'   # 'eabcd' or 'abcd' in the publihsed papers, only abcd are used, donno why!?

'''


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


