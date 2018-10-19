
''' 
Main @author: malrawi 

'''


from __future__ import division
import time
import logging

from config.load_config_file import Configuration
from tests.test_cnn_finetune import test_cnn_finetune
from utils.some_functions import random_seeding, test_varoius_thresholds, word_str_moment, word_similarity_metric #test_varoius_dist, 
from inspect import getmembers

import sys
import calendar as cal
import datetime
# from tests.test_dataset import test_dataload

start_timer = time.time()
config_path = 'config/config_file_wg.py'
test_name = '' # Optional: name of the sub folder where to store the results

logger = logging.getLogger(__name__)
# Load the configuration file
configuration = Configuration(config_path, test_name)
cf = configuration.load()

# redirect printed results to ouptut file
if cf.redirect_std_to_file:    
    dd = datetime.datetime.now()
    out_file_name = cf.dataset_name +'_'+ cal.month_abbr[dd.month]+ '_' + str(dd.day)
    print('Output sent to ', out_file_name)
    sys.stdout = open(out_file_name,  'w')

xx = getmembers(cf)
for i in range(len(xx)): 
    print (xx[i])
print('------- the_hoc length is: ', len(cf.PHOC('', cf)) )
if cf.overlay_handwritting_on_STL_img:
    print('------ Scene Handwritting Experiment ----------')
random_seeding(seed_value = cf.rnd_seed_value, use_cuda=True)


if cf.IFN_based_on_folds_experiment==True and cf.dataset_name=='IFN':
    for _xx in cf.folders_to_use:
        cf.IFN_test = 'set_'+_xx
        print('\n ##########      using', cf.IFN_test, 'for testing \n')
        cf.dataset_path_IFN  = cf.folder_of_data + 'ifnenit_v2.0p1e/data/'+ cf.IFN_test +'/bmp/' # path to IFN images
        cf.gt_path_IFN       = cf.folder_of_data + 'ifnenit_v2.0p1e/data/'+ cf.IFN_test + '/tru/' # path to IFN ground_truth    
        result, train_set, test_set, train_loader, test_loader = test_cnn_finetune(cf) # Test CNN finetune with WG dataset
        
        # test_varoius_dist(result, cf) 
else:
    result, train_set, test_set, train_loader, test_loader = test_cnn_finetune(cf) # Test CNN finetune with WG dataset    

print("Execution time is: ", time.time() - start_timer )
    
  
# test_varoius_thresholds(result, cf)  
 
# word_str_mom = word_str_moment(result['word_str'])
# word_similarity = word_similarity_metric(result['word_str'])
#print('testing set word vect Moment is: ', word_str_mom)
#print('testing set word_similarity : ', word_similarity)

# reverting back to console

    

