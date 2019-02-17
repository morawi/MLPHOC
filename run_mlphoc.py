
''' 
Main @author: malrawi 

Major run function:
    Load the experiment parameters and desing in config_file_wg.py

'''


from __future__ import division
import time
import logging

from config.load_config_file import Configuration
from tests.test_cnn_finetune import test_cnn_finetune
from utils.some_functions import random_seeding # test_varoius_thresholds, word_str_moment, word_similarity_metric, test_varoius_dist, 

start_timer = time.time()
logger = logging.getLogger(__name__)
configuration = Configuration(config_path='config/config_file_wg.py', test_name='') # test_name: Optional name of the sub folder where to store the results
cf = configuration.load(print_vals=True)
random_seeding(seed_value = cf.rnd_seed_value, use_cuda=True)
result = test_cnn_finetune(cf)  
del result
# test_varoius_dist(result, cf) 
print("Execution time is: ", time.time() - start_timer )
    

  
# test_varoius_thresholds(result, cf)  
 
# word_str_mom = word_str_moment(result['word_str'])
# word_similarity = word_similarity_metric(result['word_str'])
#print('testing set word vect Moment is: ', word_str_mom)
#print('testing set word_similarity : ', word_similarity)

    
''' Depreciated, has no effect 
if cf.IFN_based_on_folds_experiment==True and cf.dataset_name=='IFN':
    for _xx in cf.folders_to_use:
        cf.IFN_test = 'set_'+_xx
        print('\n ##########      using', cf.IFN_test, 'for testing \n')
        cf.dataset_path_IFN  = cf.folder_of_data + 'ifnenit_v2.0p1e/data/'+ cf.IFN_test +'/bmp/' # path to IFN images
        cf.gt_path_IFN       = cf.folder_of_data + 'ifnenit_v2.0p1e/data/'+ cf.IFN_test + '/tru/' # path to IFN ground_truth    
        result = test_cnn_finetune(cf) # Test CNN finetune with WG dataset
                
else:

'''
    