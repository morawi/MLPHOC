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
    print('Output sent to file')
    dd = datetime.datetime.now()
    sys.stdout = open(cf.dataset_name +'_'+ cal.month_abbr[dd.month]+ '_' + str(dd.day),  'w')

xx = getmembers(cf)
for i in range(len(xx)): 
    print (xx[i])
print('------- the_hoc length is: ', len(cf.PHOC('', cf)) )
# test_dataload(cf) # Test the data loading of the different dataset
random_seeding(seed_value = cf.rnd_seed_value, use_cuda=True)

xx = ['set_a', 'set_b', 'set_c', 'set_d', 'set_e']
for set_xx in xx:
    cf.IFN_test=set_xx
    cf.dataset_path_IFN             = cf.folder_of_data + 'ifnenit_v2.0p1e/data/'+ cf.IFN_test +'/bmp/' # path to IFN images
    cf.gt_path_IFN                  = cf.folder_of_data + 'ifnenit_v2.0p1e/data/'+ cf.IFN_test + '/tru/' # path to IFN ground_truth

    result, train_set, test_set, train_loader, test_loader = test_cnn_finetune(cf) # Test CNN finetune with WG dataset
    print("Execution time is: ", time.time() - start_timer )
    # test_varoius_dist(result, cf) 
    
    test_varoius_thresholds(result, cf)
# word_str_mom = word_str_moment(result['word_str'])
# word_similarity = word_similarity_metric(result['word_str'])
#print('testing set word vect Moment is: ', word_str_mom)
#print('testing set word_similarity : ', word_similarity)

# reverting back to console

    

