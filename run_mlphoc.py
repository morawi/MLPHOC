from __future__ import division
import time
import logging

from config.load_config_file import Configuration
from tests.test_cnn_finetune import test_cnn_finetune
from utils.some_functions import random_seeding, test_varoius_thresholds, word_str_moment, word_similarity_metric #test_varoius_dist, 
from inspect import getmembers
# from tests.test_dataset import test_dataload



start_timer = time.time()
config_path = 'config/config_file_wg.py'
test_name = '' # Optional: name of the sub folder where to store the results

logger = logging.getLogger(__name__)
# Load the configuration file
configuration = Configuration(config_path, test_name)
cf = configuration.load()
xx = getmembers(cf)
for i in range(len(xx)): 
    print (xx[i])
print('------- the_hoc length is: ', len(cf.PHOC('', cf)) )
# test_dataload(cf) # Test the data loading of the different dataset
random_seeding(seed_value = cf.rnd_seed_value, use_cuda=True)
result, train_set, test_set, train_loader, test_loader = test_cnn_finetune(cf) # Test CNN finetune with WG dataset
print("Execution time is: ", time.time()-start_timer )
# test_varoius_dist(result, cf) 

test_varoius_thresholds(result, cf)
# word_str_mom = word_str_moment(result['word_str'])
# word_similarity = word_similarity_metric(result['word_str'])
#print('testing set word vect Moment is: ', word_str_mom)
#print('testing set word_similarity : ', word_similarity)




