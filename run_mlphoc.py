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

# Load the configuration file
configuration = Configuration(config_path, test_name)
cf = configuration.load()

logger = logging.getLogger(__name__)
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




#def run_mlphoc(cf):
#    logger = logging.getLogger(__name__)
#
#    # Test the data loading of the different dataset
#    # test_dataload(cf)
#
#    # Test CNN finetune with WG dataset
#    test_cnn_finetune(cf)
#
#
#def main():
#
#    # Uncomment the following code to use argument parser
#    # Use python run_mlphoc.py --config-path config/config_file.py --test-name test_name
#
#    # # Get parameters from arguments
#    # parser = argparse.ArgumentParser(description='MLPHOC')
#    # parser.add_argument('-c', '--config-path', type=str, required=True, help='Configuration file path')
#    # parser.add_argument('-t', '--test-name', type=str, required=True, help='Name of the test')
#    #
#    # arguments = parser.parse_args()
#    #
#    # assert arguments.config_path is not None, 'Please provide a configuration' \
#    #                                           'path using -c config/path/name' \
#    #                                           ' in the command line'
#    # assert arguments.test_name is not None, 'Please provide a name for the ' \
#    #                                         'test using -e test_name in the ' \
#    #                                         'command line'
#    #
#    # config_path = arguments.config_path
#    # test_name = arguments.test_name
#
#    # Input arguments
#    start_timer = time.time()
#    config_path = 'config/config_file_wg.py'
#    test_name = '' # Optional: name of the sub folder where to store the results
#
#    # Load the configuration file
#    configuration = Configuration(config_path, test_name)
#    cf = configuration.load()
#
#    # Run task
#    run_mlphoc(cf)
#    
#    print("Execution time is: ", time.time()-start_timer ) 
#
#
## Entry point of the script
#if __name__ == "__main__":
#    # cf = main()
#   
#    main()