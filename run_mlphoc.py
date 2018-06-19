from __future__ import division

import logging

from config.load_config_file import Configuration
from tests.test_dataset import test_dataload
from tests.test_cnn_finetune import test_cnn_finetune

def run_mlphoc(cf):
    logger = logging.getLogger(__name__)

    # Test the data loading of the different dataset
    # test_dataload(cf)

    # Test CNN finetune with WG dataset
    test_cnn_finetune(cf)


def main():

    # Uncomment the following code to use argument parser
    # Use python run_mlphoc.py --config-path config/config_file.py --test-name test_name

    # # Get parameters from arguments
    # parser = argparse.ArgumentParser(description='MLPHOC')
    # parser.add_argument('-c', '--config-path', type=str, required=True, help='Configuration file path')
    # parser.add_argument('-t', '--test-name', type=str, required=True, help='Name of the test')
    #
    # arguments = parser.parse_args()
    #
    # assert arguments.config_path is not None, 'Please provide a configuration' \
    #                                           'path using -c config/path/name' \
    #                                           ' in the command line'
    # assert arguments.test_name is not None, 'Please provide a name for the ' \
    #                                         'test using -e test_name in the ' \
    #                                         'command line'
    #
    # config_path = arguments.config_path
    # test_name = arguments.test_name

    # Input arguments
    config_path = 'config/config_file_wg.py'
    test_name = '' # Optional: name of the sub folder where to store the results

    # Load the configuration file
    configuration = Configuration(config_path, test_name)
    cf = configuration.load()

    # Run task
    run_mlphoc(cf)


# Entry point of the script
if __name__ == "__main__":
    main()