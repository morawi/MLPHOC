from __future__ import print_function

import imp
import os


class Configuration(object):
    def __init__(self, config_path, test_name=''):

        self.config_path = config_path
        self.test_name = test_name
        self.configuration = None

    def load(self):

        # Get Config path
        print(self.config_path)
        config_path = os.path.join(os.getcwd(), os.path.dirname(self.config_path),
                                   os.path.basename(self.config_path))
        print('Config file loaded: ')
        print(config_path)

        cf = imp.load_source('config', config_path)

        # Save extra parameter
        cf.config_path = config_path
        cf.test_name = self.test_name

        if cf.save_results:

            # Output folder
            if self.test_name == '':
                cf.results_path = os.path.abspath(cf.results_path)
            else:
                cf.results_path = os.path.join(os.path.abspath(cf.results_path), cf.test_name)

            if not os.path.exists(cf.results_path):
                os.makedirs(cf.results_path)

            print ('Save results in: ' + cf.results_path)
            cf.log_file = os.path.join(cf.results_path, "logfile.log")

        self.configuration = cf
        return cf