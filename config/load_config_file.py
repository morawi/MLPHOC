from __future__ import print_function

import imp
import os
import datetime
import calendar as cal
from inspect import getmembers
import sys



class Configuration(object):
    def __init__(self, config_path, test_name=''):

        self.config_path = config_path
        self.test_name = test_name
        self.configuration = None
        
        
    def  print_all_parameters(self, cf):
        # Printing out all parameters of the problem
        # redirect printed results to ouptut file
        if cf.redirect_std_to_file:    
            dd = datetime.datetime.now()
            out_file_name = cf.dataset_name +'_'+ cal.month_abbr[dd.month]+ '_' + str(dd.day)
            print('Output sent to ', out_file_name)
            sys.stdout = open(out_file_name,  'w')
    
        xx = getmembers(cf)
        for i in range(len(xx)): 
            print (xx[i])
        if cf.encoder == 'script_identification': # 'label only works when we have two scripts, dataset= IAM+IFN or WG+IFN'
            print('Script identification experiment')
        else:
            print('------- the_hoc length is: ', len(cf.PHOC(cf.phoc_unigrams[1:3], cf)) ) # cf.phoc_unigrams[1:3] just passing some chars/unigrams to get the PHOC size
        if cf.overlay_handwritting_on_STL_img:
            print('------ Scene Handwritting Experiment ----------')
            
            
    def load(self, print_vals=False):

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
        
        if print_vals:
            self.print_all_parameters(cf)
        
        return cf
    
    