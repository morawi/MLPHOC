#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 22:07:30 2018

@author: malrawi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 17:27:42 2018

@author: malrawi

TensorFlow Speech Recognition dataset

"""

import matplotlib.pyplot as plt
from matplotlib.backend_bases import RendererBase
from scipy import signal
from scipy.io import wavfile
#import soundfile as sf
import os
import numpy as np
from PIL import Image
from scipy.fftpack import fft

# matplotlib inline

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, _, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, np.log(spec.T.astype(np.float32) + eps)
    

     
def wav2img(wav_path):

    """     takes in wave file path    
    """
    
    # use soundfile library to read in the wave files
    samplerate, test_sound  = wavfile.read(wav_path)
    _, spectrogram = log_specgram(test_sound, samplerate)
       
    return spectrogram
    

# Looking at the top 9 different words in Spectrogram format
def top_9_different_words_spectogram(sample_audio):
    plt.figure(figsize=(10,10))
    
    # for each of the samples
    for i, filepath in enumerate(sample_audio[:9]):
        # Make subplots
        plt.subplot(3,3,i+1)
        
        # pull the labels
        label = filepath.split('/')[-2]
        plt.title(label)
        
        # create spectogram
        samplerate, test_sound  = wavfile.read(filepath)
        _, spectrogram = log_specgram(test_sound, samplerate)
        
        plt.imshow(spectrogram.T, aspect='auto', origin='lower')
        plt.axis('off')
        

folder_of_data         = '/home/malrawi/Desktop/My Programs/all_data/'
tf_speech_recognition_data = 'tf_speech_recognition/train/'
audio_path = folder_of_data + tf_speech_recognition_data + 'audio/' 
# test_pict_Path  = folder_of_data + tf_speech_recognition_data + 'images'

def get_file_names(audio_path):
    # #### Identify all the subdirectories in the training directory
    word_names = []   # word_names = subFolderList
    for x in os.listdir(audio_path):
        if os.path.isdir(audio_path + '/' + x):
            word_names.append(x)
            # if not os.path.exists(pict_Path + '/' + x):
            #    os.makedirs(pict_Path +'/'+ x)
                        
    '''
    if not os.path.exists(pict_Path):
        os.makedirs(pict_Path)
    
    if not os.path.exists(test_pict_Path):
        os.makedirs(test_pict_Path)
    '''
                  
    sample_audio = []    
    all_files = [] 
    all_words  = []
    for a_word_name in word_names:    
        # get all the wave files
        word_files = [y for y in os.listdir(audio_path + a_word_name) if '.wav' in y]
        all_files = all_files + word_files
        all_words =  all_words + [a_word_name]*len(word_files)
        # collect the first file from each dir
        sample_audio.append(audio_path  + a_word_name + '/'+ word_files[0])
        
        # show file counts
        print('count: %d : %s' % (len(word_files), a_word_name ))
    
    return all_files, all_words, sample_audio, word_names
 
    
    
all_files, all_words, sample_audio, word_names = get_file_names(audio_path)
# top_9_different_words_spectogram(sample_audio)
print('total', len(all_files))



for idx in range(5):
#        wav2img(audio_path + x + '/' + file, pict_Path + x)
    img = wav2img(audio_path + all_words[idx] + '/' + all_files[idx])
    plt.imshow(img.T, aspect='auto', origin='lower')
    plt.show()
    print(img.shape)

