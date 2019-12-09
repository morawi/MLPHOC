#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:07:59 2019

@author: malrawi
"""


def djb2_hash(s):                                                                                                                                
    hash = 5381
    for x in s:
        hash = (( hash << 5) + hash) + ord(x)
    return hex( hash  & 0xFFFFFFFF )
   

#x = djb2_hash('skyspeech')
#print(x)


#def DJB2_hash(c):
#    while (c != '\0'){
#    		c = my_str[i];
#    	    hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
#    	    i=i+1;
#    
#    	}
