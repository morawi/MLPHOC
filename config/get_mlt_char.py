#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:03:24 2019

@author: malrawi
"""

''' Language / script dataset to use '''       




def get_char_set(MLT_lang=''):  
   
   char_set = {}
   iam_char = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', 
            '/', ':', ';', '?', '_',  
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 
            'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 
            'w', 'x', 'y', 'z'] # upper case removed
   iam_char = ''.join(map(str, iam_char))
   ifn_char = "0123456789أءابجدهوزطحيكلمنسعفصقرشتثخذضظغةى.ئإآ\'ّ''"
   gw_char = ".0123456789abcdefghijklmnopqrstuvwxyz,-;':()£|"
   iam_ifn_char = ''.join(sorted(set(iam_char + ifn_char))) 
   wg_ifn_char = ''.join(sorted( set(ifn_char + gw_char) )) 

   
   MLT_latin_script_vs_others = False
   
   if MLT_lang=='MLT_English+Instagram_test':
       MLT_languages = ['English']
       extra_MLT = "‘°\٬%$]€>@·[ؤ=٠×—،~ـ“ِ '"    
       mlt_char = ''.join(sorted( set(iam_char + extra_MLT) ))
       mlt_char = ''.join(sorted( set(gw_char + mlt_char) ))       
   
   elif MLT_lang=='Arabic':
       MLT_languages = ['Arabic']
       extra_MLT = "‘٬٫%$]€>:+/#!()@·°[ؤ=٠×،ـ١٦٧٨٩٤٥٢٣ڥ出ڤ'ٌ' ”“ِ 'ً'ڭڨ ُ َ~— ْ"
       mlt_char = ''.join(sorted( set(ifn_char + extra_MLT) ))
   
   elif MLT_lang=='English':
       MLT_languages = ['English']
       extra_MLT = "‘°\٬%$]€>@·[ؤ=٠×—،~ـ“ِ '"    
       mlt_char = ''.join(sorted( set(iam_char + extra_MLT) ))
   
   elif MLT_lang == 'Arabic+English':  
       extra_MLT = "‘٬٫%$]€>@·°[ؤ=٠×،—ـ١٦٧٨٩٤٥٢٣ڥڤ'ٌ' ”“ِ 'ً'ڭڨ ُ َ~— ْ"
       MLT_languages = ['English', 'Arabic']
       mlt_char = ''.join(sorted( set(iam_ifn_char + extra_MLT) ))
   elif MLT_lang=='Bangla':
       mlt_char = ' !"%\'()*-./0123456789:?achilmnprstuy`~।ঁংঃঅআইঈউএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ািীুূৃেৈোৌ্ৎড়য়০১২৩৪৫৬৭৮৯\u200c'
       MLT_languages = ['Bangla']
   elif MLT_lang == 'Eng+Ara+Bang':
       extra_MLT = "‘٬٫%$]€>@·°[ؤ=٠×،—ـ١٦٧٨٩٤٥٢٣ڥڤ'ٌ' ”“ِ 'ً'ڭڨ ُ َ~— ْ"
       extra_MLT = ''.join(sorted( set(ifn_char + extra_MLT) ))
       bangla_char = ' !"%\'()*-./0123456789:?achilmnprstuy`~।ঁংঃঅআইঈউএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ািীুূৃেৈোৌ্ৎড়য়০১২৩৪৫৬৭৮৯\u200c'
       extra_MLT = ''.join(sorted( set(bangla_char + extra_MLT) ))
       latin_char = '!"#$%&\'()*+-./0123456789:;<=>?€β@[\\]—¥_`abcdefghijklmnopqrstuvwxyz'
       mlt_char = ''.join(sorted( set(extra_MLT + latin_char) ))
     
       MLT_languages = ['English', 'Arabic', 'Bangla']
   
   elif MLT_lang == 'Latin' or MLT_lang=='':
       mlt_char = '!"#$%&\'()*+-./0123456789:;<=>?€β@[\\]—¥_`abcdefghijklmnopqrstuvwxyz~°²·×ßàáâãäçèéêëìîòóôöøùúûüÿōœšʒ'
       MLT_languages = ['English', 'French','German','Italian']
   
   elif MLT_lang == 'Latin+Arabic':
       extra_MLT = "‘٬٫%$]€>@·°[ؤ=٠×،—ـ١٦٧٨٩٤٥٢٣ڥڤ'ٌ' ”“ِ 'ً'ڭڨ ُ َ~— ْ"
       extra_MLT = ''.join(sorted( set(ifn_char + extra_MLT) ))
       latin_char = '!"#$%&\'()*+-./0123456789:;<=>?€β@[\\]—¥_`abcdefghijklmnopqrstuvwxyz~°²·×ßàáâãäçèéêëìîòóôöøùúûüÿōœšʒ'
       mlt_char = ''.join(sorted( set(extra_MLT + latin_char) ))
       MLT_languages = ['English', 'French','German','Italian', 'Arabic']
   elif MLT_lang == 'Latin+Arabic+Bangla':
       extra_MLT = "‘٬٫%$]€>@·°[ؤ=٠×،—ـ١٦٧٨٩٤٥٢٣ڥڤ'ٌ' ”“ِ 'ً'ڭڨ ُ َ~— ْ"
       extra_MLT = ''.join(sorted( set(ifn_char + extra_MLT) ))
       bangla_char = ' !"%\'()*-./0123456789:?achilmnprstuy`~।ঁংঃঅআইঈউএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ািীুূৃেৈোৌ্ৎড়য়০১২৩৪৫৬৭৮৯\u200c'
       extra_MLT = ''.join(sorted( set(bangla_char + extra_MLT) ))
       latin_char = '!"#$%&\'()*+-./0123456789:;<=>?€β@[\\]—¥_`abcdefghijklmnopqrstuvwxyz~°²·×ßàáâãäçèéêëìîòóôöøùúûüÿōœšʒ'
       mlt_char = ''.join(sorted( set(extra_MLT + latin_char) ))
       MLT_languages = ['English', 'French','German','Italian', 'Arabic', 'Bangla']
       MLT_latin_script_vs_others = True
    
   
   char_set['iam_char'] = iam_char
   char_set['ifn_char'] = ifn_char
   char_set['iam_ifn_char'] = iam_ifn_char
   char_set['wg_ifn_char']= wg_ifn_char
   char_set['gw_char'] = gw_char
   
   return char_set, MLT_languages,  MLT_latin_script_vs_others