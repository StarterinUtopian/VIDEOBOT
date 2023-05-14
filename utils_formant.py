# -*- coding: utf-8 -*-
import os
import sys
from required_functions.formant.formant import formant_forHAL
import json
import subprocess

def formantAnalysis(filename):
    sys.path.append(os.path.dirname(__file__))
    #print("formant start!")
    f1, f2 = formant_forHAL(filename)
    #print("haru formants ",f1,f2)
    #print("formant completed!")
    lst = []
    len1 = len(f1)
    len2 = len(f2)
    for index in range( max(len1, len2) ):
        if index+1 <= len1:
            lst += [f1[index]]
        if index+1 <= len2:
            lst += [f2[index]]
    return lst

def get_hal_path(formant_list):
    print("#=formant文字数=# : ", len(formant_list))
    print("pwd:",os.getcwd())
    if len(formant_list) <= 80 and len(formant_list) >= 2:
        movie2 = ['required_functions/lip_sync/new_vowels_r_24/{0}.mp4'.format(str(int(len(formant_list)/2))), True]
    elif len(formant_list) == 0:
        movie2 = ['required_functions/lip_sync/new_vowels_r_24/1.mp4', True]
    elif len(formant_list) > 80 and os.path.isfile("required_functions/lip_sync/new_vowels_r_24/{0}.mp4".format(len(formant_list)/2)) == True:
        movie2 = ['required_functions/lip_sync/new_vowels_r_24/{0}.mp4'.format(str(int(len(formant_list)/2))), True]
    elif len(formant_list) > 80 and os.path.isfile("required_functions/lip_sync/new_vowels_r_24/{0}.mp4".format(len(formant_list)/2)) == False:
        surplus = len(formant_list) - 80
        cmd = "./required_functions/lip_sync/hal_concat.sh required_functions/lip_sync/new_vowels_r_24/40.mp4 required_functions/lip_sync/new_vowels_r_24/{0}.mp4 required_functions/lip_sync/new_vowels_r_24/{1}.mp4".format(str(int(surplus/2)), str(int(len(formant_list)/2)))
        print("cmd:",cmd)
        print("~"*50)
        subprocess.call(cmd,shell=True)
        movie2 = ['required_functions/lip_sync/new_vowels_r_24/{0}.mp4'.format(str(int(len(formant_list)/2))), True]
    else:
        print("#=formantの数が奇数=#")
        raise ValueError("#=formantの数が奇数=#")

    #print("new_vowel:",movie2)

    return movie2

