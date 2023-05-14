# -*- coding: utf-8 -*-

vowels = ["a","i","u","e","o","A","I","U","E","O"]
alphabet = []
co = []
ascii_letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPRSTUVWXYZ'
for i in range(len(ascii_letters)):
    alphabet.append(ascii_letters[i])
co = sorted(list(set(alphabet) - set(vowels)))

"""
#tacotron2の入力データから日本語の言葉ごとに分けたインデックスを取得する関数
def roman2index(text):
    out_box = []
    i = 0
    while True:
        if i >= len(text)-1:
            break
        if text[i] == "*":
            i += 1
            #print("a",i)
            
        out_box.append(i)
        #ん
        if text[i] == "N":
            i += 1
            #print("b",i)
            
        elif text[i] in vowels:
            i += 1
            #print("c",i)
        elif text[i] == "n":
            if text[i+1] in vowels:
                i += 2
                #print("c",i)
                
            elif text[i+1] == "y":
                i += 3
                #print("d",i)
                
            elif text[i+1] == "*":
                i += 2  
                #print("e",i)
            else:
                i += 1
                #print("f",i)
        elif text[i] in co:
            if text[i+1] in vowels:
                i += 2
                #print("g",i)
            elif text[i+1] == "h" or text[i+1] == "y" or text[i+1] == "s" or text[i+1] == "w":
                i += 3
                #print("h",i)
                
        elif text[i] == "Q":
            i += 1
            #print("i",i)

        elif text[i] == " ":
            i += 1 
            
    return out_box
"""


def roman2index(input_text):
    cut_list = [0]
    last_index = len(input_text)-2
    for i, char in enumerate(input_text):
        if i == last_index:
            break

        if char == 'a' or char == 'A':
            if input_text[i+1] == "*":
                cut_list.append(i+2)
                #char_list.append(tmp)
                #print("*ari")
            else:
                cut_list.append(i+1)
                #char_list.append(tmp)
            #tmp = ''
        elif char == 'i' or char == 'I':
            if input_text[i+1] == "*":
                cut_list.append(i+2)
                #char_list.append(tmp)
                #print("*ari")
            else:
                cut_list.append(i+1)
                #char_list.append(tmp)
            #tmp = ''
        elif char == 'u' or char == 'U':
            if input_text[i+1] == "*":
                cut_list.append(i+2)
                #char_list.append(tmp)
                #print("*ari")
            else:
                cut_list.append(i+1)
                #char_list.append(tmp)
            #tmp = ''
        elif char == 'e' or char == 'E':
            if input_text[i+1] == "*":
                cut_list.append(i+2)
                #char_list.append(tmp)
                #print("*ari")
            else:
                cut_list.append(i+1)
                #char_list.append(tmp)
            #tmp = ''
        elif char == 'o' or char == 'O':
            if input_text[i+1] == "*":
                cut_list.append(i+2)
                #char_list.append(tmp)
                #print("*ari")
            else:
                cut_list.append(i+1)
                #char_list.append(tmp)
            #tmp = ''
        elif char == 'N':
            if input_text[i+1] == "*":
                cut_list.append(i+2)
                #char_list.append(tmp)
                #print("*ari")
            else:
                cut_list.append(i+1)
                #char_list.append(tmp)
            #tmp = ''
        elif char == 'Q':
            if input_text[i+1] == "*":
                cut_list.append(i+2)
                #char_list.append(tmp)
                #rint("*ari")
            else:
                cut_list.append(i+1)
                #char_list.append(tmp)
            #tmp = ''
        elif char == ' ':
            if input_text[i+1] == "*":
                cut_list.append(i+2)
                #char_list.append(tmp)
                #print("*ari")
            else:
                cut_list.append(i+1)
                #char_list.append(tmp)
            #tmp = ''
        else:
            continue
    return cut_list