import requests
import sys, os
import time


def translate_deepl(text, source_lang, target_lang):
    with open("./required_functions/deepl_api_key.txt","r") as f:
        API_KEY = f.readlines()[0].replace("\n","")
    source_lang = source_lang.upper()
    target_lang = target_lang.upper()
    params = {
        "auth_key":API_KEY,
        "text":text,
        "source_lang":source_lang,
        "target_lang":target_lang
    }
    request = requests.post("https://api.deepl.com/v2/translate", data=params)
    result = request.json()
    return result["translations"][0]["text"]

if __name__ == "__main__":
    #input_text = input("input<<")
    with open("test.txt","r") as f:
        input_text = f.read()
    print(input_text)
    translate_result = translate_deepl(input_text, "ja", "zh")
    print(translate_result)
