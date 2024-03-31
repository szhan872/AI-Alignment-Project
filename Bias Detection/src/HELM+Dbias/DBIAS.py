# !pip install Dbias
# !pip install https://huggingface.co/d4data/en_pipeline/resolve/main/en_pipeline-any-py3-none-any.whl

from Dbias.text_debiasing import *;
from Dbias.bias_classification import *;
from Dbias.bias_recognition import *;
from Dbias.bias_masking import *;
import os
import pandas as pd

def custom_classification(x):
    classi_out = classify(x)
    return classi_out[0]['label'], classi_out[0]['score']

def custom_recognizer(x):
    biased_words = recognizer(x)
    biased_words_list = []
    for id in range(0, len(biased_words)):
        biased_words_list.append(biased_words[id]['entity'])
    return ", ".join(biased_words_list)

def custom_debiasing(x):
    suggestions = run(x)
    if suggestions == None:
        return ""
    else:
      all_suggestions = []
      for sent in suggestions[0:3]:
        all_suggestions.append(sent['Sentence'])
      return "\n\n".join(all_suggestions)
    
    