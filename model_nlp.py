



# Hugginface Bias Detection Model

import pandas as pd

import numpy as np

import tensorflow as tf


from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline
tokenizer = AutoTokenizer.from_pretrained("d4data/bias-detection-model")
model = TFAutoModelForSequenceClassification.from_pretrained("d4data/bias-detection-model")

classifier = pipeline('text-classification', model=model, tokenizer=tokenizer) # cuda = 0,1 based on gpu availability



classifier("The irony, of course, is that the exhibit that invites people to throw trash at vacuuming Ivanka Trump lookalike reflects every stereotype feminists claim to stand against, oversexualizing Ivanka’s body and ignoring her hard work.")

# build a function to return the bias score

def bias_score(text):
    result = classifier(text)
    return result[0]['score']

# test the function

bias_score("The irony, of course, is that the exhibit that invites people to throw trash at vacuuming Ivanka Trump lookalike reflects every stereotype feminists claim to stand against, oversexualizing Ivanka’s body and ignoring her hard work.")

# Do it for a example dataset bias from Hugginface


#install Dbias
#install https://huggingface.co/d4data/en_pipeline/resolve/main/en_pipeline-any-py3-none-any.whl


from Dbias.text_debiasing import *;
from Dbias.bias_classification import *;
from Dbias.bias_recognition import *;
from Dbias.bias_masking import *;
import os
import pandas as pd

df_bias = pd.read_csv('sample data.csv')


df_bias.head()


# example applying this tool using GoogleNews: https://github.com/dreji18/Fairness-in-AI/blob/main/example%20notebooks/Dbias_newsapi.ipynb

