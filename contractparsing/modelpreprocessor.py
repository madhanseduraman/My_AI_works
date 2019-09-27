# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:36:46 2019

@author: Archana.Muraly
"""

import spacy

import string
import re


from nltk.stem.snowball import SnowballStemmer
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

nlp = spacy.load("en_core_web_sm")
stop_words = spacy.lang.en.stop_words.STOP_WORDS
punctuations = string.punctuation

nlp.Defaults.stop_words |= {'EY','Supplier','Customer','Licensor','Party','EYGS','EYG','Vendor','parties'}

nlp.Defaults.stop_words -= {'will', 'shall','hereby'}

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()


# In[64]:


#print(stop_words)


# In[65]:


def sentence_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)
    stemmer = SnowballStemmer(language='english')
      
    mytokens = [stemmer.stem(word.lower_.strip()) for word in mytokens ]

    # Removing stop words
    # mytokens = [stemmer.stem(word.lower_.strip()) for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words]
    mytokens = [ word for word in mytokens if word not in punctuations]  
    # return preprocessed list of tokens
    return ' '.join(mytokens)


# In[66]:


# # Custom transformer using spaCy
# class predictors(TransformerMixin):
#     def transform(self, X, **transform_params):
#         # Cleaning Text
#         return [clean_text(text) for text in X]

#     def fit(self, X, y=None, **fit_params):
#         return self

#     def get_params(self, deep=True):
#         return {}

# Basic function to clean the text
def clean_text(text):
    text = sentence_tokenizer(text)
    cleantext = text.strip().lower()
    pattern = '[0-9]'
    cleantext = re.sub(pattern, '', cleantext) 
    pattern = '[,?/+.]'
    cleantext = re.sub(pattern, '', cleantext) 
    return cleantext
