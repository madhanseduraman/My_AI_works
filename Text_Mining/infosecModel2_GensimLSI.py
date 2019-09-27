
# coding: utf-8

# In[ ]:


#####################
# gensim LSIModel   #
#####################


# In[1]:


# Semantic Smiliarity
from __future__ import division
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
import math
import numpy as np
import sys
# For removing stop words,lemmatization
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import openpyxl
import pandas as pd
import gensim 
import re as re


# In[2]:


#Stopwords removal
stoplist = set(stopwords.words('english')) 

porter_stemmer = PorterStemmer()
lm = WordNetLemmatizer()

def cleanData(sentence):
    
    # convert to lowercase, ignore all special characters - keep only alpha-numericals and spaces (not removing full-stop here)
    sentence = re.sub(r'[^A-Za-z0-9\s.]',r'',str(sentence).lower())
    sentence = re.sub(r'\n',r' ',sentence)
    
    # remove stop words
    #sentence = " ".join([word for word in sentence.split() if word not in stoplist])
    sentence = " ".join([lm.lemmatize(porter_stemmer.stem(word)) for word in sentence.split() if word not in stoplist])
    return sentence


# In[3]:


from collections import Counter

WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)


# ---

# In[4]:


###################
# Gensim LSI Model#
###################

from gensim import corpora, models, similarities
from collections import defaultdict

QuestionBank = pd.read_excel("C:\Girish\GDSAutomationCentral\EYGDSAC_AnalyticsInitiatives\GDSAnalytics\Infosec_Questionnaire\Colated_Questionnaire.xlsx",sheet_name="Q&A")

inputquestions = pd.read_excel("C:\Girish\GDSAutomationCentral\EYGDSAC_AnalyticsInitiatives\GDSAnalytics\Infosec_Questionnaire\InputQuestions.xlsx",sheet_name="Questions")

#QuestionBank['Combined'] = QuestionBank['Questions'].fillna('') + ' ' + QuestionBank['Response'].fillna('') + ' ' +QuestionBank['Procedure_Process_Used'].fillna('') + ' ' + QuestionBank['Supporting_documentation'].fillna('') + ' ' + QuestionBank['Additional_Comments'].fillna('') +  ' ' + QuestionBank['Topic'].fillna('') 
   
QuestionBank['Combined'] = QuestionBank['Questions'].fillna('')
    
QuestionBank['Combined'] = QuestionBank['Combined'].map(lambda x: cleanData(x))
    
documents = QuestionBank['Combined']

texts = [[word.lower() for word in document.split()
          if word.lower() not in stoplist]
         for document in documents]

frequency = defaultdict(int)

for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1]
         for text in texts]
dictionary = corpora.Dictionary(texts)
 
corpus = [dictionary.doc2bow(text) for text in texts]
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)    
       
colnames = ['Index','Score']

df = pd.DataFrame()

for q in range(len(inputquestions['Questions'])):
    questionTocheck = inputquestions['Questions'][q]
    questionTocheck = cleanData(questionTocheck)
    tempdf = pd.DataFrame()
    vec_bow = dictionary.doc2bow(questionTocheck.lower().split())
    # convert the query to LSI space
    vec_lsi = lsi[vec_bow]
    index = similarities.MatrixSimilarity(lsi[corpus])    
    # perform a similarity query against the corpus
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])    
    Gensim_OP = pd.DataFrame(sims, columns=['Index','Score'])
    Gensim_OP["Input_Quest"] = inputquestions['Questions'][q]
    Gensim_OP = Gensim_OP.sort_values('Score',ascending = False).groupby('Input_Quest').head(5)
    df = df.append(Gensim_OP)


# In[5]:


###################################
# Cosine Similarity of Gensim o/p #
###################################

QuestionBank['Index'] = QuestionBank.index  
result = pd.merge(df, QuestionBank, how='left', left_on=['Index'], right_on=['Index'])
result.drop('Index',axis=1,inplace=True)
result.drop('No',axis=1,inplace=True)
result.reset_index(drop=True,inplace=True)
cosineList = []
for r in range(len(result['Questions'])):
    #vector1 = text_to_vector(result['Input_Quest'][r])
    #vector2 = text_to_vector(result['Questions'][r])
    vector1 = text_to_vector(cleanData(result['Input_Quest'][r]))
    vector2 = text_to_vector(cleanData(result['Questions'][r]))
    cosine = get_cosine(vector1, vector2)
    cosineList.append(cosine)
result["CosineSimilarity"] = cosineList
result.to_excel('C:\Girish\GDSAutomationCentral\EYGDSAC_AnalyticsInitiatives\GDSAnalytics\Infosec_Questionnaire\Model2\Model2A_Intermediate_Output.xlsx')


# In[6]:


df_answered = result.sort_values('CosineSimilarity',ascending = False).groupby('Input_Quest').head(1)
df_answered.drop('Questions', axis=1, inplace=True)
df_answered.drop('Combined', axis=1, inplace=True)
df_answered.drop('Score', axis=1, inplace=True)
#df_answered.drop('Index', axis=1, inplace=True)
df_answered.drop('CosineSimilarity', axis=1, inplace=True)
df_answered.to_excel('C:\Girish\GDSAutomationCentral\EYGDSAC_AnalyticsInitiatives\GDSAnalytics\Infosec_Questionnaire\Model2\Model2A_Answers.xlsx',index = False)


# ---
