
# coding: utf-8

# In[ ]:


#######################
# Gensim Word2Vec     #
#######################


# In[1]:


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
from gensim.models import Word2Vec
model = gensim.models.KeyedVectors.load_word2vec_format("C:\Girish\GDSAutomationCentral\EYGDSAC_AnalyticsInitiatives\GDSAnalytics\Infosec_Questionnaire\Model2\GoogleNews-vectors-negative300.bin.gz", binary=True)


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


# In[8]:


QuestionBank = pd.read_excel("C:\Girish\GDSAutomationCentral\EYGDSAC_AnalyticsInitiatives\GDSAnalytics\Infosec_Questionnaire\Colated_Questionnaire.xlsx",sheet_name="Q&A")

inputquestions = pd.read_excel("C:\Girish\GDSAutomationCentral\EYGDSAC_AnalyticsInitiatives\GDSAnalytics\Infosec_Questionnaire\InputQuestions.xlsx",sheet_name="Questions")

#QuestionBank['Combined'] = QuestionBank['Questions'].fillna('') + ' ' + QuestionBank['Response'].fillna('') + ' ' +QuestionBank['Procedure_Process_Used'].fillna('') + ' ' + QuestionBank['Supporting_documentation'].fillna('') + ' ' + QuestionBank['Additional_Comments'].fillna('') +  ' ' + QuestionBank['Topic'].fillna('') 

QuestionBank['Combined'] = QuestionBank['Questions'].fillna('')
#QuestionBank['Combined'] = QuestionBank['Combined'].map(lambda x: cleanData(x))

dataset = pd.DataFrame()

for q in range(len(inputquestions['Questions'])):
    tempdf = pd.DataFrame()
    questionTocheck = inputquestions['Questions'][q]
    #questionTocheck = cleanData(questionTocheck)
    tempdf['sent_2'] = QuestionBank['Combined']
    tempdf['sent_1'] = questionTocheck
    tempdf['Questions'] = QuestionBank['Questions']
    tempdf['Response'] = QuestionBank['Response']
    tempdf['Procedure_Process_Used'] = QuestionBank['Procedure_Process_Used']
    tempdf['Supporting_documentation'] = QuestionBank['Supporting_documentation']
    tempdf['Additional_Comments'] = QuestionBank['Additional_Comments']
    tempdf['Topic'] = QuestionBank['Topic']    
    tempdf['Input_Quest'] = inputquestions['Questions'][q]
    dataset = dataset.append(tempdf)

dataset.reset_index(drop = True,inplace = True)


# In[9]:


#dataset
model.init_sims()
model.init_sims(replace=True)
score = []
for i in range(len(dataset['Questions'])) :
    s1 = cleanData(dataset['sent_1'][i])
    s2 = cleanData(dataset['sent_2'][i])
#    distance = model.wmdistance(s1, s2)
    distance = model.wmdistance(s1.split(), s2.split())
    score.append(distance)
dataset['Scores'] = score


# In[13]:


datasetTop5 = pd.DataFrame()
datasetTop5 = datasetTop5.append(dataset.sort_values('Scores',ascending = True).groupby('Input_Quest').head(10))
datasetTop5.reset_index(drop=True,inplace=True)
datasetTop5 = datasetTop5[['Input_Quest','Questions','Response','Procedure_Process_Used','Supporting_documentation','Additional_Comments','Topic','Scores']]
cosineList = []
for r in range(len(datasetTop5['Questions'])):
    vector1 = text_to_vector(cleanData(datasetTop5['Input_Quest'][r]))
    vector2 = text_to_vector(cleanData(datasetTop5['Questions'][r]))
    cosine = get_cosine(vector1, vector2)
    cosineList.append(cosine)
datasetTop5["CosineSimilarity"] = cosineList
datasetTop5 = datasetTop5.sort_values(['Input_Quest', 'CosineSimilarity'], ascending=[True, False])
datasetTop5.to_excel('C:\Girish\GDSAutomationCentral\EYGDSAC_AnalyticsInitiatives\GDSAnalytics\Infosec_Questionnaire\Model2\Model2B_Intermediate_Output.xlsx')


# In[14]:


df_answered = datasetTop5.sort_values('CosineSimilarity',ascending = False).groupby('Input_Quest').head(1)
df_answered.drop('Questions', axis=1, inplace=True)
df_answered.drop('CosineSimilarity', axis=1, inplace=True)
df_answered.drop('Scores', axis=1, inplace=True)
df_answered.to_excel('C:\Girish\GDSAutomationCentral\EYGDSAC_AnalyticsInitiatives\GDSAnalytics\Infosec_Questionnaire\Model2\Model2B_Answers.xlsx',index = False)


# In[12]:


datasetTop5

