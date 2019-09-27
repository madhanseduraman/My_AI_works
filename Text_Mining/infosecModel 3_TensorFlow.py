
# coding: utf-8

# In[ ]:


########################
# Tensor Flow example  #
########################


# In[12]:


import pandas
import scipy
import math
import sys
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import sentencepiece as spm
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import nltk
import math
import re as re
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


#module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

module_url = "https://tfhub.dev/google/universal-sentence-encoder-lite/2?tf-hub-format=compressed"

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

with tf.Session() as sess:
  spm_path = sess.run(embed(signature="spm_path"))

sp = spm.SentencePieceProcessor()
sp.Load(spm_path)
#print("SentencePiece model loaded at {}.".format(spm_path))


# In[13]:


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


# In[14]:


def process_to_IDs_in_sparse_format(sp, sentences):
  # An utility method that processes sentences with the sentence piece processor
  # 'sp' and returns the results in tf.SparseTensor-similar format:
  # (values, indices, dense_shape)
  ids = [sp.EncodeAsIds(x) for x in sentences]
  max_len = max(len(x) for x in ids)
  dense_shape=(len(ids), max_len)
  values=[item for sublist in ids for item in sublist]
  indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
  return (values, indices, dense_shape)

input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
encodings = embed(
    inputs=dict(
        values=input_placeholder.values,
        indices=input_placeholder.indices,
        dense_shape=input_placeholder.dense_shape))


# In[15]:


# Training with a sample by downloading a dataset
def load_sts_dataset(filename):
  # Loads a subset of the STS dataset into a DataFrame. In particular both
  # sentences and their human rated similarity score.
  sent_pairs = []
  with tf.gfile.GFile(filename, "r") as f:
    for line in f:
      ts = line.strip().split("\t")
      # (sent_1, sent_2, similarity_score)
      sent_pairs.append((ts[5], ts[6], float(ts[4])))
  return pandas.DataFrame(sent_pairs, columns=["sent_1", "sent_2", "sim"])


def download_and_load_sts_data():
  sts_dataset = tf.keras.utils.get_file(
      fname="Stsbenchmark.tar.gz",
      origin="http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz",
      extract=True)

  sts_dev = load_sts_dataset(
      os.path.join(os.path.dirname(sts_dataset), "stsbenchmark", "sts-dev.csv"))
  sts_test = load_sts_dataset(
      os.path.join(
          os.path.dirname(sts_dataset), "stsbenchmark", "sts-test.csv"))

  return sts_dev, sts_test


sts_dev, sts_test = download_and_load_sts_data()


# In[16]:


sts_input1 = tf.sparse_placeholder(tf.int64, shape=(None, None))
sts_input2 = tf.sparse_placeholder(tf.int64, shape=(None, None))


# For evaluation we use exactly normalized rather than
# approximately normalized.
sts_encode1 = tf.nn.l2_normalize(
    embed(
        inputs=dict(values=sts_input1.values,
                    indices=sts_input1.indices,
                    dense_shape=sts_input1.dense_shape)),
    axis=1)
sts_encode2 = tf.nn.l2_normalize(
    embed(
        inputs=dict(values=sts_input2.values,
                    indices=sts_input2.indices,
                    dense_shape=sts_input2.dense_shape)),
    axis=1)

sim_scores = -tf.acos(tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1))


# In[17]:


#dataset = sts_test[11:16] #@param ["sts_dev", "sts_test"] {type:"raw"}

#dataset = pandas.DataFrame()
#dataset['sent_2'] = sts_test['sent_2'][11:16]
#dataset['sent_1'] = "A man is cutting an onion"
#dataset.reset_index(drop = True,inplace = True)
#dataset = dataset[['sent_1','sent_2']]
#dataset


QuestionBank = pd.read_excel("C:\Girish\GDSAutomationCentral\EYGDSAC_AnalyticsInitiatives\GDSAnalytics\Infosec_Questionnaire\Colated_Questionnaire.xlsx",sheet_name="Q&A")

inputquestions = pd.read_excel("C:\Girish\GDSAutomationCentral\EYGDSAC_AnalyticsInitiatives\GDSAnalytics\Infosec_Questionnaire\InputQuestions.xlsx",sheet_name="Questions")

#QuestionBank['Combined'] = QuestionBank['Questions'].fillna('') + ' ' + QuestionBank['Response'].fillna('') + ' ' +QuestionBank['Procedure_Process_Used'].fillna('') + ' ' + QuestionBank['Supporting_documentation'].fillna('') + ' ' + QuestionBank['Additional_Comments'].fillna('') +  ' ' + QuestionBank['Topic'].fillna('') 

QuestionBank['Combined'] = QuestionBank['Questions'].fillna('')

QuestionBank['Combined'] = QuestionBank['Combined'].map(lambda x: cleanData(x))

dataset = pd.DataFrame()

for q in range(len(inputquestions['Questions'])):
    tempdf = pd.DataFrame()
    questionTocheck = inputquestions['Questions'][q]
    tempdf['sent_2'] = QuestionBank['Combined']
    tempdf['sent_1'] = cleanData(inputquestions['Questions'][q])
    tempdf['Input_Quest'] = questionTocheck
    tempdf['Questions'] = QuestionBank['Questions']
    tempdf['Response'] = QuestionBank['Response']
    tempdf['Procedure_Process_Used'] = QuestionBank['Procedure_Process_Used']
    tempdf['Supporting_documentation'] = QuestionBank['Supporting_documentation']
    tempdf['Additional_Comments'] = QuestionBank['Additional_Comments']
    tempdf['Topic'] = QuestionBank['Topic']
    dataset = dataset.append(tempdf)
        


# In[18]:


values1, indices1, dense_shape1 = process_to_IDs_in_sparse_format(sp, dataset['sent_1'].tolist())
values2, indices2, dense_shape2 = process_to_IDs_in_sparse_format(sp, dataset['sent_2'].tolist())
#similarity_scores = dataset['sim'].tolist()


# In[19]:


def run_sts_benchmark(session):
  """Returns the similarity scores"""
  scores = session.run(
      sim_scores,
      feed_dict={
          sts_input1.values: values1,
          sts_input1.indices:  indices1,
          sts_input1.dense_shape:  dense_shape1,
          sts_input2.values:  values2,
          sts_input2.indices:  indices2,
          sts_input2.dense_shape:  dense_shape2,
      })
  return scores


with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  session.run(tf.tables_initializer())
  scores = run_sts_benchmark(session)


# In[23]:


dataset['Scores'] = scores
datasetTop5 = pd.DataFrame()
#datasetTop5 = dataset.sort_values('Scores',ascending = False).groupby('sent_1').head(5)
datasetTop5 = datasetTop5.append(dataset.sort_values('Scores',ascending = False).groupby('Input_Quest').head(5))


# In[24]:


datasetTop5 = datasetTop5[['Input_Quest','Questions','Response','Procedure_Process_Used','Supporting_documentation','Additional_Comments','Topic','Scores']]
datasetTop5 = datasetTop5.sort_values(['Input_Quest', 'Scores'], ascending=[True, False])
datasetTop5.to_excel('C:\Girish\GDSAutomationCentral\EYGDSAC_AnalyticsInitiatives\GDSAnalytics\Infosec_Questionnaire\Model3\Model3_Intermediate_Output.xlsx')


# In[25]:


df_answered = datasetTop5.sort_values('Scores',ascending = False).groupby('Input_Quest').head(1)
df_answered.drop('Questions', axis=1, inplace=True)
df_answered.drop('Scores', axis=1, inplace=True)
df_answered.to_excel('C:\Girish\GDSAutomationCentral\EYGDSAC_AnalyticsInitiatives\GDSAnalytics\Infosec_Questionnaire\Model3\Model3_Answers.xlsx',index = False)


# ---
#     
#             
#     
#     
