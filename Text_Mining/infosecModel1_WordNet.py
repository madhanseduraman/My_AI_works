
# coding: utf-8

# In[24]:


# Model 1 -- For testing Semantic Similiarity with WordNet


# In[25]:


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
import re

# Parameters to the algorithm. Currently set to values that was reported
# in the paper to produce "best" results.
#ALPHA = 0.2
#BETA = 0.45
#ETA = 0.4
#PHI = 0.2
#DELTA = 0.85
ALPHA = 0.2
BETA = 0.7
ETA = 0.4
PHI = 0.2
DELTA = 0.85

brown_freqs = dict()
N = 0


# In[26]:


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


# In[27]:


def get_best_synset_pair(word_1, word_2):
    """ 
    Choose the pair with highest path similarity among all pairs. 
    Mimics pattern-seeking behavior of humans.
    """
    max_sim = -1.0
    synsets_1 = wn.synsets(word_1)
    synsets_2 = wn.synsets(word_2)
    if len(synsets_1) == 0 or len(synsets_2) == 0:
        return None, None
    else:
        max_sim = -1.0
        best_pair = None, None
        for synset_1 in synsets_1:
            for synset_2 in synsets_2:
               sim = wn.path_similarity(synset_1, synset_2)
               if sim is None:
                   sim = 0.0 
               if sim > max_sim:
                   max_sim = sim
                   best_pair = synset_1, synset_2
        return best_pair

def length_dist(synset_1, synset_2):
    """
    Return a measure of the length of the shortest path in the semantic 
    ontology (Wordnet in our case as well as the paper's) between two 
    synsets.
    """
    l_dist = sys.maxsize
    if synset_1 is None or synset_2 is None: 
        return 0.0
    if synset_1 == synset_2:
        # if synset_1 and synset_2 are the same synset return 0
        l_dist = 0.0
    else:
        wset_1 = set([str(x.name()) for x in synset_1.lemmas()])        
        wset_2 = set([str(x.name()) for x in synset_2.lemmas()])
        if len(wset_1.intersection(wset_2)) > 0:
            # if synset_1 != synset_2 but there is word overlap, return 1.0
            l_dist = 1.0
        else:
            # just compute the shortest path between the two
            l_dist = synset_1.shortest_path_distance(synset_2)
            if l_dist is None:
                l_dist = 0.0
    # normalize path length to the range [0,1]
    return math.exp(-ALPHA * l_dist)

def hierarchy_dist(synset_1, synset_2):
    """
    Return a measure of depth in the ontology to model the fact that 
    nodes closer to the root are broader and have less semantic similarity
    than nodes further away from the root.
    """
    h_dist = sys.maxsize
    if synset_1 is None or synset_2 is None: 
        return h_dist
    if synset_1 == synset_2:
        # return the depth of one of synset_1 or synset_2
        h_dist = max([x[1] for x in synset_1.hypernym_distances()])
    else:
        # find the max depth of least common subsumer
        hypernyms_1 = {x[0]:x[1] for x in synset_1.hypernym_distances()}
        hypernyms_2 = {x[0]:x[1] for x in synset_2.hypernym_distances()}
        lcs_candidates = set(hypernyms_1.keys()).intersection(
            set(hypernyms_2.keys()))
        if len(lcs_candidates) > 0:
            lcs_dists = []
            for lcs_candidate in lcs_candidates:
                lcs_d1 = 0
                if lcs_candidate in hypernyms_1:
                    lcs_d1 = hypernyms_1[lcs_candidate]
                lcs_d2 = 0
                if lcs_candidate in hypernyms_2:
                    lcs_d2 = hypernyms_2[lcs_candidate]
                lcs_dists.append(max([lcs_d1, lcs_d2]))
            h_dist = max(lcs_dists)
        else:
            h_dist = 0
    return ((math.exp(BETA * h_dist) - math.exp(-BETA * h_dist)) / 
        (math.exp(BETA * h_dist) + math.exp(-BETA * h_dist)))
    
def word_similarity(word_1, word_2):
    synset_pair = get_best_synset_pair(word_1, word_2)
    return (length_dist(synset_pair[0], synset_pair[1]) * 
        hierarchy_dist(synset_pair[0], synset_pair[1]))


# In[28]:


######################### sentence similarity ##########################

def most_similar_word(word, word_set):
    """
    Find the word in the joint word set that is most similar to the word
    passed in. We use the algorithm above to compute word similarity between
    the word and each word in the joint word set, and return the most similar
    word and the actual similarity value.
    """
    max_sim = -1.0
    sim_word = ""
    for ref_word in word_set:
      sim = word_similarity(word, ref_word)
      if sim > max_sim:
          max_sim = sim
          sim_word = ref_word
    return sim_word, max_sim
    
def info_content(lookup_word):
    """
    Uses the Brown corpus available in NLTK to calculate a Laplace
    smoothed frequency distribution of words, then uses this information
    to compute the information content of the lookup_word.
    """
    global N
    if N == 0:
        # poor man's lazy evaluation
        for sent in brown.sents():
            for word in sent:
                word = word.lower()
                if word not in brown_freqs:
                    brown_freqs[word] = 0
                brown_freqs[word] = brown_freqs[word] + 1
                N = N + 1
    lookup_word = lookup_word.lower()
    n = 0 if lookup_word not in brown_freqs else brown_freqs[lookup_word]
    return 1.0 - (math.log(n + 1) / math.log(N + 1))
    
def semantic_vector(words, joint_words, info_content_norm):
    """
    Computes the semantic vector of a sentence. The sentence is passed in as
    a collection of words. The size of the semantic vector is the same as the
    size of the joint word set. The elements are 1 if a word in the sentence
    already exists in the joint word set, or the similarity of the word to the
    most similar word in the joint word set if it doesn't. Both values are 
    further normalized by the word's (and similar word's) information content
    if info_content_norm is True.
    """
    sent_set = set(words)
    semvec = np.zeros(len(joint_words))
    i = 0
    for joint_word in joint_words:
        if joint_word in sent_set:
            # if word in union exists in the sentence, s(i) = 1 (unnormalized)
            semvec[i] = 1.0
            if info_content_norm:
                semvec[i] = semvec[i] * math.pow(info_content(joint_word), 2)
        else:
            # find the most similar word in the joint set and set the sim value
            sim_word, max_sim = most_similar_word(joint_word, sent_set)
            semvec[i] = PHI if max_sim > PHI else 0.0
            if info_content_norm:
                semvec[i] = semvec[i] * info_content(joint_word) * info_content(sim_word)
        i = i + 1
    return semvec                
            
def semantic_similarity(sentence_1, sentence_2, info_content_norm):
    """
    Computes the semantic similarity between two sentences as the cosine
    similarity between the semantic vectors computed for each sentence.
    """
    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    
    porter_stemmer = PorterStemmer()
    lm = WordNetLemmatizer()
    
    # Adding for correcting similiarity score by removing stopwords starts
    stop_words = set(stopwords.words('english')) 
    
    words_1 = [lm.lemmatize(porter_stemmer.stem(w)) for w in words_1 if not w in stop_words] 
    words_2 = [lm.lemmatize(porter_stemmer.stem(w)) for w in words_2 if not w in stop_words]
    
    # Adding for correcting similiarity score by removing stopwords ends
    
    joint_words = set(words_1).union(set(words_2))
    vec_1 = semantic_vector(words_1, joint_words, info_content_norm)
    vec_2 = semantic_vector(words_2, joint_words, info_content_norm)
    return np.dot(vec_1, vec_2.T) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))

######################### word order similarity ##########################

def word_order_vector(words, joint_words, windex):
    """
    Computes the word order vector for a sentence. The sentence is passed
    in as a collection of words. The size of the word order vector is the
    same as the size of the joint word set. The elements of the word order
    vector are the position mapping (from the windex dictionary) of the 
    word in the joint set if the word exists in the sentence. If the word
    does not exist in the sentence, then the value of the element is the 
    position of the most similar word in the sentence as long as the similarity
    is above the threshold ETA.
    """
    wovec = np.zeros(len(joint_words))
    i = 0
    wordset = set(words)
    for joint_word in joint_words:
        if joint_word in wordset:
            # word in joint_words found in sentence, just populate the index
            wovec[i] = windex[joint_word]
        else:
            # word not in joint_words, find most similar word and populate
            # word_vector with the thresholded similarity
            sim_word, max_sim = most_similar_word(joint_word, wordset)
            if max_sim > ETA:
                wovec[i] = windex[sim_word]
            else:
                wovec[i] = 0
        i = i + 1
    return wovec

def word_order_similarity(sentence_1, sentence_2):
    """
    Computes the word-order similarity between two sentences as the normalized
    difference of word order between the two sentences.
    """
    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    joint_words = list(set(words_1).union(set(words_2)))
    windex = {x[1]: x[0] for x in enumerate(joint_words)}
    r1 = word_order_vector(words_1, joint_words, windex)
    r2 = word_order_vector(words_2, joint_words, windex)
    return 1.0 - (np.linalg.norm(r1 - r2) / np.linalg.norm(r1 + r2))

######################### overall similarity ##########################

def similarity(sentence_1, sentence_2, info_content_norm):
    """
    Calculate the semantic similarity between two sentences. The last 
    parameter is True or False depending on whether information content
    normalization is desired or not.
    """
    #converting to lower case
    sentence_1 = sentence_1.lower()
    sentence_2 = sentence_2.lower()
    
    return DELTA * semantic_similarity(sentence_1, sentence_2, info_content_norm) +         (1.0 - DELTA) * word_order_similarity(sentence_1, sentence_2)


# In[33]:


QuestionBank = pd.read_excel("C:\Girish\GDSAutomationCentral\EYGDSAC_AnalyticsInitiatives\GDSAnalytics\Infosec_Questionnaire\Colated_Questionnaire.xlsx",sheet_name="Q&A")

inputquestions = pd.read_excel("C:\Girish\GDSAutomationCentral\EYGDSAC_AnalyticsInitiatives\GDSAnalytics\Infosec_Questionnaire\InputQuestions.xlsx",sheet_name="Questions")

#QuestionBank['Combined'] = QuestionBank['Questions'].fillna('') + ' ' + QuestionBank['Response'].fillna('') + ' ' +QuestionBank['Procedure_Process_Used'].fillna('') + ' ' + QuestionBank['Supporting_documentation'].fillna('') + ' ' + QuestionBank['Additional_Comments'].fillna('') +  ' ' + QuestionBank['Topic'].fillna('') 
                            
QuestionBank['Combined'] = QuestionBank['Questions'].fillna('') 
     
colnames = ['Input_Quest','question','answer','procedure_used','additional_comments','semanticsimiliarityscore']

df_top5 = pd.DataFrame()

for q in range(len(inputquestions['Questions'])):
    questionTocheck = inputquestions['Questions'][q]
    Input_Quest=[]
    question=[]
    procedure_used = []
    additional_comments = []
    answer=[]
    similiarityscore = []
#    orderandsimiliarityscore = []
    df = pd.DataFrame(columns = colnames)
    tempdf = pd.DataFrame()
    #print(questionTocheck)
    for i in range(len(QuestionBank['Combined'])) :
        #print(test_df['Combined'][i])
        inputtext = questionTocheck
        Input_Quest.append(inputtext)
        #questiontext.append(test_df['Combined'][i])
        question.append(QuestionBank['Questions'][i])    
        answer.append(QuestionBank['Response'][i])
        procedure_used.append(QuestionBank['Procedure_Process_Used'][i])
        additional_comments.append(QuestionBank['Additional_Comments'][i] ) 
#        simiscore = similarity(inputtext,QuestionBank['Combined'][i],False)
#        score =semantic_similarity(inputtext,QuestionBank['Combined'][i],False)
#        score =semantic_similarity(inputtext,re.sub("\d+", " ", QuestionBank['Combined'][i]),False)
        score = semantic_similarity(cleanData(inputtext),cleanData(QuestionBank['Combined'][i]),False)
        similiarityscore.append(score)
#       orderandsimiliarityscore.append(simiscore)
    df["Input_Quest"] = Input_Quest
    df["question"] = question
    df["answer"] = answer
    df["procedure_used"] = procedure_used
    df["additional_comments"] = additional_comments
    df["semanticsimiliarityscore"] = similiarityscore
#    df["orderandsimiliarityscore"] = orderandsimiliarityscore
    tempdf = df.sort_values('semanticsimiliarityscore',ascending = False).groupby('Input_Quest').head(5)
    df_top5 = df_top5.append(tempdf)


df_top5.to_excel('C:\Girish\GDSAutomationCentral\EYGDSAC_AnalyticsInitiatives\GDSAnalytics\Infosec_Questionnaire\Model1\Model1_Intermediate_Output.xlsx')


# In[29]:


df_answered = df_top5.sort_values('semanticsimiliarityscore',ascending = False).groupby('Input_Quest').head(1)
df_answered.drop('question', axis=1, inplace=True)
df_answered.drop('semanticsimiliarityscore', axis=1, inplace=True)
df_answered.to_excel('C:\Girish\GDSAutomationCentral\EYGDSAC_AnalyticsInitiatives\GDSAnalytics\Infosec_Questionnaire\Model1\Model1_Answers.xlsx',index = False)


# ---

# In[34]:


#sent1 = "Do you perform security testing of applications ?"
#sent2 = "When secure code review is performed, which parts of the applications are reviewed?"
#sent3 = "Who performs the compiled application security tests??"

#sent1 = cleanData(sent1)
#sent2 = cleanData(sent2)
#sent3 = cleanData(sent3)
#sim1 = semantic_similarity(sent1,sent2,False)
#sim2 = semantic_similarity(sent1,sent3,False)

#print(sim1)
#print(sim2)


# ---
