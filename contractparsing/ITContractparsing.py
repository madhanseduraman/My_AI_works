# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 12:21:23 2019

@author: Archana.Muraly
"""

import io
import re
#import PyPDF2
import os
import pandas as pd
import numpy as np
import spacy
import nltk 
import fitz
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LAParams
from sklearn.externals import joblib
from modelpreprocessor import *
from dataprotectionpreprocessor import *
from deliverablespreprocessor import *
from contractrequirementpreprocessor import *


def extractTextfromPDF(filename):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    #converter = TextConverter(resource_manager, fake_file_handle) 
    converter = TextConverter(resource_manager, fake_file_handle,codec='utf-8', laparams=LAParams(line_margin=0.2))
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
 
    counter = 0
    with open(filename, 'rb') as fh:
        for page in PDFPage.get_pages(fh, 
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
            counter += 1
 
        text = fake_file_handle.getvalue()

    # close open handles
    converter.close()
    fake_file_handle.close()
 
    if text:
        return text,counter


# In[4]:


def extract_text_by_page(filename):
    with open(filename, 'rb') as fh:
        for page in PDFPage.get_pages(fh, 
                                      caching=True,
                                      check_extractable=True):
            resource_manager = PDFResourceManager()
            fake_file_handle = io.StringIO()
            converter = TextConverter(resource_manager, fake_file_handle)
            page_interpreter = PDFPageInterpreter(resource_manager, converter)
            page_interpreter.process_page(page)
 
            text = fake_file_handle.getvalue()
            yield text
 
            # close open handles
            converter.close()
            fake_file_handle.close()


# In[5]:


def writetoText (filename):
    TextExtract,pageCount = extractTextfromPDF(filename)
    txtfile = filename.replace(r'.pdf','.txt')
    f = open(txtfile, 'w',encoding='utf-8')
    f.write(TextExtract)
    f.close()
    return  txtfile,pageCount

#TextFileName,pageCount = writetoText(pdfFileName)


# In[7]:


def readfromText(filename):
    f = open(filename,"r",encoding='utf-8')
    text = f.read()
    f.close()
    return text


# In[8]:


def ORGentities(TextFileName):

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(readfromText(TextFileName))
        sentences = []
        entities = pd.DataFrame()

        for sentence in doc.sents:
            sentences.append(sentence)

        sentences = list(dict.fromkeys(sentences))
        

        for i in range(0,len(sentences)):
            doc = nlp(str(sentences[i]))         
            for ent in doc.ents:
                entities = entities.append({'Entity_Text': ent.text, 'Entity_Label': ent.label_ }, ignore_index=True)

        EntityCounts = entities['Entity_Text'].groupby(entities['Entity_Label']).value_counts().reset_index(name='Entity_Count')
        EntityCounts = EntityCounts.nlargest(20,"Entity_Count")
        ORG_Entity = EntityCounts.loc[EntityCounts['Entity_Label'] == 'ORG']
        
        return ORG_Entity['Entity_Text'].tolist()





def extract_pageText(pdfFileName,pagenumber):
   counter = 0
   for page in extract_text_by_page(pdfFileName):
       counter += 1
       if counter == pagenumber:
           pageText = str(page)
           break
   return pageText    




def extract_pageFonts(pdfFileName):
    doc = fitz.open(pdfFileName)
    Fontlist = []
    for z in range(0,len(doc)):
    
        fontlist = doc.getPageFontList(z)
        page = doc[z]
        d = page.getText("dict")  
        #print(d["blocks"])
        for i in range(0,len(d["blocks"])):
            if d["blocks"][i]["type"] == 0:   
                for j in range (0,len(d["blocks"][i]["lines"])):
                    for k in range(0,len(d["blocks"][i]["lines"][j]["spans"])):
                        fontdict = d["blocks"][i]["lines"][j]["spans"][k]
                        fontdict["block"] = i
                        fontdict["page"] = z
                        Fontlist.append(fontdict)

    PageFontAnalysis = pd.DataFrame(Fontlist)  
    PageFontAnalysis['size'] = round(PageFontAnalysis['size'])

    return PageFontAnalysis


def CommonRows (pgNum,pdfFileName):
        PageFontAnalysis = extract_pageFonts(pdfFileName)
        Top10pgData  = PageFontAnalysis.query('page=='+str(pgNum)).reset_index()
        Top10pg1Data = PageFontAnalysis.query('page=='+str(pgNum+1)).reset_index()
        Top10pg2Data = PageFontAnalysis.query('page=='+str(pgNum+2)).reset_index()
        
        CommonRows = np.intersect1d(Top10pgData.text, np.intersect1d(Top10pg1Data.text, Top10pg2Data.text))
        i=0
        for i in range(0,len(CommonRows)):
                if CommonRows[i] == ' ':
                    index = i
                    break
        x = np.delete(CommonRows, i, axis=0)
        return x.tolist()



def extractCleanedLines (pdfFileName):
    df = extract_pageFonts(pdfFileName)
    df = df.reset_index()
    return df


def _brkupBlocks (_blockList):
    BlockList = []
    for i in range(0,len(_blockList)-1):
        block = _blockList[i]
        if (len(block) > 300):
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(block)
            for sentence in doc.sents:
                BlockList.append(str(sentence))
        else:
            BlockList.append(block)
    return BlockList




def extractFontBlocks(pagenum):
    Cleaned_Lines = extractCleanedLines(pdfFileName)
    Cleaned_Lines = Cleaned_Lines.query('page=='+str(pagenum))
    Cleaned_Lines.reset_index(drop=True)
    del Cleaned_Lines['index']
    TextList = []
    df = Cleaned_Lines
    prevFont = ""
    prevText = ""
    for i in range(0,len(df)):
        currFont = df.iloc[i]['font']
        currText = df.iloc[i]['text']  
        if(currFont == prevFont):     
            prevText =  prevText + currText 
            if(i==(len(df)-1)):                
                TextList.append(prevText)
        else:
            TextList.append(prevText)
            prevText = currText
        prevFont = currFont 
            
    TextList = list(filter(None, TextList))
    while ' ' in TextList:
        TextList.remove(' ')
    TextList = _brkupBlocks(TextList)
    return TextList




def createBlockdf() :
    Blockdf = pd.DataFrame()
    for i in range(0,pageCount-1):
        blocklist = extractFontBlocks(i)
        tempdf = pd.DataFrame(data = blocklist , columns=['Blocks'])
        tempdf['PageNum'] = i+1
        Blockdf = Blockdf.append(tempdf)
        Blockdf.reset_index(drop = True)
    return Blockdf   



def extractEntityBlock() :
    df = createBlockdf()
    df = df[~df['Blocks'].isin(_orgentities)]
    df = df.reset_index()
    return df





def checkEntity(entitiesList):
    flag = False
    for entity in _orgentities:
        if entity in entitiesList:
            flag = True
    return flag




# In[20]:

def CheckObligation (block):
    _orgEntities = _orgentities
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(block)
    Has_Obligation = 0
    _entity =  ""
    pos_Exception = ["VBN","VBD","JJR"]
    for chunk in doc.noun_chunks:
        
         if(str(chunk.root.dep_) == "nsubj" and str(chunk.root.head.dep_) == "ROOT"):
             matching = [s for s in _orgEntities if chunk.text.lower() in s.lower()]
             if (len(matching) == 0 ): 
                
                matching = [s for s in _orgEntities if chunk.root.text.lower() in s.lower()]

             if (len(matching) > 0 ):            
                text = nltk.word_tokenize(block)
                data = nltk.pos_tag(text)
                pos_df =pd.DataFrame(data, columns=['Text', 'POS'])
                var = chunk.root.head.text
                pos_df =pos_df.query('Text == @var')
                POS_list = pos_df['POS'].tolist()
                
                if(any(i not in pos_Exception for i in POS_list)):

                    _hasModal = True
                else:
                    _hasModal = False
                
                if(_hasModal):
                           Has_Obligation = 1 
                           _entity = matching 
                            
 
         if(str(chunk.root.dep_) == "conj" and str(chunk.root.head.dep_) == "nsubj" and Has_Obligation > 0):
            matching = [s for s in _orgEntities if chunk.text in s]    
            if (len(matching) > 0 ): 
                return "Joint"
                break

    if len(_entity) == 0:
        return "TRUE"
        
    for keys in _entity:

        list1=["EY","EYGS","EYG SERVICE","EYGSERVICES","EYG SERVICES", \
               "EYGSERVICE","CUSTOMER","LICENSEE", \
               "EY GLOBAL SERVICES LIMITED","EY NETWORK"]
        list2=["SUPPLIER","VENDOR","LICENSOR"]

        if keys.upper() in list1:
            
            oblig_entity="EY"
            return oblig_entity
        elif keys.upper() in list2:
            oblig_entity="Supplier"
            return oblig_entity
        else:
            
            if(Has_Obligation > 0):   
                oblig_entity=_entity[0]
                
                return oblig_entity
            else:
                return "TRUE"
        return "TRUE"




def CheckKeyDates (block):
    hasKeyDates = False
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(block)
    for ent in doc.ents:
        if(ent.label_ == "DATE"):
            hasKeyDates = True
            break
    return hasKeyDates


    


# In[55]:


def CheckGoals (block):
    goalKeyword = ["goal","targets","target","performs","policies","measure","objective","optimize","goals"]
    for keys in goalKeyword:
        list1=re.findall(r"\b%s\b" % keys.lower(), block.lower())
        if len(list1)>0:
            hasKeyword = True
        else:
            hasKeyword = False
    return hasKeyword

def CheckScope (block):
    goalKeyword = ["scope","scopes","service","services", "work performed","party’s","workmanship","sublicensed","scalability","integrity","milestone","agreement","confidentiality","principle","clause","architectural","request/documentation","warranty","authorised","listening/monitoring"]
    for keys in goalKeyword:
#        print(block.lower())
        list1=re.findall(r"\b%s\b" % keys.lower(), block.lower())
        if len(list1)>0:
            hasKeyword = True
        else:
            hasKeyword = False
    return hasKeyword
def CheckKPIsSLA (block):
    goalKeyword = ["KPI","KPI’s","KPIs","SLA","metrics","Response Time","service level","performancemetrics"]
    for keys in goalKeyword:
        list1=re.findall(r"\b%s\b" % keys.lower(), block.lower())
        if len(list1)>0:
            hasKeyword = True
            return hasKeyword
        else:
            hasKeyword = False
    return hasKeyword

def CheckServiceCredits (block):
    goalKeyword = ["service credits","claims", "discount","offer"]
    for keys in goalKeyword:
        list1=re.findall(r"\b%s\b" % keys.lower(), block.lower())
        if len(list1)>0:
            hasKeyword = True
        else:
            hasKeyword = False
    return hasKeyword

    

def contractmodel(df1):
    folder=samplepath+'/model/contractrequirement.joblib'

    mod = joblib.load(folder)
    text = df1["Blocks"].tolist()
    

    predictedlist=[]
    for i in range(0,len(text)):
#        print(text[i])
        if len(text[i]) >=25:
            predictedvalues = mod.predict([clean_text_contract(text[i])])
            predictedlist.append(predictedvalues[0])
        else:
            predictedlist.append("FALSE")
    for n, i in enumerate(predictedlist):
        if i == 1:
            predictedlist[n] = True
        else:
            predictedlist[n] = False
    return predictedlist
def deliverablesmodel(df1):
    folder=samplepath+'/model/deliverables.joblib'

    mod = joblib.load(folder)

    text = df1["Blocks"].tolist()
    
    predictedlist=[]
    for i in range(0,len(text)):
        if len(text[i]) >=25:
            predictedvalues = mod.predict([clean_text_deliverables(text[i])])
            predictedlist.append(predictedvalues[0])
        else:
            predictedlist.append("FALSE")
    for n, i in enumerate(predictedlist):
        if i == 1:
            predictedlist[n] = True
        else:
            predictedlist[n] = False
    return predictedlist
def dataprotectionmodel(df1):
    folder=samplepath+'/model/dataprotection.joblib'

    mod = joblib.load(folder)
    text = df1["Blocks"].tolist()
    
    predictedlist=[]
    for i in range(0,len(text)):
        if len(text[i]) >=25:
            predictedvalues = mod.predict([clean_text_data(text[i])])
            predictedlist.append(predictedvalues[0])
        else:
            predictedlist.append("FALSE")
    for n, i in enumerate(predictedlist):
        if str(i) == str(1):
            predictedlist[n] = True
        else:
            predictedlist[n] = False
    
    return predictedlist


def checkEntity1(entitiesList):
    flag = False
    for entity in _orgentities:
        if entity in entitiesList:
            entitiesList=entitiesList+" "+"True"

            flag = True
            return entitiesList
    if flag== False:
        entitiesList=entitiesList+" "+"False"
    return entitiesList

def obligationmodel(df1):
    folder=samplepath+'/model/VotingClassifier.joblib'

    mod = joblib.load(folder)

    text = df1["Blocks"].tolist()
    
    predictedlist=[]
    for i in range(0,len(text)):
        if len(text[i]) >=25:
            
            featured_text=checkEntity1(text[i])
            predictedvalues = mod.predict([clean_text(featured_text)])

            if predictedvalues[0] == 1:
                output=CheckObligation(text[i])
                predictedlist.append(output)
            else:
                predictedlist.append("FALSE")
        else:
            value=checkEntity(text[i])
            if value==True:
                featured_text=checkEntity1(text[i])
                predictedvalues = mod.predict([clean_text(featured_text)])
                if predictedvalues[0] == 1:
                    output=CheckObligation(text[i])
                    predictedlist.append(output)
                else:
                    predictedlist.append("FALSE")
            else:
            
                predictedlist.append("FALSE")
                

    return predictedlist
    
def finaldf(pdfFilename,path):
    global pdfFileName
    global TextFileName
    global pageCount
    global _orgentities
    
    _orgentities=["Affiliates","ALPHAPIPE","AmendmentApp","AMS Vendor","Application","Audit Analytics® Site Audit Analytics", \
    "AuditReport","AUP","Authorized Users","BDO","BlackBerry","Bupa Arabia","BT","CIPHERCLOUD","Client", \
    "Company","Customer","DIB","Enrolled Affiliate","Equinix","Ernst & Young","Exit Assistance the Terms and Conditions","EY" , \
    "EY Global Services Limited","EY Network","EYG" ,"EYG Services","EYGS","IBM","HP","Hexaware","HRCMM","IIA","Imperva","IPRO", \
    "IVES Group","Licensee","Licensor","London Stock Exchange","Microsoft","MIT","Neo Technology","Oracle" ,"OST","parties","Partner","Party", \
    "PRO IPRO","Rackspace","Tata Consultancy Services","SAP", "Service Auditor","Service Level Default","Subscriber","Supplier","Systems","Tailored Reports","TCS", \
    "TECH","Telstra Corporation Limited","the Data Centre","the IBX Centers","Service Organization","the Service Start Date", \
    "Tribridge" ,"Vanco","Vendor","Virtustream","Vodafone","Winshuttle","Waldo",]
    global samplepath
    samplepath=path
    
    pdfFileName=pdfFilename
    TextFileName,pageCount = writetoText(pdfFileName)
    df1 = createBlockdf()
    df1=df1.dropna()
    df1['Goal_Keyword'] = df1.Blocks.apply(CheckGoals)
    df1['Scope'] = df1.Blocks.apply(CheckScope)
    df1['KPIs/SLA'] = df1.Blocks.apply(CheckKPIsSLA)
    df1['Service Credits'] = df1.Blocks.apply(CheckServiceCredits)

    df1['contract requirement']=contractmodel(df1)
    df1['Data protection']=dataprotectionmodel(df1)
    df1['Deliverables']=deliverablesmodel(df1)
    df1['Obligation']=obligationmodel(df1)
    df1.to_excel(path+"/output/Output.xlsx")

    

if __name__ == '__main__':
    
    path=os.getcwd()
    textFiles=[]
    try:
        for root, dirs, files in os.walk(path+"/input"):
            for file in files:
                if file.endswith(".pdf"):
                    textFiles.append(os.path.join(root, file))
                elif file.endswith(".PDF"):
                    textFiles.append(os.path.join(root, file))
     
        
        finaldf(textFiles[0],path)
    except Exception as errors:
        text = str(errors)
        textFilename = path + "/output/"+"error.txt"

        textFile = open(textFilename, "w",encoding="utf-8") #make text file
        textFile.write(text)
        
	
   
    