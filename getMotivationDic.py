#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 

@author: tong
"""
#remove stop words
#change all letters into lower case

import pandas 
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import csv
import math
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import pdb

from nltk.corpus import stopwords


def getDicMo():
    infile_rating=pandas.read_csv("final_ratings", header=None,names=['id',"classMotivation1" ,"classMotivation2" , "classMotivation3" , "classMotivation4",  "classMotivation5",
    "classMotivation6", "classMotivation7" ,"classMotivation8" ,"classMotivation9" ,"classMotivation10"])
        
    infile_moti=pandas.read_csv("motivations", sep='\t', encoding = "latin-1", header=None,names=['id',"motivation"])

    idIndex=[]
    lsAll=[]
    stops = set(stopwords.words("english"))

    #print infile_moti.iat[0,1]
    for n in range(infile_moti.shape[0]):
        #print (infile_moti.at[n,"motivation"])
        ls=word_tokenize(str(infile_moti.iat[n,1]).lower())
        filtered_words = [word for word in ls if word not in stops]
        lsAll.append(filtered_words)
        
    infile_moti['tokenMotiv']=lsAll
    
    dic={}
    classMotivation1=[]

    df = infile_rating.set_index('id').join(infile_moti.set_index('id'))
    for n in range(10):
        for index, row in df.iterrows():
            if row["classMotivation{}".format(n+1)] ==1:
                if "classMotivation{}".format(n+1) not in dic:
                    dic["classMotivation{}".format(n+1)] = row['tokenMotiv']
                else:
                    try:
                        dic["classMotivation{}".format(n+1)]+= row['tokenMotiv']
                    except:
                        pass
                
    dicfinal={}
    for moticlass in dic.keys():
        dicfinal[moticlass]={}
        als= dic[moticlass]
        for token in set(als):
            dicfinal[moticlass][token]=als.count(token)
                
    return dicfinal, len(lsAll)

def getWordCounts(desc):
    stops = set(stopwords.words("english"))
    tokens = word_tokenize(desc.lower())
    filtered_words = [word for word in tokens if word not in stops]
    counts = Counter(filtered_words)
    return counts

def getSimilarity(d1, d2):
    tf = 0
    for key in d1:
        if key in d2:
            tf += d1[key] * d2[key]
    s1 = sum([x*x for x in d1.values()])
    s2 = sum([x*x for x in d2.values()])
    try:
        sim = tf / math.sqrt(s1*s2)
    except:
        sim = 0
    return sim

if __name__ == '__main__':
    d, vocab_size = getDicMo()
    s = set([])
    for key,item in d.items():
        s = s.union(set(item.keys()))
    pdb.set_trace()
    d1 = {'hello': 2, 'world': 2, 'hi': 1}
    d2 = {'hello': 1, 'friend': 1}
    d3 = {'hello': 1, 'friend': 1, 'how': 1, 'are': 1, 'you': 1, 'doing': 1}
    print(getSimilarity(d1,d2))
    print(getSimilarity(d1,d3))
    assert round(getSimilarity(d1,d2), 2) == 0.47
    assert round(getSimilarity(d1,d3), 2) == 0.27
    pdb.set_trace()
    model = gensim.models.Word2Vec()
    with open('loans_test.csv', 'r', encoding='latin-1') as f:
        csv_reader = csv.reader(f)
        vectorizer = CountVectorizer(lowercase=True, stop_words='english')
        count = 0
        for row in csv_reader:
            count += 1
            if count%10000 == 0:
                print("Count: ", count)
            if count == 1:
                headings = row
                desc_index = headings.index('description')
            else:
                desc = row[desc_index]
                if len(desc) > 0:
                    word_counts = getWordCounts(desc)
                    for key,value in d.items():
                        sim = getSimilarity(word_counts, value)
                        print("Similarity: ", sim)
