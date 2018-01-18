import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas
import math
import json
import matplotlib.pyplot as plt 
import numpy as np
from collections import Counter
import gensim, logging
# from nltk.corpsentiment_featuresport sentiwordnet as swn
import pandas as pd
import pdb

def get_description_length(desc):
    tokens = nltk.word_tokenize(desc)
    return len(tokens)

def get_description_sentiment():
    df = pd.read_csv("liwc.txt", sep='\t')
    df_sentiment_features = df.drop(['Segment'], axis=1)
    df_sentiment_features.to_csv('sentiment_features.csv')

def word_doc_similarity(text1, text2):
    tfidf = TfidfVectorizer().fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]

def aggregate_lender_motivations():
    all_motivations = ""
    with open('motivations', 'r', encoding='latin-1') as fm:
        for line in fm:
            items = line.split('\t')
            lender_motivation = items[1]
            all_motivations += lender_motivation
    return all_motivations

def getDicMo():
    try:
        with open('class_motivations_dic.json', r) as f:
            dic_final = json.load(f)
        return dic_final
    except:
        pass

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
    
    f = open('class_motivations_dic.json', 'w')
    json.dump(dicfinal, f)
    f.close()

    return dicfinal


def getWordCounts(desc):
    stops = set(stopwords.words("english"))
    tokens = word_tokenize(desc.lower())
    filtered_words = [word for word in tokens if word not in stops]
    counts = Counter(filtered_words)
    return counts


def getSimilarity(count_dic1, count_dic2):
    tf = 0
    for key in count_dic1:
        if key in count_dic2:
            tf += count_dic1[key] * count_dic2[key]
    s1 = sum([x*x for x in count_dic1.values()])
    s2 = sum([x*x for x in count_dic2.values()])
    try:
        sim = tf / math.sqrt(s1*s2)
    except:
        sim = 0
    return sim


def main():
    fo = open('features.csv', 'w', encoding='latin-1', newline='')
    csv_writer = csv.writer(fo, delimiter=',')

    d = getDicMo()
    sorted_keys = sorted(d.keys())
    print("Sorted Keys: ", sorted_keys)

    with open('features_test.csv', 'r', encoding='latin-1') as f:
        csv_reader = csv.reader(f)

        count = 0
        for row in csv_reader:
            similarity_motivation = []
            count += 1
            if count%10000 == 0:
                print("Count: ", count)
            if count == 1:
                headings = row
                desc_index = headings.index('description')
                use_index = headings.index('use')
                for key in sorted_keys:
                    headings.append('sim_desc_'+key)
                    headings.append('sim_use_'+key)
                csv_writer.writerow(headings)
            else:
                desc = row[desc_index]
                use = row[use_index]
                desc_word_counts = getWordCounts(desc)
                use_word_counts = getWordCounts(use)
                for key in sorted_keys:
                    sim_desc = getSimilarity(desc_word_counts, d[key])
                    row.append(sim_desc)
                    sim_use = getSimilarity(use_word_counts, d[key])
                    row.append(sim_use)
                csv_writer.writerow(row)
    fo.close()

def add_feature_values(filename):
    fo = open('features_test.csv', 'w', encoding='latin-1', newline='')
    csv_writer = csv.writer(fo, delimiter=',')

    items = [[None for x in range(94)] for y in range(993981)]
    feature_headings = []
    item_count = 0
    f2 = open(filename)
    csv_reader2 = csv.reader(f2)
    for row in csv_reader2:
        item_count += 1
        if item_count == 1:
            feature_headings = row[1:]
        else:
            i = int(row[0].replace('.txt',''))
            items[i] = row[1:]
    f2.close()

    with open('features.csv', 'r', encoding='latin-1') as f:
        csv_reader = csv.reader(f)
        count = 0
        for row in csv_reader:
            count += 1
            if count%10000 == 0:
                print("Count: ", count)
            if count == 1:
                headings = row
                desc_index = headings.index('description')
                headings.pop(desc_index)
                for heading in feature_headings:
                    headings.append(heading)
                csv_writer.writerow(headings)
            else:
                loan_id = int(row[0])
                # pdb.set_trace()
                row.pop(desc_index)
                for item in items[loan_id]:
                    row.append(item)
                csv_writer.writerow(row)
    fo.close()

# main()
# get_description_sentiment()
add_feature_values('sentiment_features.csv')
