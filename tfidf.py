# coding:utf-8
__author__ = "liuxuejiang"
import os
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

PATH = "tmp/"

def savefile(savepath, content):
    fp = open(savepath, "a",encoding='utf-8', errors='ignore')
    fp.write(content + '\n')
    fp.close()

if __name__ == "__main__":
    corpus = []
    #ctgys = ['cj', 'mil', 'auto', 'gj']
    ctgys = ['ylbak','auto','cjbak','game','xingzuo','it','mil','health','ty','edu']
    #ctgys = ['cj']
    for ctgy in ctgys:
        fstop = open('train_seg_3/' + ctgy, 'r', encoding='utf-8', errors='ignore')
        words = fstop.read()
        fstop.close()
        word_lists = words.split('\n')
        del word_lists[-1]
        content = " ".join(word_lists)
        corpus.append(content)
    #i = 0
    #con = []
    #for cor in corpus:
    #    num = cor.split(' ')
    #    for nu in num:
    #        if nu not in con:
    #            con.append(nu)
    #            i += 1
    ##fp = open('number', 'r', encoding='gb2312', errors='ignore')
    #fp.write(i)
    #fp.close()
    #print(i)
    vectorizer=CountVectorizer()
    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))
    word=vectorizer.get_feature_names()
    weight=tfidf.toarray()
    #print(len(word))
    #print (len(weight))
    staty = {}
    for i in range(len(weight)):
        staty[i] = {}
        for j in range(len(word)):
            #print(len(word))
            if weight[i][j] >= 0.0001:
                content = word[j]
                scale = weight[i][j]
            #print(content)
                staty[i][content] = scale
        features = sorted(staty[i].items(), key=lambda d: d[1])[-30:]
        print(features)
        #total = ''
        #for feature in features:
        #    total += feature[0]
        #    total += ' '
        #fp = open('temp/' + str(i), 'w', encoding='utf-8', errors='ignore')
        #fp.write(total)
        #fp.close()

