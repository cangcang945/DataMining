import DataProces
import scipy
from gensim import matutils
from numpy import *
from gensim import corpora

def make_cls_mtrix(cls):
    files = DataProces.get_cls_Files(cls)
    word_mat = [dictionary.doc2bow(file) for file in files]
    return word_mat

def getTestFiles(cls):
    fstop = open('test_seg_3/' + cls, 'r', encoding='utf-8', errors='ignore')
    words = fstop.read()
    fstop.close()
    word_lists = words.split('\n')
    files = []
    del word_lists[-1]
    for word_list in word_lists:
        word = word_list.split()
        files.append(word)
    return files

def make_test_mtx(cls):
    files = getTestFiles(cls)
    word_mat = [dictionary.doc2bow(file) for file in files]
    return word_mat

def trainNB(matrix):
    numTrainDocs = matrix.shape[1]
    numWords = matrix.shape[0]
    p_cls_wordnum = sum(matrix,axis=1)
    total_word_num = dictionary.num_pos
    cls_total = total_word_num + sum(p_cls_wordnum)
    p_cls_wordnum = ones((numWords,1)) + sum(matrix,axis=1)
    p_cls_vec = log(divide(p_cls_wordnum,cls_total))
    return p_cls_vec

def classifyNB(word_vec,cls_vec):
    s = sum(word_vec,axis=1)
    p = s.T*cls_vec + log(1/10)
    return p

if __name__ == '__main__':
    # dictionary = DataProces.chi_process()
    dictionary = corpora.Dictionary.load('file/chidict.dict')
    number_of_feature = len(dictionary)
    clsList = ['yl', 'auto', 'cj', 'game', 'xingzuo', 'it', 'mil', 'health', 'ty','edu']
    word_mtrix_list = {}
    result = {}
    #各个分类下，错分的单词去向
    cls_words_dict = {}
    for cls in clsList:
        cls_words_dict[cls] = {}
        for cls2 in clsList:
            cls_words_dict[cls][cls2] = 0
    for cls in clsList:
         word_mtrix = make_cls_mtrix(cls)
         csc_matrix = matutils.corpus2csc(word_mtrix,num_terms=number_of_feature)
         cls_vec = trainNB(csc_matrix)
         word_mtrix_list[cls] = cls_vec
    total = 0
    total_docnum = 0
    for cls in clsList:
        word_mat = matutils.corpus2csc(make_test_mtx(cls),num_terms=number_of_feature)
        p = -float('Inf')
        clas = ''
        num = 0
        for i in range(0,word_mat.shape[1]):
            word_vc = word_mat[:, i]
            for k,v in word_mtrix_list.items():
              temp = classifyNB(word_vc,v)
              if(temp[0][0] > p):
                    p = temp[0][0]
                    clas = k
            cls_words_dict[cls][clas] = cls_words_dict[cls][clas]+1
            if(clas == cls):
                num = num+1
            p = -float('Inf')
        result[cls] = num/word_mat.shape[1]
        print(result[cls])
        total = total + num
        total_docnum = total_docnum + word_mat.shape[1]
    print(str(total/total_docnum))
    print(cls_words_dict)
        # for cls in clsList:
        #     DataProces.make_vector(files,cls)
        # scipy_csc_matrix = matutils.corpus2csc(DataProces.word_vector_list)