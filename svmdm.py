import DataProces
import scipy
from gensim import matutils
from numpy import *
from gensim import corpora
from sklearn import svm

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

def svm_classify(train_set,train_tag,test_set,test_tag):

    clf = svm.LinearSVC()
    clf_res = clf.fit(train_set,train_tag)
    train_pred  = clf_res.predict(train_set)
    test_pred = clf_res.predict(test_set)

    train_err_num, train_err_ratio = checkPred(train_tag, train_pred)
    test_err_num, test_err_ratio  = checkPred(test_tag, test_pred)

    print('=== 分类训练完毕，分类结果如下 ===')
    print('训练集误差: {e}'.format(e=train_err_ratio))
    print('检验集误差: {e}'.format(e=test_err_ratio))

    return clf_res

def checkPred(data_tag, data_pred):
    if data_tag.__len__() != data_pred.__len__():
        raise RuntimeError('The length of data tag and data pred should be the same')
    err_count = 0
    for i in range(data_tag.__len__()):
        if data_tag[i]!=data_pred[i]:
            err_count += 1
    err_ratio = err_count / data_tag.__len__()
    return [err_count, err_ratio]


if __name__ == '__main__':
    # dictionary = DataProces.chi_process()
    dictionary = corpora.Dictionary.load('file/chidict.dict')
    number_of_feature = len(dictionary)
    clsList = {0:'yl', 1:'auto', 2:'cj', 3:'game',4:'xingzuo', 5:'it', 6:'mil', 7:'health', 8:'ty',9:'edu'}
    word_mtrix_list = {}
    result = {}
    #各个分类下，错分的单词去向
    train_set = []
    train_tag = []
    test_set = []
    test_tag = []
    for k,cls in clsList.items():
         temp = make_cls_mtrix(cls)
         train_set = train_set + temp
         for i in range(0,len(temp)):
             train_tag.append(k)
    train_matrix = matutils.corpus2csc(train_set,num_terms=number_of_feature).T
    for k,cls in clsList.items():
         temp =  make_test_mtx(cls)
         test_set = test_set + temp
         for i in range(0,len(temp)):
             test_tag.append(k)
    test_matrix = matutils.corpus2csc(test_set,num_terms=number_of_feature).T
    print('test')
    predictor = svm_classify(train_matrix, array(train_tag,ndmin=2).T, test_matrix, array(test_tag,ndmin=2).T)
    # x = open(path_tmp_predictor, 'wb')
    # pkl.dump(predictor, x)
    # x.close()