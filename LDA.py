from numpy import *
from gensim import corpora,models
from sklearn import svm
#词和词频
word_dictionary = corpora.Dictionary()
#结构为{‘class’:[]}
seperate_word_dict = {}
word_vector_list = {}
clsList = ['yl','auto','cj','game','xingzuo','it','mil','health','ty','edu']
clsMap = {'yl':0,'auto':1, 'cj':2,'game': 3,'xingzuo':4, 'it':5, 'mil':6, 'health':7, 'ty':8,'edu':9}
def get_cls_Files(cls):
    fstop = open('train_seg_3/' + cls, 'r', encoding='utf-8', errors='ignore')
    words = fstop.read()
    fstop.close()
    word_lists = words.split('\n')
    files = []
    del word_lists[-1]
    for word_list in word_lists:
        word = word_list.split()
        files.append(word)
    return files

def get_test_Files(cls):
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

def make_cls_dict(files):
    dictionary = corpora.Dictionary()
    #file 为['hhh','ppp']
    for file in files:
        dictionary.add_documents([file])
    return dictionary


def make_word_dict():
    for cls in clsList:
        word_dictionary.merge_with(make_cls_dict(get_cls_Files(cls)))
        print('merge')

def make_train_corpus():
    corpus = []
    for cls in clsList:
        temp = [word_dictionary.doc2bow(file) for file in get_cls_Files(cls)]
        corpus = corpus + temp
        for i in range(len(temp)):
            train_tag.append(clsMap.get(cls))
    return corpus

def make_test_corpus():
    corpus = []
    for cls in clsList:
        temp = [word_dictionary.doc2bow(file) for file in get_test_Files(cls)]
        corpus = corpus + temp
        for i in range(len(temp)):
            test_tag.append(clsMap.get(cls))
    return corpus

def chi_process():
    make_word_dict()
    corpus = make_train_corpus()
    print ('正在计算文档TF-IDF --')
    corpus_tfidf = models.TfidfModel(corpus)[corpus]
    corpus_tfidf.save('corpus_tfidf_model')
    #corpus_tfidf = models.TfidfModel.load('corpus_tfidf_model') load tf_ifd model
    print ("建立文档TF——IDF完成")
    print ('LDA模型拟合推断--')
    num_topics = 100  #设置主题数目
    lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=word_dictionary,alpha=0.01,eta=0.01,
                          minimum_probability=0.001, update_every=1, chunksize=100, passes=1)
    lda.save('corpus_lda.model') #保存一下训练的模型
    #lda = models.LdaModel.load('corpus_lda.model')
    print ('LDA模型完成，训练时间为%.3f秒')


    # new_word_dictionary.save('file/chidict.dict')
    # print(new_word_dictionary)
    # return new_word_dictionary

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
    # clsList = {0:'yl', 1:'auto', 2:'cj', 3:'game',4:'xingzuo', 5:'it', 6:'mil', 7:'health', 8:'ty',9:'edu'}
    train_set = []
    train_tag = []
    test_set = []
    test_tag = []
    lda = models.LdaModel.load('corpus_lda.model')
    corpus_train = make_train_corpus()
    train_set = lda[corpus_train]
    corpus_test = make_test_corpus()
    test_set = lda[corpus_test]
    predictor = svm_classify(train_set, array(train_tag,ndmin=2).T, test_set, array(test_tag,ndmin=2).T)
    # dictionary = corpora.Dictionary.load('file/chidict.dict')
    # number_of_feature = len(dictionary)
    #
    # word_mtrix_list = {}
    # result = {}
    # #各个分类下，错分的单词去向
    # for k,cls in clsList.items():
    #      temp = make_cls_mtrix(cls)
    #      train_set = train_set + temp
    #      for i in range(0,len(temp)):
    #          train_tag.append(k)
    # train_matrix = matutils.corpus2csc(train_set,num_terms=number_of_feature).T
    # for k,cls in clsList.items():
    #      temp =  make_test_mtx(cls)
    #      test_set = test_set + temp
    #      for i in range(0,len(temp)):
    #          test_tag.append(k)
    # test_matrix = matutils.corpus2csc(test_set,num_terms=number_of_feature).T
    # print('test')
    # predictor = svm_classify(train_matrix, array(train_tag,ndmin=2).T, test_matrix, array(test_tag,ndmin=2).T)