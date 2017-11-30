import numpy
from gensim import corpora
#词和词频
word_dictionary = corpora.Dictionary()
new_word_dictionary = corpora.Dictionary()
#结构为{‘class’:[]}
seperate_word_dict = {}
word_vector_list = {}
clsList = ['yl','auto','cj','game','xingzuo','it','mil','health','ty','edu']

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


#
# def make_vector(files,cls):
#     word_vec = [word_dictionary.doc2bow(file) for file in files]
#     word_vector_list[cls] = word_vec

# def get_total_wordnum(word):
#     return word_dictionary.dfs.get(word_dictionary.token2id.get(word,-1),0)

# def get_cls_wordnum(word,cls):
#     cls_word_vec = word_vector_list.get(cls)
#     cls_sum = 0
#     for vec in cls_word_vec:
#        for word_tuple in vec:
#            if(word_tuple[0] == word_dictionary.token2id.get(word,-1)):
#                cls_sum = cls_sum + word_tuple[1]
#     return cls_sum

# def get_cls_totalnum(cls):
#     cls_word_vec = word_vector_list.get(cls)
#     return
'''
    χ2=N∗(AD−BC)2/(A+B)(C+D)(A+C)(B+D)
'''
def chi_process():
    make_word_dict()
    for cls in clsList:
        seperate_word_dict[cls] = chi_cls_process(cls)
        with open('file/'+str(cls)+'.txt','w') as f:
            f.write(str([value for value in seperate_word_dict[cls].itervalues()]) + '\n')
        print(seperate_word_dict[cls])
        new_word_dictionary.merge_with(seperate_word_dict[cls])
    #新增
    # clsListCopy = []
    # for cls in clsList:
    #     clsListCopy.append(cls)
    # vList = []
    # for cls in clsList:
    #     clsListCopy.remove(cls)
    #     #某类特征
    #     value_cls = [value for value in seperate_word_dict[cls].itervalues()]
    #     for clsC in clsListCopy:
    #         for k,v in seperate_word_dict[clsC].items():
    #             if v in value_cls:
    #                 vList.append(v)
    # vList = list(set(vList))
    # kList = []
    # for key,value in new_word_dictionary.token2id.items():
    #     if(key in vList):
    #         kList.append(value)
    # new_word_dictionary.filter_tokens(bad_ids=kList)
    #新增
    new_word_dictionary.save('file/chidict.dict')
    print(new_word_dictionary)
    return new_word_dictionary

def chi_cls_process(cls):
    dict_cls = make_cls_dict(get_cls_Files(cls))
    dict_cls.filter_extremes(no_below=5, no_above=1.0, keep_n=1000000)
    dict_cls.compactify()
    # dict_cls_items = dict(dict_cls.items())
    dict_cls_items_dfs = dict_cls.dfs
    # dict_items = dict(word_dictionary.items())

    # dict_items = dict_items
    dict_chi = {}
    a = int
    b = int
    c = int
    d = int
    for key,value in dict_cls_items_dfs.items():
        a = value
        w = dict_cls.token2id
        w = dict((v,k) for k,v in w.items())
        word = w.get(key)
        key2 = word_dictionary.token2id.get(word)
        if(word_dictionary.dfs.get(key2) == None):
            print(word + str(key2))
        b = word_dictionary.dfs.get(key2) - a
        c = dict_cls.num_docs - a
        d = word_dictionary.num_docs - dict_cls.num_docs - b
        x2 = (float)(a+b+c+d)*(a*d-b*c)*(a*d-b*c)/((a+b)*(a+c)*(b+d)*(c+d))
        dict_chi[key] = x2
    dict_token_ids = []
    for k in dict(sorted(dict_chi.items(),key= lambda x:x[1])[-600:]).keys():
        dict_token_ids.append(k)
    # print(dict_token_ids)
    dict_cls.filter_tokens(good_ids=dict_token_ids)
    dict_cls.compactify()
    dict_cls_items_dfs.clear()
    dict_chi.clear()
    for i in dict_token_ids:
        dict_token_ids.remove(i)
    print('ok')
    return dict_cls
