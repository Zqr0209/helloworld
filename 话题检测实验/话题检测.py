# *- coding: utf-8 -*-


import sys
import os
import jieba
import jieba.analyse
import codecs
import re
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from zhon.hanzi import punctuation
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import importlib

importlib.reload(sys)



# Step1:Read file

def read_file():
    path = "C:\\Users\\lenovo\\Desktop\\话题检测实验\\第五课实验说明及训练集\\train\\C4-Literature\\"
    resName = "Result.txt"
    if os.path.exists(resName):
        os.remove(resName)
    result = codecs.open(resName, 'w', 'utf-8')

    num = 1
    while num <= 33:
        name = "C4-Literature%d" % num
        fileName = path + str(name) + ".txt"
        source = open(fileName, 'r')
        line = source.readline()
        line = line.strip('\n')
        line = line.strip('\r')

        while line != "":
            
            line = line.replace('\n', ' ')
            line = line.replace('\r', ' ')
            result.write(line + ' ')
            line = source.readline()
        else:
            result.write('\r\n')
            source.close()
        num = num + 1

    else:
        result.close()

    return resName

# Step2:cut file and get feature vector matrixes
def get_TFIDF(resname,filename):
    corpus = []  # 语料库 空格连接

    # 读取语料  一行为一个文档
    for line in open(resname, 'r',encoding='UTF-8').readlines():
        line=line.strip() # 删除末尾的'/n'
        string = re.sub(r"[%s]+" % punctuation, "",line)  # 去标点
        seg_list = jieba.cut(string,cut_all=False) # 结巴分词
        corpus.append(' '.join(seg_list))

    # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()

    # 该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()

    # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names()

    # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
    weight = tfidf.toarray()

    # fileName = "TF-IDF_Result.txt"
    result = codecs.open(filename, 'w', 'utf-8')
    for j in range(len(word)):
        result.write(word[j] + ' ')
    result.write('\r\n\r\n')

    # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    for i in range(len(weight)):
        for j in range(len(word)):
            result.write(str(weight[i][j]) + ' ')
        result.write('\r\n\r\n')

    result.close()
    return weight

# Step3:Clustering
def K_Means(weight):
    print('Start Kmeans:')

    # 选择3个中心点
    clf = KMeans(n_clusters=3)

    # clf.fit(X)可以把数据输入到分类器里
    clf.fit(weight)

    # 3个中心点
    print('cluster_center:')
    print((clf.cluster_centers_))

    # 每个样本所属的簇
    # print(clf.labels_)
    print('list_number label  ')
    i = 1
    while i <= len(clf.labels_):
        print(i,'          ',clf.labels_[i - 1])
        i = i + 1

    # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    print('inertia:')
    print((clf.inertia_))

    # 保存模型
    joblib.dump(clf, 'km.pkl')

# Step4:Test
def test():
    path = "C:\\Users\\lenovo\\Desktop\\话题检测实验\\第五课实验说明及训练集\\test\\C4-Literature\\"
    test_name = "test_result.txt"
    file_name = "test_TF-IDF.txt"
    if os.path.exists(test_name):
        os.remove(test_name)
    test_result = codecs.open(test_name,'w','utf-8')

    for file in os.listdir(path):
        source = open(path + file,'r')
        line = source.readline()
        line = line.strip('\n')
        line = line.strip('\r')

        while line !="":
            
            line = line.replace('\n',' ')
            line = line.replace('\r',' ')
            test_result.write(line + ' ')
            line=source.readline()

        else:
            test_result.write('\n\r')
            source.close()
    test_result.close()

    test_weight = get_TFIDF(test_name,file_name)

    # 载入保存的模型
    clf = joblib.load('km.pkl')

    clf.fit_predict(test_weight)

    print('list_number label  ')
    i = 1
    while i <= len(clf.labels_):
        print(i, '          ', clf.labels_[i - 1])
        i = i + 1


if __name__ == '__main__':
    resName = read_file()
    filename = "TF-IDF_Result.txt"
    weight=get_TFIDF(resName,filename)
    K_Means(weight)
    test()
