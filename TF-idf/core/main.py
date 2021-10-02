import jieba
import jieba.analyse as analyse
import jieba.posseg  # 输出带词性
import copy
import wordcloud
import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
from wordcloud import WordCloud  # 词云包
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import imageio

matplotlib.rcParams['figure.figsize'] = (18.0, 18.0)  # 定义长宽
from wordcloud import WordCloud, ImageColorGenerator
import pylab

data = list(open(r'D:\conda\TF-idf\data\test.txt', encoding='utf-8'))
datawenzi = copy.copy(data[0])
data = jieba.lcut(data[0])  # 输出不带词性
# print(jieba.posseg.lcut(data[0]))  # 输出带词性
data = pd.DataFrame(data, columns=['ci'])
# 方法一：用正则表达式的形式去除掉停用词中的换行符号
# stop=list(open(r'D:\conda\TF-idf\data\stopwords.txt',encoding='utf-8'))
# for i in range(len(stop)):
#     stop[i]=re.sub('\n','',stop[i])
# stop=pd.DataFrame(stop,columns=['stop'])
# 方法二：直接用pd获取停用词大全 其中quoting=3 代表将 英文双引号的内容也要识别出来，而txt文件的默认编码方式为encoding='utf-8'
stop = pd.read_csv(r'D:\conda\TF-idf\data\stopwords.txt', encoding='utf-8', index_col=False, sep='\t', names=['stop'],
                   quoting=3)
data = data[~data.ci.isin(stop.stop)]  # 用匹配的方式，将data中的停用词给去除


def cipin(data1):  # 导入data
    data1gr = data1.groupby('ci')['ci'].agg(np.size)
    data1gr.name = 'shu'
    data1gr = data1gr.reset_index().sort_values(by=['shu'], ascending=False)
    return data1gr


def tf(data2):  # 导入datawenzi
    key = analyse.extract_tags(data2, topK=30, withWeight=True, allowPOS=())  # withWeight为加上权重
    keyci = []
    keyshu = []
    for i in range(len(key)):
        keyci.append(key[i][0])
        keyshu.append(key[i][1])
    keyci1 = pd.DataFrame(keyci, columns=['ci'])
    keyshu1 = pd.DataFrame(keyshu, columns=['shu'])
    keynew = pd.concat([keyci1, keyshu1], axis=1)  # 横向链接
    df1 = keynew.sort_index(axis=0, ascending=True, inplace=False)  # inplace=False 代表对其进行一个赋值,ascending=True代表升序
    return df1


def tr(data3):
    key = analyse.textrank(data3, topK=30, withWeight=True)  # withWeight为加上权重
    print(key)
    keyci = []
    keyshu = []
    for i in range(len(key)):
        keyci.append(key[i][0])
        keyshu.append(key[i][1])
    keyci1 = pd.DataFrame(keyci, columns=['ci'])
    keyshu1 = pd.DataFrame(keyshu, columns=['shu'])
    keynew = pd.concat([keyci1, keyshu1], axis=1)  # 横向链接
    df1 = keynew.sort_index(axis=0, ascending=True, inplace=False)  # inplace=False 代表对其进行一个赋值,ascending=True代表升序
    return df1

def ciyunmorenbeijing(data4):  # 任意导入 cipin tf tr 的返回的数

    wc = WordCloud(font_path=r'D:\conda\TF-idf\data\simhei.ttf', background_color='white',
                   max_font_size=200)  # 自己安排自己的词的数量
    wf = {x[0]: x[1] for x in data4.head(len(data4)).values}  # 从中抽取词，head()中的词的数量可以自定义，最高不能够超过len(data4)
    wc = wc.fit_words(wf)
    plt.axis("off")
    plt.imshow(wc)
    pylab.show()


def zidingyiciyun(data5):  # 任意导入 cipin tf tr 的返回的数
    bimg = imageio.imread(r'D:\conda\TF-idf\data\1.jpg')
    wc = WordCloud(font_path=r'D:\conda\TF-idf\data\simhei.ttf', background_color='white', max_font_size=200,
                   mask=bimg)  # max_font_size自己安排自己的词的大小程度 mask表示根据图片情况生成词云
    wf = {x[0]: x[1] for x in data5.head(len(data5)).values}  # 从中抽取词，head()中的词的数量可以自定义，最高不能够超过len(data4)
    wc = wc.fit_words(wf)
    bC = ImageColorGenerator(bimg)
    plt.axis("off")
    plt.imshow(wc.recolor(color_func=bC))
    pylab.show()

# zidingyiciyun(cipin(data))
# zidingyiciyun(tf(datawenzi))
# zidingyiciyun(tr(datawenzi))