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
import os

matplotlib.rcParams['figure.figsize'] = (18.0, 18.0)  # 定义长宽
from wordcloud import WordCloud, ImageColorGenerator
import pylab


class Main:
    def __init__(self, file1, file2, file3):
        self.file1 = file1  # 待识别文字所存储的位置
        # 例如 D:\conda\TF-idf\data\test.txt
        self.file2 = file2  # 图片所存储的位置
        # 例如 D:\conda\TF-idf\data\1.jpg
        self.file3 = file3  # 停用词所存储的位置
    def main1(self):

        data = list(open(r''+self.file1, encoding='utf-8'))  # 读取txt文件

        reg = "[^0-9A-Za-z\u4e00-\u9fa5]"  # 正则表达式，强制去掉标点符号
        data[0]= re.sub(reg, '', data[0])
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

        stop = pd.read_csv(self.file3, encoding='utf-8', index_col=False, sep='\t',
                           names=['stop'],
                           quoting=3)  # 默认强行读取此停用词文件，可手动改路径
        data = data[~data.ci.isin(stop.stop)]  # 用匹配的方式，将data中的停用词给去除
        return data,datawenzi

        #
        # Main.zidingyiciyun(self,Main.cipin(data))
        # zidingyiciyun(self,tf(datawenzi))
        # zidingyiciyun(self,tr(datawenzi))



    def main2(self):
        datacsv=pd.read_csv(r''+self.file1,encoding='ANSI')

        datacsv=datacsv['content']
        datacsv1=datacsv.values.tolist()
        data=['']
        for i in range(len(datacsv1)):
            data[0] += (datacsv1[i])
        reg = "[^0-9A-Za-z\u4e00-\u9fa5]"  # 正则表达式，强制去掉标点符号
        datawenzi = re.sub(reg, '', data[0])
        a=jieba.lcut(datawenzi)
        data=pd.DataFrame(a,columns=['ci'])
        stop = pd.read_csv(self.file3, encoding='utf-8', index_col=False, sep='\t',
                           names=['stop'],
                           quoting=3)  # 默认强行读取此停用词文件，可手动改路径
        data = data[~data.ci.isin(stop.stop)]
        return data, datawenzi
        #
        # zidingyiciyun(self,cipin(data))
        # zidingyiciyun(self,tf(datawenzi))
        # zidingyiciyun(self,tr(datawenzi))

    def cipin(self,data1):  # 导入data
        data1gr = data1.groupby('ci')['ci'].agg(np.size)
        data1gr.name = 'shu'
        data1gr = data1gr.reset_index().sort_values(by=['shu'], ascending=False)
        return data1gr

    def tf(self,data2):  # 导入datawenzi
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

    def tr(self,data3):
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

    def ciyunmorenbeijing(self,data4):  # 任意导入 cipin tf tr 的返回的数
        filepathwc = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir)) + '\data\simhei.ttf'
        wc = WordCloud(font_path=r''+filepathwc, background_color='white',
                       max_font_size=200)  # 自己安排自己的词的数量
        wf = {x[0]: x[1] for x in data4.head(len(data4)).values}  # 从中抽取词，head()中的词的数量可以自定义，最高不能够超过len(data4)
        wc = wc.fit_words(wf)
        plt.axis("off")
        plt.imshow(wc)
        pylab.show()
        st.pyplot()

    def zidingyiciyun(self,data5):  # 任意导入 cipin tf tr 的返回的数
        filepath2=self.file2
        bimg = imageio.imread(r''+filepath2)
        filepathwc=os.path.abspath(os.path.join(os.getcwd(), os.path.pardir)) + '\data\simhei.ttf'
        wc = WordCloud(font_path=r''+filepathwc, background_color='white', max_font_size=200,
                       mask=bimg)  # 自己安排自己的词的数量 mask表示根据具体情况设置词云
        wf = {x[0]: x[1] for x in data5.head(len(data5)).values}  # 从中抽取词，head()中的词的数量可以自定义，最高不能够超过len(data4)
        wc = wc.fit_words(wf)
        bC = ImageColorGenerator(bimg)
        plt.axis("off")
        plt.imshow(wc.recolor(color_func=bC))
        st.pyplot()
