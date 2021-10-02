import streamlit as st
from stmain import Main
st.set_option('deprecation.showPyplotGlobalUse', False)  # 防止出现烦人的报错信息，

file1=None
file3=st.text_input('输入停用词的存储位置')
st.info('例如：D:\coding\TF-idf\data\stopwords.txt')
file2 = st.text_input('请输入图片存储的地址')
st.info('例如：D:\coding\TF-idf\data\st1.jpg')

an=st.radio('单文本,多文本',("单文本","多文本"))

panduan=0
try:
    if an=="单文本":
        panduan=0
        file1=st.text_input('请输入单文本的地址')
        st.info('例如：D:\coding\TF-idf\data\sttxt.txt')
    else:
        panduan=1
        file11 = st.text_input('请输入多文字的地址')
        st.info('例如：D:\coding\TF-idf\data\st.csv')
        st.info('此处csv文件的编码为ANSI')

    if (file1!=None and panduan==0):
        a = Main(file1, file2,file3)
        data,datawenzi=a.main1()
    elif(file11!=None and panduan == 1):
        a=Main(file11,file2,file3)
        data, datawenzi = a.main2()

    anniu=st.radio("词频,TF-IDF,TextRank",("词频","TF-IDF","TextRank"))

    if anniu=="词频":
        st.info("词频")
        try:
            a.zidingyiciyun(a.cipin(data))
        except:print(None)

    elif anniu=="TF-IDF":
        st.info("TF-IDF")
        try:
            a.zidingyiciyun(a.tf(datawenzi))
        except:print(None)
    elif anniu=="TextRank":
        try:
            a.zidingyiciyun(a.tr(datawenzi))
        except:print(None)

except:print(None)