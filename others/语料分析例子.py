# encoding: utf-8
"""
@version:3.6
@author:lamplight
@file:语料分析例子.py
@time:2017/12/817:10
"""
import sys
import os
import jieba

def savefile(savepath, content):
    fp = open(savepath, "w", encoding="gb18030")
    fp.write(content)
    fp.close()

def readfile(path):
    fp = open(path, "r", encoding="gb18030")
    content = fp.read()
    fp.close()
    return  content

#主程序
corpus_path = "train_exam\\01\\"
seg_path = "train_exam\\02\\"

catelist = os.listdir(corpus_path)
# print(catelist)
for mydir in catelist:
    class_path = corpus_path+mydir+"\\"
    seg_dir = seg_path+mydir+"\\"
    if not os.path.exists(seg_dir):
        os.mkdir(seg_dir)
    file_list = os.listdir(class_path)
    # print(file_list)
    for file_path in file_list:
        fullname = class_path + file_path
        content = readfile(fullname).strip()
        # print(content)
        content = content.replace(" \n", " ").strip()
        content_seg = jieba.cut(content, cut_all=True)
        # print(content)
        savefile(seg_dir+file_path, "/".join(content_seg))

    print("finishing")