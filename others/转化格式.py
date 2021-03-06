# encoding: utf-8
"""
@version:3.6
@author:lamplight
@file:转化格式.py
@time:2017/12/81:10
"""
#-*-coding:utf-8 -*-
import os
folder ='语料库' #存储文本的目录

listDir = [ dirs[0] for dirs in os.walk(folder)][1:]#获取所有的子目录
for dataDir in listDir:
    files = [os.path.join(dataDir,i) for i in os.listdir(dataDir)]#获取绝对路径
    for words in files:
        pos,filename = os.path.split(words)
        newFile = file(os.path.join(pos,filename[:-4]+'_.txt'),'w')#建立新文件
        try:#转码
            newFile.write(file(words,'r').read().decode('gb2312').encode('utf-8'))
        except:
            newFile.write(file(words,'r').read().decode('gbk','ignore').encode('utf-8'))
        newFile.close()
        os.remove(words)#删除旧文件