# encoding: utf-8
"""
@version:3.6
@author:lamplight
@file:sklearn_语料分析.py
@time:2017/12/820:18
"""
from sklearn.datasets.base import Bunch
import os
import pickle

bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])

wordbag_path = "train_exam\\sk_01\\sport\\8.txt"
seg_path = "train_exam\\sk_02\\"

catelist = os.listdir(seg_path)
bunch.target_name.extend(catelist)
def readfile(path):
    fp = open(path, "r", encoding='gb18030', errors='ignore')
    content = fp.read()
    fp.close()
    return content
for mydir in catelist:
    class_path = seg_path + mydir + "\\"
    file_list = os.listdir(class_path)
    for file_path in file_list:
        fullname = class_path + file_path
        bunch.label.append(mydir)
        bunch.filenames.append(fullname)
        bunch.contents.append(readfile(fullname).strip())

file_obj = open(wordbag_path, "wb")
pickle.dump(bunch, file_obj)
file_obj.close()
print("文本对象构建结束")