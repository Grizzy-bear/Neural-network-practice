# encoding: utf-8
"""
@version:3.6
@author:lamplight
@file:keras扩充图片集.py
@time:2017/11/2323:45
"""
import os
import re
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'

)

write_path = "E:\\Pictures\\pri\\"

def eachFile(filepath):
    count = 0
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s%s'%(filepath, allDir))
    write_child = os.path.join('%s%s' % (write_path,allDir))
    img = load_img(child)
    nul_num = re.findall(r'\d',child)
    nul_num = int(nul_num[0])
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(
        x,
        batch_size=1,
        save_to_dir=write_path,
        save_prefix=nul_num,save_format='jpeg'
    ):
        count +=1
        i+=1
        if i >= 10:
            break
    return count
count = eachFile("E:\\Pictures\\守望先锋\\")
print ("一共产生了%d张图片"%count)