# encoding: utf-8
"""
@version:3.6
@author:lamplight
@file:练习03.py
@time:2017/11/2923:17
"""
import pygame
from PIL import Image
import random
import os
from io import StringIO

f = open('word.txt','r')
words = f.readline().strip()
f.close()
# print(words)
# im = Image.new('RGB', (28, 28), (255, 255, 255))
# print(im.show())
def numRandom():
    return random.randint(0, 9999999999999)

def paste(text, font, imgName, area=(5,3)):
    im = Image.new('RGB', (28, 28), (255, 255, 255))
    # print(im)
    rtext = font.render(text, True, (0, 0, 0), (255, 255, 255))
    sio = StringIO()
    pygame.image.save(rtext, sio)
    line = Image.open(sio)
    im.paste(line, area)
    im.save(imgName)

ttf = os.listdir('./ttf')
# for n in range(len(ttf)):
#     ttf[n] = './ttf/' + ttf[n]
#     print(ttf[n])
pygame.init()
pygame.font.init()
font_path =[]
for m in  range(len(ttf)):
    ttf[m] = './ttf/' + ttf[m]
    font_path.append(ttf[m])
font = pygame.font.Font(font_path, 25)
os.chdir('./words')
text_list = words.split(' ')
length = len(text_list)
for i in range(length):
    text = text_list[i]#.decode('utf-8', 'ignore')
    imgName = text_list[i] + '_' + str(numRandom()) + '.png'
    if os.path.isfile(imgName):
        imgName = text_list[i] + '_' + str(numRandom()) + str(numRandom()) + '.png'
        paste(text, font, imgName)

    else:
        try:
            paste(text, font, imgName)
        except:
            pass
os.chdir('..')


