# encoding: utf-8
"""
@version:3.6
@author:lamplight
@file:汉字识别.py
@time:2017/12/1223:26
"""
from keras.applications import vgg16
from keras.preprocessing import image
import numpy as np
from keras import backend as K

from keras.models import Model
model = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

def preprocess_image(image_path):
    # 使用keras内置函数读取图片，没有全连接层，target_size随便设置
    img = image.load_img(image_path, target_size=(img_nrows, img_ncols))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img - vgg16.preprocess_input(img)
    return img

def deprocess_image(x):
    x = np.reshape((img_nrows, img_ncols))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    #‘bgr' -> rgb
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 225).astype('unit8')
    return x

base_image = K.variable(preprocess_image(r"C:\Users\Vencent\Desktop\13.jpg"))
style_reference_img = K.variable(preprocess_image(r"C:\Users\Vencent\Desktop\14.jpg"))

combination_image = K.placeholder((1, img_nrows, img_nclos, 3))

input_tensor = K.concatenate([base_image, style_reference_img, combination_img], axis=0)

def gram_matrix(x):
    assert K.ndim(x) == 3
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination)  ==3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_nclos
    return K.sum(K.square(S-C)) / (4. * (channels **2) * (size**2))

def content_loss(base, combination):
    return K.sum(K.square(combination-base))

def total_variation_loss(x):
    assert K.ndim(x) == 4
    a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

output_dict = dict([(layer.name, layer.output) for layer in model.layers])

loss = K.variable(0.)
layer_features = output_dict['block4_conv2']
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(base_image_features,
                                      combination_features)

feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']

for layer_name in feature_layers:
    layer_features = output_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight/len(feature_layers)) * sl

loss += total_variation_weight * total_variation_loss(combination_image)




