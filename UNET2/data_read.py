import os
import numpy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image_path = "./capture/"

img_list = os.listdir(image_path) #디렉토리 내 모든 파일 불러오기
img_list = [img for img in img_list if img.endswith(".bmp")] #지정된 확장자만 필터링

print ("img_list_jpg: {}".format(img_list))

n_train = 121
n_val =  10
n_test = 10


path_save_train = os.path.join(image_path, 'train')
path_save_val = os.path.join(image_path, 'val')
path_save_test = os.path.join(image_path, 'test')

if not os.path.exists(path_save_train):
    os.makedirs(path_save_train)
if not os.path.exists(path_save_val):
    os.makedirs(path_save_val)
if not os.path.exists(path_save_test):
    os.makedirs(path_save_test)

imgs_input = []
imgs_label = []
for i in range(img_list):
    imgs_input.append(Image.open(os.path.join(image_path, img_list[i])))

    if i==(len(img_list)-1):
        imgs_label.append(Image.open(os.path.join(image_path, img_list[i])))
    else:
        imgs_label.append(Image.open(os.path.join(image_path, img_list[i+1])))

ny, nx = imgs_label[0].size
nframe = len(img_list)

id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

##
offset_nframe = 0

for i in range(n_train):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)
