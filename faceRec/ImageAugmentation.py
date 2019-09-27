# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:47:07 2019

@author: Vishnu.Kumar1
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from PIL import Image
import random
import os 

path = os.getcwd()
data_folder = path + '\DemoData'
folders = os.listdir(data_folder)

for i in folders:
    if len(os.listdir(data_folder + '/' + i)) == 1:
        print(data_folder + '/' + i)
        input_path = data_folder + '/' + i + '/' + os.listdir(data_folder + '/' + i)[0]
        output_path = data_folder + '/' + i + '/' + i + '_random{}.png'
        count = 10
        image_file = Image.open(input_path)
        image_file_blackwhite = image_file.convert('1')
        image_file_grey = image_file.convert('LA')
        image_file_blackwhite.save(output_path.format(random.randint(50,100)))
        image_file_grey.save(output_path.format(random.randint(100,150)))
        gen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True
                )
        image = img_to_array(load_img(input_path))
        image = image.reshape((1,) + image.shape)
        images_flow = gen.flow(image, batch_size=1)
        
        for i, new_images in enumerate(images_flow):
            new_image = array_to_img(new_images[0], scale=True)
            new_image.save(output_path.format(i + 1))
            if i >= count:
                break