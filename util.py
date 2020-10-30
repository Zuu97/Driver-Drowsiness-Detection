import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from pathlib import Path

from variables import*

np.random.seed(seed)

def get_class_labels():
    return os.listdir(train_dir)

def preprocessing_function(img):
    H, W, C = img.shape
    if (H > crop_size_middle * 2) and (W > crop_size_middle * 2):
        Hcenter = H // 2
        Wcenter = W // 2
        process_img = img[Hcenter - crop_size_middle : Hcenter + crop_size_middle,
                          Wcenter - crop_size_middle : Wcenter + crop_size_middle,  
                                                     :]
        return process_img
    else:
        return img

def image_data_generator():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                    rescale = rescale,
                                    rotation_range = rotation_range,
                                    shear_range = shear_range,
                                    zoom_range = zoom_range,
                                    width_shift_range=shift_range,
                                    height_shift_range=shift_range,
                                    horizontal_flip = True,
                                    validation_split= val_split,
                                    # preprocessing_function = preprocessing_function
                                    )
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = rescale)


    train_generator = train_datagen.flow_from_directory(
                                    train_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = batch_size,
                                    classes = get_class_labels(),
                                    subset='training',
                                    shuffle = True)

    validation_generator = train_datagen.flow_from_directory(
                                    train_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = valid_size,
                                    classes = get_class_labels(),
                                    subset='validation',
                                    shuffle = True)

    test_generator = test_datagen.flow_from_directory(
                                    test_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = batch_size,
                                    classes = get_class_labels(),
                                    shuffle = True)

    return train_generator, validation_generator, test_generator

def move_image(source_path, destination_path):
    Path(source_path).rename(destination_path)

def split_train_test():
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        labels = os.listdir(train_dir)

        for label in labels:
            source_label_dir = os.path.join(train_dir, label)
            destination_label_dir = os.path.join(test_dir, label)

            if not os.path.exists(destination_label_dir):
                os.makedirs(destination_label_dir)

            image_arr = os.listdir(source_label_dir)
            Ntest = int(test_split * len(image_arr))
            test_img_arr = np.random.choice(image_arr, Ntest, replace=False)

            for test_img in test_img_arr:
                destination_path = os.path.join(destination_label_dir, test_img)
                source_path = os.path.join(source_label_dir, test_img)
                move_image(source_path, destination_path)