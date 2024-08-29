import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
from xml.etree import ElementTree
from matplotlib import pyplot as pl
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras import datasets, layers, models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.applications import VGG19
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input
keras = tf.keras

class_names = ['person','person-like']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}
n_classes = 2
size = (200,200)

train_data_gen = ImageDataGenerator(width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True)

def load_data():
    datasets = ['Train/Train', 'Test/Test', 'Val/Val']
    output = []

    for dataset in datasets:
        imags = []
        labels = []
        directoryA = dataset +"/Annotations"
        directoryIMG = dataset +"/JPEGImages/"
        file = os.listdir(directoryA)
        img = os.listdir(directoryIMG)
        file.sort()
        img.sort()

        i = 0
        for xml in file:

            xmlf = os.path.join(directoryA,xml)
            dom = ElementTree.parse(xmlf)
            vb = dom.findall('object')
            label = vb[0].find('name').text
            labels.append(class_names_label[label])

            img_path = directoryIMG + img[i]
            curr_img = cv2.imread(img_path)
            curr_img = cv2.resize(curr_img, size)
#             if is_train:
#                 curr_img = train_data_gen.random_transform(curr_img)
            imags.append(curr_img)
            i +=1
        
        imags = np.array(imags, dtype='float32')
        imags = imags / 255
        
      #  labels = pd.DataFrame(labels)
        labels = np.array(labels, dtype='int32')

        output.append((imags, labels))
    return output

(train_images, train_labels),(test_images, test_labels),(val_images, val_labels) = load_data()



base_model1 = InceptionV3(include_top=False,
                         weights='imagenet',
                         input_shape=(200, 200, 3))

for layer in base_model1.layers[:15]:
    layer.trainable = False

last_output = base_model1.output
x = GlobalAveragePooling2D()(last_output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(2, activation='softmax')(x)

model_inception = Model(inputs=base_model1.inputs, outputs=outputs)
model_inception.compile(optimizer=Adam(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model_name1 = "inceptionv3_model.h5"
checkpoint1 = ModelCheckpoint(model_name1,
                              monitor="val_loss",
                              mode="min",
                              save_best_only=True,
                              save_weights_only=False,
                              verbose=1)

lr_reduction1 = ReduceLROnPlateau(monitor='val_loss',
                                patience=2,
                                verbose=1,
                                factor=0.3,
                                min_lr=0.00001)

earlystopping1 = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, restore_best_weights=True)

try:
    history = model_inception.fit(train_images, train_labels,
                        epochs=10,
                        validation_data=(test_images, test_labels),
                        callbacks=[checkpoint1, earlystopping1, lr_reduction1])
except KeyboardInterrupt:
    print("\nTraining Stopped")

model_json = model_inception.to_json()
with open("inceptionv3_model.json", "w") as json_file:
    json_file.write(model_json)


base_model2 = VGG19(include_top = False,
                       weights = 'imagenet',
                       input_shape = (200,200,3))


for layer in base_model2.layers[:15]:
    layer.trainable = False
last_output = base_model2.output
x = GlobalAveragePooling2D()(last_output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(2, activation='softmax')(x)
model_vgg = Model(inputs=base_model2.inputs, outputs=outputs)
model_vgg.compile(optimizer=Adam(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model_name2 = "VGG19_model.h5"
checkpoint2 = ModelCheckpoint(model_name2,
                            monitor="val_loss",
                            mode="min",
                            save_best_only = True,
                            save_weights_only=False,
                            verbose=1)

lr_reduction2 = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.3, 
                                            min_lr=0.00001)

earlystopping2 = EarlyStopping(monitor='val_loss',min_delta = 0, patience = 3, verbose = 1, restore_best_weights=True)

try:
    history = model_vgg.fit(train_images, train_labels,
                        epochs=10,
                        validation_data=(test_images, test_labels),
                        callbacks=[checkpoint2,earlystopping2, lr_reduction2])
except KeyboardInterrupt:
    print("\nTraining Stopped")

model_json = model_vgg.to_json()
with open("VGG19_model.json", "w") as json_file:
    json_file.write(model_json)


base_model3 = ResNet101(include_top = False,
                       weights = 'imagenet',
                       input_shape = (200,200,3))


for layer in base_model3.layers[:15]:
    layer.trainable = False
last_output = base_model3.output
x = GlobalAveragePooling2D()(last_output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(2, activation='softmax')(x)
model_resnet = Model(inputs=base_model3.inputs, outputs=outputs)
model_resnet.compile(optimizer=Adam(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model_name3 = "ResNet101_model.h5"
checkpoint3 = ModelCheckpoint(model_name3,
                            monitor="val_loss",
                            mode="min",
                            save_best_only = True,
                            save_weights_only=False,
                            verbose=1)

lr_reduction3 = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.3, 
                                            min_lr=0.00001)

earlystopping3 = EarlyStopping(monitor='val_loss',min_delta = 0, patience = 3, verbose = 1, restore_best_weights=True)

try:
    history = model_resnet.fit(train_images, train_labels,
                        epochs=10,
                        validation_data=(test_images, test_labels),
                        callbacks=[checkpoint3,earlystopping3, lr_reduction3])
except KeyboardInterrupt:
    print("\nTraining Stopped")

model_json = model_resnet.to_json()
with open("ResNet101_model.json", "w") as json_file:
    json_file.write(model_json)