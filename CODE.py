 #importing other required libraries
import numpy as np
import pandas as pd
from sklearn.utils.multiclass import unique_labels
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD,Adam
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout
from keras.utils import to_categorical
 #DATA PROCESSING
 
Input_shape = (224, 224, 3)
Generator = ImageDataGenerator(validation_split =0.2,rotation_range=10, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True,fill_mode="nearest")
 
train_generator = Generator.flow_from_directory("/content/DATAENTERAV3/DATASETS", 
                                                   target_size=Input_shape[:2],
                                                    classes=['Score0', 'Score1', 'Score2', 'Score3', 'Score4', 'Score5'],
                                                   batch_size=32,
                                                    shuffle = False,
                                                  subset='training',
                                                   class_mode='categorical')

val_generator = Generator.flow_from_directory("/content/DATAENTERAV3/DATASETS", 
                                                   target_size=Input_shape[:2],
                                                    classes=['Score0', 'Score1', 'Score2', 'Score3', 'Score4', 'Score5'],
                                                   batch_size=32,
                                                    shuffle = False,
                                                   subset='validation',
                                                   class_mode='categorical')
 #MODEL DOWNLOAD
 
resnet50_model = keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(224,224,3))


x = resnet50_model.output
x = layers.GlobalAveragePooling2D()(x)
preds = layers.Dense(6, activation='softmax', name = 'Output')(x) #final layer with softmax activation
model = keras.Model(inputs = resnet50_model.input, outputs = preds)

#freeze some layers and unfreez
for layer in model.layers[65:]:
    layer.trainable = False
for layer in model.layers[:-100]:
    layer.trainable = True
    
model.summary()

#COMPILE

INIT_LR = 0.0001

model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(lr=INIT_LR,decay=INIT_LR / 100), metrics=['accuracy',tf.keras.metrics.TruePositives(),
             tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.Recall(),
             tf.keras.metrics.FalsePositives(),
             tf.keras.metrics.FalseNegatives(),
             tf.keras.metrics.Precision()])
             
#TRAIN

History = model.fit_generator(train_generator, 
                                epochs=30,
                                verbose=1,
                                validation_data =val_generator)
                                
                                
                                
 #RESUME
 
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows = 4, ncols=1, figsize=(18,12))  
#ax1.subplot(211)
ax1.set_title('Loss')
ax1.plot(History.history['loss'], label='train')
ax1.plot(History.history['val_loss'], label='test')
ax1.legend()
# plot accuracy during training
#ax2.subplot(212)
ax2.set_title('precision')
ax2.plot(History.history['precision'], label='train')
ax2.plot(History.history['val_precision'], label='test')
ax2.legend()

ax3.set_title('recall')
ax3.plot(History.history['recall'], label='train')
ax3.plot(History.history['val_recall'], label='test')
ax3.legend()

ax4.set_title('accuracy')
ax4.plot(History.history['accuracy'], label='train')
ax4.plot(History.history['val_accuracy'], label='test')
