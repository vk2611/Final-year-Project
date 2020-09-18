import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import matplotlib.gridspec as gridspec
import zlib
import itertools
import sklearn
import itertools
import scipy
from skimage.transform import resize
import csv
from tqdm import tqdm
from sklearn import model_selection
from sklearn.model_selection import train_test_split, learning_curve,KFold,cross_val_score,StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import keras
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import class_weight
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta, RMSprop
from keras.models import Sequential, model_from_json
from keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D,MaxPooling2D,AveragePooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
from keras.models import Model
import matplotlib.pyplot as plt
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam
#from generator import DataGenerator
import keras
from keras.callbacks import ModelCheckpoint
#get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import warnings
warnings.filterwarnings("ignore")


# In[2]:


IMG_SIZE = 224
batch_size = 32
img_in = Input((224,224,3))
t_x = (224,224,3)


# In[3]:


def inception_v3(): 
    model =  keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=img_in, input_shape=t_x, pooling='avg')
    x = model.output
    predictions = Dense(4, activation="softmax", name="predictions")(x)
    model = Model(inputs=img_in, outputs=predictions)
    return model


# In[4]:


model = inception_v3()
print(model)
model.load_weights("weights.best.hdf5")


# In[6]:


# model.summary()


# In[52]:


import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.layers.advanced_activations import ELU
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import cv2
from os import listdir
from os.path import isfile, join
import re
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils.vis_utils import plot_model
import matplotlib.image as mpimg


# In[54]:


validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
        'C:\Python\Major\val',
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        shuffle=False)


# In[55]:


class_labels = validation_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())


# In[64]:


def draw_test(name, pred, im, true_label):
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 160, 0, 0, 300 ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, "predited - "+ pred, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    cv2.putText(expanded_image, "true - "+ true_label, (20, 120) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)
    cv2.imshow(name, expanded_image)
    

def getRandomImage(path, img_width, img_height):
    """function loads a random images from a random folder in our test path """
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    # random_directory = np.random.randint(0,len(folders))
    # path_class = folders[random_directory]
    # file_path = path + path_class
    file_path = path
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    final_path = file_path + "/" + image_name
    return image.load_img(final_path, target_size = (img_width, img_height)), final_path, 'path_class'

# dimensions of our images
img_width, img_height = 224, 224

files = []
predictions = []
true_labels = []

res = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

def get_pred():
    # predicting images
    for _ in range(0, 1):
        path = './pred_img/'
        img, final_path, _ = getRandomImage(path, img_width, img_height)
        files.append(final_path)
        # true_labels.append(true_label)
        x = image.img_to_array(img)
        x = x * 1./255
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size = 1)
        predictions.append(classes)
        print(res[np.argmax(predictions[0])])
    return res[np.argmax(predictions[0])]

#print(get_pred())
