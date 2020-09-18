from flask import Flask, jsonify, request, json, render_template, redirect, url_for, make_response
from werkzeug.utils import secure_filename
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
from datetime import datetime, timedelta
from flask_bcrypt import Bcrypt
from flask_cors import CORS
import os
import tensorflow as tf
import numpy as np # linear algebra
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
#from keras.applications.densenet import DenseNet121, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam
#from generator import DataGenerator
import keras
from keras.callbacks import ModelCheckpoint
#get_ipython().run_line_magic('matplotlib', 'inline')
from keras import backend as K
import cv2
import warnings
warnings.filterwarnings("ignore")
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

from flask_jwt_extended import (JWTManager, jwt_required, 
                                jwt_refresh_token_required, 
                                jwt_optional, fresh_jwt_required, 
                                get_raw_jwt, get_jwt_identity,
                                create_access_token, create_refresh_token, 
                                set_access_cookies, set_refresh_cookies, 
                                unset_jwt_cookies,unset_access_cookies)

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.basename('pred_img')
app.config['UPLOAD_FOLDER_DB'] = os.path.basename('static')
app.config['MONGO_DBNAME'] = 'major'
app.config['MONGO_URI'] = ''
app.config['JWT_SECRET_KEY'] = 'xyzabc'
app.config['JWT_TOKEN_LOCATION'] = ['cookies']
app.config['JWT_COOKIE_CSRF_PROTECT'] = False
app.config['JWT_CSRF_CHECK_FORM'] = False

mongo = PyMongo(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)
CORS(app)

def assign_access_refresh_tokens(user_id, url):
    exp = timedelta(days=365)
    access_token = create_access_token(identity=str(user_id), expires_delta=exp)
    refresh_token = create_refresh_token(identity=str(user_id), expires_delta=exp)
    resp = make_response(redirect(url, 302))
    set_access_cookies(resp, access_token)
    set_refresh_cookies(resp, refresh_token)
    return resp

def model():
    


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
    
    model = inception_v3()
    #print(model)
    model.load_weights("weights.best.hdf5")


    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
            'val',
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical',
            shuffle=False)


    class_labels = validation_generator.class_indices
    class_labels = {v: k for k, v in class_labels.items()}
    classes = list(class_labels.values())

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

    res = ['Choroidal NeoVascularization', 'Diabetic Macular Edema', 'DRUSEN', 'NORMAL']

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
            #print(res[np.argmax(predictions[0])])
        return res[np.argmax(predictions[0])]
    ss=get_pred()
    return ss

    
def unset_jwt():
    resp = make_response(redirect(url_for('index') ))
    unset_jwt_cookies(resp)
    return resp

@jwt.unauthorized_loader
def unauthorized_callback(callback):
    print(' No auth header')
    return redirect(url_for('index') )

@jwt.invalid_token_loader
def invalid_token_callback(callback):
    # Invalid Fresh/Non-Fresh Access token in auth header
    resp = make_response(redirect(url_for('index') ))
    unset_jwt_cookies(resp)
    return resp, 302

@jwt.expired_token_loader
def expired_token_callback(callback):
    # Expired auth header
    resp = make_response(redirect(url_for('refresh')))
    unset_access_cookies(resp)
    return resp, 302

@app.route('/token/refresh', methods=['GET'])
@jwt_refresh_token_required
def refresh():
    # Refreshing expired Access token
    user_id = get_jwt_identity()
    access_token = create_access_token(identity=str(user_id))
    resp = make_response(redirect(url_for('index')))
    set_access_cookies(resp, access_token)
    return resp

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/user')
@jwt_required
def user():
    userid = get_jwt_identity()
    userimg = mongo.db.userImg
    images = []
    for i in userimg.find({'user_id': userid}, limit=3):
        images.append(i['img'])
    return render_template('user.html', userid=userid, pred="", history=images)

@app.route('/user/<pred>')
@jwt_required
def userP(pred):
    userid = get_jwt_identity()
    userimg = mongo.db.userImg
    images = []
    for i in userimg.find({'user_id': userid}, limit=3):
        images.append(i['img'])
    return render_template('user.html', userid=userid, pred=pred, history=images)


@app.route('/upload', methods=['POST'])
@jwt_required
def upload_file():
    userid = get_jwt_identity()
    file = request.files['img']
    
    img_name = file.filename
    #print(dir(file))
    f = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
    #f2 = os.path.join(app.config['UPLOAD_FOLDER_DB'], img_name)

    file.save(f)
    #file.save(f2)

    UserImg = mongo.db.userImg
    K.clear_session()
    pred = model()
    K.clear_session()
    created = datetime.now()
    UserImg.insert({
        'user_id': userid,
        'img': img_name,
        'pred': pred,
        'created': created
    })
    #os.rename(f,f2)
    os.remove(f)
    return redirect(url_for('userP', pred=pred))

@app.route('/api/signup', methods=['POST'])
def signup():
    user_collection = mongo.db.users
    body = dict(request.form)
    userid = body['userid']
    email = body['email']
    password = body['password']

    password_bcrypt = bcrypt.generate_password_hash(password).decode('utf-8')
    
    user_collection.insert({
        'user_id': userid,
        'email': email,
        'password': password_bcrypt,
    })

    return assign_access_refresh_tokens(userid , url_for('user'))

@app.route('/api/login', methods=['POST'])
def login():
    user = mongo.db.users
    body = dict(request.form)
    userid = body['userid']
    password = body['password']
    response = user.find_one(userid)
    if response:
        if bcrypt.check_password_hash(response['password'], password):
            return assign_access_refresh_tokens(response['userid'] , url_for('user'))
        else:
            return jsonify({'error': 'invalid username and password'})
    else:
        return jsonify({'error': 'no result found'})

@app.route('/api/logout')
@jwt_required
def logout():
    # Revoke Fresh/Non-fresh Access and Refresh tokens
    return unset_jwt(), 302

if __name__ == "__main__":
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=0, port=5001)
