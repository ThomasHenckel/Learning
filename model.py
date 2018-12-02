import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Flatten,Dense,MaxPooling2D,Conv2D,Lambda,Cropping2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, History
import tensorflow as tf
print("Tensorflow version: "+tf.__version__)
from sklearn.model_selection import train_test_split
import json
from PIL import Image

import os
import argparse

# tensorflow setup functions, was neted to run on GPU on my machine, 
# and allowed me to run both the model training and autonomous driving at the same time
def initialize_session():
    clear_session()
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras
    

def clear_session():
    """Destroys the current TF graph and creates a new one.
    Useful to avoid clutter from old models / layers.
    """
    global _SESSION
    global _GRAPH_LEARNING_PHASES
    tf.reset_default_graph()
    reset_uids()
    _SESSION = None
    phase = tf.placeholder_with_default(False,
                                        shape=(),
                                        name='keras_learning_phase')
    _GRAPH_LEARNING_PHASES = {}
    _GRAPH_LEARNING_PHASES[tf.get_default_graph()] = phase
    
def reset_uids():
    """Resets graph identifiers.
    """
    global _GRAPH_UID_DICTS
    _GRAPH_UID_DICTS = {}

initialize_session()

def load_driving_log(path):
    # Load path to training images, and the steering angel
    driving_log = pd.read_csv(path, names = ["center_image", "left_image", "right_image", "steering_angle","throttle","brake","speed"])
    # strip away the absolut path from the images, keeping only the filename
    driving_log['center_image'] = driving_log['center_image'].apply(lambda x: x.split('\\')[-1])
    driving_log['left_image'] = driving_log['left_image'].apply(lambda x: x.split('\\')[-1])
    driving_log['right_image'] = driving_log['right_image'].apply(lambda x: x.split('\\')[-1])
    return driving_log


def flip_images(from_path, to_path):
    # flip and save all images on disk
    # flow_from_dataframe can only load the images from thisk, 
    # and by saving them we only need to flip the images one time
    for filename in os.listdir(from_path):
        if(filename[:4] != 'flip'):
            if(os.path.isfile(to_path+'/flip_'+filename) == False):
                imageSource=from_path+'/'+filename
                image_obj = Image.open(imageSource)
                rotated_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
                rotated_image.save(to_path+'/flip_'+filename)

def use_flip_images(driving_log):
    # add the flipped images to the driving_log dataframe
    driving_log_flip = driving_log.copy()
    driving_log_flip['center_image'] = driving_log_flip['center_image'].apply(lambda x: 'flip_'+ x)
    driving_log_flip['left_image'] = driving_log_flip['left_image'].apply(lambda x: 'flip_'+ x)
    driving_log_flip['right_image'] = driving_log_flip['right_image'].apply(lambda x: 'flip_'+ x)
    driving_log_flip['steering_angle'] = driving_log_flip['steering_angle'].apply(lambda x: -x)
    return driving_log_flip

def use_side_camera(driving_log):
    # use the side camera images, 
    # add an steering angel offset, 
    # to steer the car towards the center of the road
    driving_log_left = driving_log.copy()
    driving_log_left['center_image'] = driving_log_left['left_image']
    driving_log_left['steering_angle'] = driving_log_left['steering_angle'].apply(lambda x: x+0.15)

    driving_log_right = driving_log.copy()
    driving_log_right['center_image'] = driving_log_right['right_image']
    driving_log_right['steering_angle'] = driving_log_right['steering_angle'].apply(lambda x: x-0.15)

    return driving_log_right.append(driving_log_left,ignore_index=True)

def train_model_dataframe(X_train, X_test, y_train, y_test,location,batch_size,model_path,epoch,model = None):
  
    # Train or retrain a regression model, 
    # The model architecture is from Nvidias self driving car article
    # https://devblogs.nvidia.com/deep-learning-self-driving-cars/ 
    img_height, img_width = 160,320

    if model:
        print("retraining model")
    else:
        model = Sequential()
        model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
        model.add(Lambda(lambda x: (x / 255.0) - 0.5))
        model.add(Conv2D(24,(5,5),strides=(2,2), activation="relu"))
        model.add(Conv2D(36,(5,5),strides=(2,2), activation="relu"))
        model.add(Conv2D(48,(5,5),strides=(2,2), activation="relu"))
        model.add(Conv2D(64,(3,3), activation="relu"))
        model.add(Conv2D(64,(3,3), activation="relu"))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))
        model.compile(loss='mse',optimizer='adam')
     
    # flow_from_dataframe needs the indexes to be in order
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    train_datagen = ImageDataGenerator()
    
    test_datagen = ImageDataGenerator()

    train_generator= train_datagen.flow_from_dataframe(dataframe=X_train,directory=location,x_col="center_image",y_col='steering_angle',has_ext=True, 
                                                     target_size=(img_height, img_width),
                                                     batch_size=batch_size,
                                                     class_mode='other')

    validation_generator = test_datagen.flow_from_dataframe(dataframe=X_test,directory=location,x_col="center_image",y_col='steering_angle',has_ext=True, 
                                                     target_size=(img_height, img_width),
                                                     batch_size=batch_size,
                                                     class_mode='other')
    
    # only save the model if the validation loss decreases
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', 
                                 verbose=1, save_best_only=True, 
                                 save_weights_only=False, mode='auto')
    
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
    
    # train the model
    fit = model.fit_generator(train_generator,
                              steps_per_epoch=STEP_SIZE_TRAIN,
                              epochs=epoch,
                              validation_data=validation_generator,
                              validation_steps=STEP_SIZE_VALID,
                              verbose=1,
                              callbacks=[checkpoint]
                              #,use_multiprocessing=True, workers=16 # if you like to use multiple cores
                              )

    # Save the training history  
    with open('history.txt', 'w') as f:
        json.dump(fit.history, f)
    
    return model, fit.history

def parse_args():
    """Parse input arguments.
    Parameters
    -------------------
    No parameters.
    Returns
    -------------------
    args: argparser.Namespace class object
        An argparse.Namespace class object contains parameters.
    """
    parser = argparse.ArgumentParser(description='Train a CNN to predict steering angels')
    parser.add_argument('--i', dest='input',
                        help='Path to driving data',
                        default='../driving_data/', type=str)
    parser.add_argument('--m', dest='modelPath',
                        help='Continue training model',
                        default="model.h5", type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    print ('Called with args:')
    print (args)

    location = args.input+'IMG/'
    batch_size = 16
    model_path = 'model.h5'
    epoch = 5

    driving_log = load_driving_log(args.input+"driving_log.csv")
    flip_images(location, location)
    driving_log = driving_log.append(use_side_camera(driving_log),ignore_index=True)
    driving_log = driving_log.append(use_flip_images(driving_log),ignore_index=True)

    X_train, X_test, y_train, y_test = train_test_split(driving_log[['center_image','steering_angle']], driving_log['steering_angle'], test_size=0.20, random_state=42)
    
    model = None
    if (args.modelPath):
        model = load_model('model.h5')

    train_model_dataframe(X_train, X_test, y_train, y_test,location,batch_size,model_path,epoch,model)

