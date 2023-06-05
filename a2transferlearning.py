import os, os.path
import numpy as np
import matplotlib as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

'''
    1. Importing MobileNetV2 and Modifying it
    (Task 1, Task 2, Task 3)
'''
#download (excluding output layer) & freeze base model
baseModel = MobileNetV2(weights='imagenet', include_top=False)
baseModel.trainable = False

#append new custom prediction (output) layer
newLayers = baseModel.output
newLayers = layers.GlobalAveragePooling2D()(newLayers)
preditctionLayer = layers.Dense(5, activation="softmax", name="flower_predictions")(newLayers) 

myModel = keras.Model(inputs=baseModel.input, outputs=preditctionLayer)


'''
    2. Preparing Data (could of used tensorflow image gen but just didn't realise)
    (Task 4)
'''
#set constants
DATASET_DIR = ".\small_flower_dataset"
ALL_FLOWER_TYPES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
SAMPLES_PER_TYPE = 200
TOTAL_SAMPLES = len(ALL_FLOWER_TYPES) * SAMPLES_PER_TYPE

#intialise master data arrs
xData = np.empty((TOTAL_SAMPLES, 224, 224, 3))
yData = np.zeros((TOTAL_SAMPLES, len(ALL_FLOWER_TYPES)))

#abstracted image processing
def prepare_image(imgPath):
    # MobileNetV2 accepts 224 x 224, so resize is VERY important
    img = keras.preprocessing.image.load_img(imgPath, target_size=(224, 224)) 
    imgArray = keras.preprocessing.image.img_to_array(img)
    return preprocess_input(imgArray)

#iterate over all samples, process and store
print("processing all sample images ...")
for i, flowerType in enumerate(ALL_FLOWER_TYPES):
    currentDir = os.path.join(DATASET_DIR, flowerType)
    print(f"processing images in {currentDir} ...")
    
    for j, imgName in enumerate(os.listdir(currentDir)):
        fullImgPath = os.path.join(currentDir, imgName)
        processedImage = prepare_image(fullImgPath)
        
        indexToInsert = j + (i * SAMPLES_PER_TYPE)
        xData[indexToInsert] = processedImage
        yData[indexToInsert][i] = 1

#split into train, validaition and test sets (70% train, 15% validation, 15% test)
print("splitting data into training, validation and test sets...")
TRAINING_SET_SIZE = int(TOTAL_SAMPLES * 0.7)
VALIDATION_SET_SIZE = int(TOTAL_SAMPLES * 0.15)
TEST_SET_SIZE = int(TOTAL_SAMPLES * 0.15)

xTrain = np.empty((TRAINING_SET_SIZE, 224, 224, 3))
yTrain = np.empty((TRAINING_SET_SIZE, len(ALL_FLOWER_TYPES)))

xValidation = np.empty((VALIDATION_SET_SIZE, 224, 224, 3))
yValidation = np.empty((VALIDATION_SET_SIZE, len(ALL_FLOWER_TYPES)))

xTest = np.empty((TEST_SET_SIZE, 224, 224, 3))
yTest = np.empty((TEST_SET_SIZE, len(ALL_FLOWER_TYPES)))

# THIS IS REALLY UGLY & ALL HARDCODED, I WILL FIX LATER!
# get 70% (140 samples) of each flower type for training
xTrain[0:140], xTrain[140:280], xTrain[280:420], xTrain[420:560], xTrain[560:700] = xData[0:140], xData[200:340], xData[400:540], xData[600:740], xData[800:940]
yTrain[0:140], yTrain[140:280], yTrain[280:420], yTrain[420:560], yTrain[560:700] = yData[0:140], yData[200:340], yData[400:540], yData[600:740], yData[800:940]  
# get 15% (30 samples not already in another set) of each flower type for validation
xValidation[0:30], xValidation[30:60], xValidation[60:90], xValidation[90:120], xValidation[120:150] = xData[140:170], xData[340:370], xData[540:570], xData[740:770], xData[940:970]
yValidation[0:30], yValidation[30:60], yValidation[60:90], yValidation[90:120], yValidation[120:150] = yData[140:170], yData[340:370], yData[540:570], yData[740:770], yData[940:970]
# get 15% (30 samples not already in another set) of each flower type for testing
xTest[0:30], xTest[30:60], xTest[60:90], xTest[90:120], xTest[120:150] = xData[170:200], xData[370:400], xData[570:600], xData[770:800], xData[970:1000]
yTest[0:30], yTest[30:60], yTest[60:90], yTest[90:120], yTest[120:150] = yData[170:200], yData[370:400], yData[570:600], yData[770:800], yData[970:1000]

"""
    3. Compiling/Training Our Model (Base w/ 5 node output layer)
    (Task 5)
"""

myModel.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.SGD(
        learning_rate=0.01,
        momentum=0.0,
        nesterov=False
    ),
    metrics=['accuracy']
)

myModel.fit(
    x=xTrain,
    y=yTrain,
    epochs=20,
    batch_size=1000,
    verbose=2,
    validation_data=(xValidation, yValidation)
)

"""
    4. Plotting loss (error) & accuracy over time
    (Task 6, also used heavily in other sections)
"""

def plotLossAndAccuracy(trainingLoss, trainingAccuracy, validationLoss, validationAccuracy):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(trainingAccuracy, label='Training Accuracy')
    plt.plot(validationAccuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(trainingLoss, label='Training Loss')
    plt.plot(validationLoss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    # once model classification layer working, get loss & accuracy data via model.history['loss']/model.history['accuracy']

"""
    5. Retraining with various learning rates/momentum
    (Task 7, Task 8)
"""

    # literally just section 3 but with
    #   learning_rate=x,
    #   momentum=x
    # for optimizer params

myModel.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.SGD(
        learning_rate=0.05,
        momentum=0.0,
        nesterov=False
    ),
    metrics=['accuracy']
)

# train here (myModel.fit())
# plot here (func in section 4)

# repeat for 2 more (total of 3) dif learning rates

myModel.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.SGD(
        learning_rate=0.05,
        momentum=0.1,
        nesterov=False
    ),
    metrics=['accuracy']
)

# train here (myModel.fit())
# plot here (func in section 4)

# repeat for 2 more (total of 3) dif momentums

"""
    6. Further modifications to baseModel and param tweaking
    (Task 9, Task 10)
"""

    # add extra feature detection layers between base model output and our custom 5 classification output layer (see section 1 as to how to create layers)
    # retrain



