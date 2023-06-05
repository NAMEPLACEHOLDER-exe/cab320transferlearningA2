import os, os.path
import numpy as np
import matplotlib as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

'''
    AUXILARY FUNCS
'''

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

'''
    TASKS
    (these wont run properly yet, but can't test them on my laptop. will test and fix ASAP tomorrow morning when i have my PC)
'''

def task_1():
    # download flowe dataset from canvas
    # is this supposed to be coded?
    # do I really have to do a http request and all that just to download the zip?
    raise NotImplementedError()

def task_2():
    #download (excluding output layer) & freeze base model
    baseModel = MobileNetV2(weights='imagenet', include_top=False)
    baseModel.trainable = False

    return baseModel

def task_3():
    baseModel = task_2()

    #append new custom prediction (output) layer
    newLayers = baseModel.output
    newLayers = layers.GlobalAveragePooling2D()(newLayers)
    preditctionLayer = layers.Dense(5, activation="softmax", name="flower_predictions")(newLayers) 

    myModel = keras.Model(inputs=baseModel.input, outputs=preditctionLayer)

    return myModel

def task_4():
    """
        note: I really should change this to use tensorflow's image gen class,
        i just already did all this and don't know if i'll have time to change it.
        works tho :)
    """
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

    return (xTrain, yTrain, xValidation, yValidation, xTest, yTest)



def task_5():
    myModel = task_3()
    xTrain, yTrain, xValidation, yValidation, xTest, yTest = task_4()

    myModel.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.SGD(
            learning_rate=0.01,
            momentum=0.0,
            nesterov=False
        ),
        metrics=['accuracy']
    )

    history = myModel.fit(
        x=xTrain,
        y=yTrain,
        epochs=20,
        batch_size=1000,
        verbose=2,
        validation_data=(xValidation, yValidation)
    )

    return history

def task_6():
    history = task_5().history
    plotLossAndAccuracy(history["accuracy"], history["val_accuracy"], history["loss"], history["val_loss"])

def task_7():
    myModel = task_3()
    xTrain, yTrain, xValidation, yValidation, xTest, yTest = task_4()

    # essentially run task_5 3 times with dif learning rates, plot for each

def task_8():
    myModel = task_3()
    xTrain, yTrain, xValidation, yValidation, xTest, yTest = task_4()

    # same as task_7, but using best found learning rate and 3 dif momentums

def task_9():
    # yeah don't know if this one is gettin done
    raise NotImplementedError()

def task_10():
    # yeah don't know if this one is gettin done
    raise NotImplementedError()


# HYPOTHETICALLY, all we have to do for task 9/10 is
    # add extra feature detection layers between base model output and our custom 5 classification output layer (see tasl 3 as to how to create layers)
    # retrain


