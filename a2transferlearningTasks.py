import os, os.path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input

'''
    AUXILARY FUNCS
'''
def compileAndTrainModel(model, customOptimizer, numEpochs, datasetTuple):
    """
        Compiles and trains the supplied model using the supplied optimizer for a specified number of epochs.
        Uses the provided dataset (passed in as tuple(trainingData, trainingLabels, validationData, validationLabels)).
    """
    
    xTrain, yTrain, xValidation, yValidation = datasetTuple

    #compilation
    model.compile(
        loss='categorical_crossentropy',
        optimizer=customOptimizer,
        metrics=['accuracy']
    )

    #training
    history = model.fit(
        x=xTrain,
        y=yTrain,
        epochs=numEpochs,
        verbose=2,
        validation_data=(xValidation, yValidation)
    )

    return history.history # .history needed cause history returned by .fit is actually obj with history property


def plotLossAndAccuracy(trainingLoss, trainingAccuracy, validationLoss, validationAccuracy, customTitles = [False, False]):
    """
        Plots the provided training & validation loss/accuracy data in a single figure with 2 graphs.
        Optional custom titles are accepted (formmated as ["custom title for accuracy graph", "custom title for loss graph"]).
    """
    #intialise custom x axis
    customXTicks = list(range(len(trainingLoss)))
    customXLabels = list(range(1, len(trainingLoss)+1))
    
    
    plt.figure(figsize=(8, 8)) # plot container

    # create accuracy graph
    plt.subplot(2, 1, 1)
    plt.plot(trainingAccuracy, label='Training Accuracy')
    plt.plot(validationAccuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    if customTitles[0] != False: plt.title(customTitles[0])
    else: plt.title('Training and Validation Accuracy')
    plt.xlabel('epoch')
    plt.xticks(customXTicks, customXLabels)

    #create loss graph
    plt.subplot(2, 1, 2)
    plt.plot(trainingLoss, label='Training Loss')
    plt.plot(validationLoss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,max(max(trainingLoss), max(validationLoss))])
    if customTitles[1] != False: plt.title(customTitles[1])
    else: plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.xticks(customXTicks, customXLabels)

    plt.subplots_adjust(hspace=0.4)
    plt.show()

'''
    TASKS
'''

def my_team():
    """
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    """

    return [ (10874917, 'Tali', 'Dugandzic'), (11292938, 'Sam', 'Price'), (11205342, 'Mohammed Mutahher', 'Mohammed Naseer') ]

def task_1():
    """
        Unsure of what this task is supposed to be.
        Flower dataset is downloaded from canvas by user?
        IMPORTANT: FLOWER DATA SET MUST BE IN THE SAME DIRECTORY AS THIS FILE AND CALLED 'small_flower_dataset'
    """
    pass

def task_2():
    """
      Download and intialise pretrained MobileNetV2 network. (tf.keras.applications already imported)
    """
    #download (excluding output layer) & freeze base model
    baseModel = MobileNetV2(weights='imagenet', include_top=False)
    baseModel.trainable = False

    return baseModel

def task_3():
    """
        Append appropriate output layers for our flower problem to pretrained model.
    """
    baseModel = task_2() # grab MobileNetV2

    #create new custom prediction (output) layers
    newLayers = baseModel.output
    newLayers = layers.GlobalAveragePooling2D()(newLayers)
    preditctionLayer = layers.Dense(5, activation="softmax", name="flower_predictions")(newLayers) 

    #append new layers to pretrained model
    myModel = keras.Model(inputs=baseModel.input, outputs=preditctionLayer)

    return myModel

def task_4():
    """
        Preprocess all data appropriately for model input. Split into training, validation & test sets (70/15/15)

        note: I really should change this to use tensorflow's image gen class, I just already did all this and can't bring myself to delete it.
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

    # THIS IS REALLY UGLY & ALL HARDCODED, I APOLOGISE TO WHOEVER HAS TO LOOK AT THIS
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
    """
        Compile and train the new model with base SGD optimizer (LR = 0.01, Momentum = 0)
    """
    myModel = task_3() #grab model from task 3
    xTrain, yTrain, xValidation, yValidation, xTest, yTest = task_4() #grab dataset from task 4

    #create basic SGD optimizer
    basicSGD = keras.optimizers.SGD(
            learning_rate=0.01,
            momentum=0.0,
            nesterov=False
        )

    #compile and train
    history = compileAndTrainModel(
            model=myModel, 
            customOptimizer=basicSGD, 
            numEpochs=20,
            datasetTuple=(xTrain, yTrain, xValidation, yValidation)
        )

    return history #return training info

def task_6():
    """
        Plot accuracy and loss (errors) for both training and validation from training performed in task 5.
    """
    history = task_5() #grab training info from task 5
    plotLossAndAccuracy(history["loss"], history["accuracy"], history["val_loss"], history["val_accuracy"])

def task_7():
    """
        Compile and train the model using 3 SGD optimizers of different orders of magnitude for learning rate.
        Plot results after each training session. 
    """
    myModel = task_3() #grab model from task 3
    xTrain, yTrain, xValidation, yValidation, xTest, yTest = task_4() #grab dataset from task 4

    #choose 3 different orders of magnitude for learning rate to experiment with
    learningRates = [0.0001, 0.001, 1]

    #compile, train and plot for each LR
    for LR in learningRates:
        print(f"compiling/training model with learning rate '{LR}'...")

        optimizer = keras.optimizers.SGD(
            learning_rate=LR,
            momentum=0.0,
            nesterov=False
        )

        history = compileAndTrainModel(
            model=myModel, 
            customOptimizer=optimizer, 
            numEpochs=20,
            datasetTuple=(xTrain, yTrain, xValidation, yValidation)
        )

        print("training done. plotting loss and accuracy...")

        plotLossAndAccuracy(
            trainingAccuracy= history["accuracy"], 
            validationAccuracy= history["val_accuracy"], 
            trainingLoss= history["loss"], 
            validationLoss= history["val_loss"],
            customTitles= [
                f"Training and Validation Accuracy (LR = {LR})",
                f"Training and Validation Loss (LR = {LR})"
            ])
    
def task_8():
    """
        Compile and train the model using 3 SGD optimizers of different momentums.
        Plot results after each training session.
    """
    myModel = task_3() #grab model from task 3
    xTrain, yTrain, xValidation, yValidation, xTest, yTest = task_4() #grab dataset from task 4

    LR = 0.01 #best found learning rate
    momentums = [0.4, 0.6, 0.9] #3 different momentums to experiment with

    #compile, train and plot for each momentum
    for mom in momentums:
        optimizer = keras.optimizers.SGD(
            learning_rate=LR,
            momentum=mom,
            nesterov=False
        )

        history = compileAndTrainModel(
            model=myModel, 
            customOptimizer=optimizer, 
            numEpochs=20,
            datasetTuple=(xTrain, yTrain, xValidation, yValidation)
        )

        plotLossAndAccuracy(
            trainingAccuracy= history["accuracy"], 
            validationAccuracy= history["val_accuracy"], 
            trainingLoss= history["loss"], 
            validationLoss= history["val_loss"],
            customTitles= [
                f"Training and Validation Accuracy (LR = {LR}, Momentum = {mom})",
                f"Training and Validation Loss (LR = {LR}, Momentum = {mom})"
            ])

def task_9():
    
    raise NotImplementedError()

def task_10():
    
    raise NotImplementedError()


if __name__ == "__main__":
    # my_team()
    # task_1()
    # task_2()
    # task_3()
    # task_4()
    # task_5()
    # task_6()
    # task_7()
    # task_8()
    # task_9()
    # task_10()
    pass


