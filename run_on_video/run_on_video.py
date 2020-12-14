import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import cv2 as cv
sys.path.append('../lstm_hand_prediction')
from lstm_return_predict import lstm_return_predict
sys.path.append('../hand_object_detector/hand_object_processing')
from iou import VideoVisualizer, _DetectedInstance, getCentroid

vidName = 'P95_110'

seriesList = np.load(vidName+'stab.npy', allow_pickle=True)
meta = np.load(vidName+'meta.npz', allow_pickle=True)

times = meta['times']
handsList = meta['handsList']
objList = meta['objList']
objIDs = meta['objIDs']

modelLSTM = tf.keras.models.load_model('../lstm_hand_prediction/lstm_model_best')

for filename in sorted(os.listdir(vidName)):
    showPlot = False
    predictedSeries = []
    print(filename)
    img = cv.imread(vidName+'/'+filename)
    frameNum = int(filename.split('.')[0][6:])
    
    #Get true past and future hand positions
    trueHand = []
    for i in range(5):
        if handsList[frameNum+i].size >0:
            handx, handy = getCentroid(handsList[frameNum+i,0].bbox)
            trueHand.append(np.array([handx, handy]))
    trueHand = np.array(trueHand)

    pastHand = []
    for i in range(min(5,frameNum)):
        if handsList[frameNum-i].size >0:
            handx, handy = getCentroid(handsList[frameNum-i,0].bbox)
            pastHand.append(np.array([handx, handy]))
    pastHand = np.array(pastHand)

    for i,time in enumerate(times):
        if time != frameNum - 5:
            continue
        showPlot = True
        pastSeries = seriesList[i][-13:-5]
        futureSeries = seriesList[i][-5:]
        predictedSeries = lstm_return_predict(pastSeries[:,:2], 5, modelLSTM)

        #Get object position of series
        for obj in objList[i]:
            if obj.objID != objIDs[i]:
                continue
            objx, objy = getCentroid(obj.bbox)
            seriesObj = obj
            print(obj.label)

        cv.rectangle(img,(seriesObj.bbox[0], seriesObj.bbox[1]),(seriesObj.bbox[2], seriesObj.bbox[3]), (0,255,0),2)
        plt.plot(objx-pastSeries[:,0], objy-pastSeries[:,1], label='Past Series')
        plt.plot(objx-futureSeries[:,0], objy-futureSeries[:,1], label='Future Series')
        plt.plot(objx-predictedSeries[:,0], objy-predictedSeries[:,1], label='Predicted Series')

    if len(predictedSeries) > 0:
        print("Has prediction")
        #MLP code goes here

    if showPlot:
        plt.imshow(img)
        #plt.plot(trueHand[:,0], trueHand[:,1], label="True Motion")
        #if pastHand.size>1:
            #plt.plot(pastHand[:,0], pastHand[:,1], label="Past Motion")
        plt.legend()
        plt.show()
