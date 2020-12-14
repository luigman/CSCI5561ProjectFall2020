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

for filename in sorted(os.listdir(vidName)):
    print(filename)
    img = cv.imread(vidName+'/'+filename)
    frameNum = int(filename.split('.')[0][6:])
    
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

    plt.imshow(img)
    plt.plot(trueHand[:,0], trueHand[:,1], label="True Motion")
    if pastHand.size>1:
        plt.plot(pastHand[:,0], pastHand[:,1], label="Past Motion")
    plt.legend()
    plt.show()

    for i,time in enumerate(times):
        if time != frameNum:
            continue
        