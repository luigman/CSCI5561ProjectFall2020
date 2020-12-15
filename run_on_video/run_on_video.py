import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import cv2 as cv
sys.path.append('../lstm_hand_prediction')
from lstm_return_predict import lstm_return_predict
sys.path.append('../hand_object_detector/hand_object_processing')
from iou import VideoVisualizer, _DetectedInstance, getCentroid

vidName = 'P95_102'

seriesList = np.load(vidName+'stab.npy', allow_pickle=True)
meta = np.load(vidName+'meta.npz', allow_pickle=True)

times = meta['times']
handsList = meta['handsList']
objList = meta['objList']
objIDs = meta['objIDs']
objBBsPrev = meta['objBBsPrev']

modelLSTM = tf.keras.models.load_model('../lstm_hand_prediction/lstm_model_best',compile=False)
modelMLP = tf.keras.models.load_model('../mlp_contact_prediction/mlp_model',compile=False)

errors = []
for filename in sorted(os.listdir(vidName)):
    if False: #Set True to only process video
        break
    fig = plt.figure()
    showPlot = False
    predictedSeries = []
    centroids_location = [] 
    print(filename)
    frameNum = int(filename.split('.')[0][6:])
    if frameNum < 6:
        continue
    img = cv.imread(vidName+'/frame_'+str(frameNum-5).zfill(10)+'.jpg')
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    #print("Displaying:", vidName+'/frame_'+str(frameNum-5).zfill(10)+'.jpg')
    
    #Get true past and future hand positions
    #trueHand = []
    #for i in range(5):
    #    if len(handsList[frameNum+i]) >0:
    #        handx, handy = getCentroid(handsList[frameNum+i][0].bbox)
    #        trueHand.append(np.array([handx, handy]))
    #trueHand = np.array(trueHand)

    pastHand = []
    for i in range(min(5,frameNum)):
        if len(handsList[frameNum-1-i]) >0:
            handx, handy = getCentroid(handsList[frameNum-1-i][0].bbox)
            pastHand.append(np.array([handx, handy]))
    pastHand = np.array(pastHand)

    for i,time in enumerate(times):
        if time != frameNum - 5:
            continue
        showPlot = True
        pastSeries = np.flipud(seriesList[i][5:13]) #flip because LSTM is predicting backwards
        futureSeries = seriesList[i][:5]
        predictedSeries.append(lstm_return_predict(pastSeries[:,:2], 5, modelLSTM))
        frameLabel = 1 in futureSeries[:,2]

        objx, objy = getCentroid(objBBsPrev[i])
        centroids_location.append((objBBsPrev[i]))  # make sure to get the probabilities in the same order as the boxes appear 

        cv.rectangle(img,(objBBsPrev[i][0], objBBsPrev[i][1]),(objBBsPrev[i][2], objBBsPrev[i][3]), (0,255,0),2)
    

    contact_prob = []
    if len(predictedSeries) > 0:
        predictedSeries = np.array(predictedSeries)
        print("Has prediction")
        #MLP code goes here
        def rowNorm(X):
            return np.sum(np.abs(X)**2,axis=-1)**(1./2)
        for series in predictedSeries:
            x1 = rowNorm(series)
            m = 1106.50695399817    # Max value of training data after taking norm. Need to divide by this to get accurate results from the model
            x1 = (x1 / m).reshape((-1,5))
            y_pred = modelMLP.predict(x1)
            contact_prob.append(y_pred[0,1])

            frameLabelPred = contact_prob[-1] > 0.5
            errors.append(frameLabel == frameLabelPred)
        centroids_location = np.array(centroids_location) 
        for i in range(centroids_location.shape[0]): 
            plt.text(centroids_location[i][0],centroids_location[i][1],"P: " + f'{contact_prob[i]:.2f}',fontsize=10) 


        plt.plot(objx-pastSeries[:,0], objy-pastSeries[:,1], label='Past Series')
        plt.plot(objx-futureSeries[:,0], objy-futureSeries[:,1], label='Future Series')
        plt.plot(objx-predictedSeries[-1][:,0], objy-predictedSeries[-1][:,1], label='Predicted Series')
        #plt.plot(trueHand[:,0], trueHand[:,1], label="True Motion")
        #if pastHand.size>1:
            #plt.plot(pastHand[:,0], pastHand[:,1], label="Past Motion")
        plt.legend(loc="upper left")
    plt.imshow(img)
    plt.savefig("saved_imgs/" + filename) 
    #if showPlot:
    #    plt.show()

print("Accuracy:", np.sum(errors)/len(errors))

image_folder = 'saved_imgs/'
video_name = 'video.mp4'

images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".jpg")]
frame = cv.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv.VideoWriter(video_name, 0x7634706d, 30, (width,height))

for image in images:
    video.write(cv.imread(os.path.join(image_folder, image)))

cv.destroyAllWindows()
video.release()