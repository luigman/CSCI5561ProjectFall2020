import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import os
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
import sys
sys.path.append('../lstm_hand_prediction')
from lstm_return_predict import lstm_return_predict

#Computes the norm of each row of a matrix
def rowNorm(X):
    return np.sum(np.abs(X)**2,axis=-1)**(1./2)

vids = [np.load('../hand_object_detector/maskrcnn_det/P96/rgb_frames/P96_101stab.npy', allow_pickle=True),
        np.load('../hand_object_detector/maskrcnn_det/P96/rgb_frames/P96_104stab.npy', allow_pickle=True),
        np.load('../hand_object_detector/maskrcnn_det/P96/rgb_frames/P96_105stab.npy', allow_pickle=True),
        np.load('../hand_object_detector/maskrcnn_det/P96/rgb_frames/P96_106stab.npy', allow_pickle=True),
        np.load('../hand_object_detector/maskrcnn_det/P96/rgb_frames/P96_107stab.npy', allow_pickle=True),
        np.load('../hand_object_detector/maskrcnn_det/P96/rgb_frames/P96_108stab.npy', allow_pickle=True),
        np.load('../hand_object_detector/maskrcnn_det/P96/rgb_frames/P96_109stab.npy', allow_pickle=True),
        np.load('../hand_object_detector/maskrcnn_det/P96/rgb_frames/P96_112stab.npy', allow_pickle=True),
        np.load('../hand_object_detector/maskrcnn_det/P96/rgb_frames/P96_114stab.npy', allow_pickle=True),
        np.load('../hand_object_detector/maskrcnn_det/P96/rgb_frames/P96_116stab.npy', allow_pickle=True),
        np.load('../hand_object_detector/maskrcnn_det/P96/rgb_frames/P96_117stab.npy', allow_pickle=True),
        np.load('../hand_object_detector/maskrcnn_det/P95/rgb_frames/P95_101stab.npy', allow_pickle=True),
        np.load('../hand_object_detector/maskrcnn_det/P95/rgb_frames/P95_102stab.npy', allow_pickle=True),
        np.load('../hand_object_detector/maskrcnn_det/P95/rgb_frames/P95_103stab.npy', allow_pickle=True),
        np.load('../hand_object_detector/maskrcnn_det/P95/rgb_frames/P95_104stab.npy', allow_pickle=True),
        np.load('../hand_object_detector/maskrcnn_det/P95/rgb_frames/P95_105stab.npy', allow_pickle=True),
        np.load('../hand_object_detector/maskrcnn_det/P95/rgb_frames/P95_106stab.npy', allow_pickle=True),
        np.load('../hand_object_detector/maskrcnn_det/P95/rgb_frames/P95_107stab.npy', allow_pickle=True),
        np.load('../hand_object_detector/maskrcnn_det/P95/rgb_frames/P95_108stab.npy', allow_pickle=True),
        np.load('../hand_object_detector/maskrcnn_det/P95/rgb_frames/P95_109stab.npy', allow_pickle=True),
        #np.load('../hand_object_detector/maskrcnn_det/P95/rgb_frames/P95_110stab.npy', allow_pickle=True),
        np.load('../hand_object_detector/maskrcnn_det/P95/rgb_frames/P95_111stab.npy', allow_pickle=True)]



vids = np.array(vids)

data = []
for vid in vids:
    for series in vid:
        for i in range(len(series) - 5):
            mini_series = []
            for j in range(5):
                mini_series.append(series[i+j])
            data.append(mini_series)


data = np.array(data)
print("Loaded Training data")

data.shape


np.random.shuffle(data)


x = []
y = []
for d in data:
    row = []
    labels = []
    for i in range(5):
        row.append(d[i][:2])
        labels.append(d[i,2])
    x.append(row)
    y.append(int(1 in labels))
x = np.array(x)
y = np.array(y)
x_data = rowNorm(x)


m = max(x_data.flatten())

x_data2 = x_data/m


n = int(len(x_data2)*0.8)


x_train, x_test = np.split(x_data2, [n])
y_train, y_test = np.split(y, [n])
y_train=to_categorical(y_train,2)
y_test=to_categorical(y_test,2)



model = Sequential()
model.add(Dense(10, input_dim=5, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(2, activation='softmax'))


model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100)
model.save('mlp_model') 


test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(test_loss)
print(test_accuracy)


test_vids = [np.load('../hand_object_detector/maskrcnn_det/P95/rgb_frames/P95_110stab.npy', allow_pickle=True)]
    #np.load('../hand_object_detector/maskrcnn_det/P99/rgb_frames/P99_107stab.npy', allow_pickle=True)]
'''
x1 = []
y1 = []
for i in range(len(new_test[0]) - 5):
    point = []
    for j in range(5):
        point.append(np.linalg.norm(new_test[0][i+j][:2]))
    x1.append(point)
    y1.append(new_test[0][i+j][2])
x1 = np.array(x1)
y1 = np.array(y1)
'''

x1 = []
lstm_predictions = []
true_positions = []
contact = []
y1 = []
modelLSTM = tf.keras.models.load_model('../lstm_hand_prediction/lstm_model_best',compile=False)
for vid in test_vids:
    for series in vid:
        for i in range(len(series) - 13):
            mini_series = []
            labels = []
            for j in range(8):
                mini_series.append(series[i+j,:2])
                labels.append(series[i+j,2])
            
            prediction = lstm_return_predict(mini_series, 5, modelLSTM)
            mini_series = np.array(mini_series)
            if False:
                plt.plot(mini_series[:,0], mini_series[:,1], label='Previous')
                plt.plot(prediction[:,0], prediction[:,1], label='Predicted')
                plt.plot(series[i+8:i+13,0], series[i+8:i+13,1], label='True')
                plt.legend()
                plt.show()
            x1.append(rowNorm(prediction))
            lstm_predictions.append(prediction)
            true_positions.append(series[i+8:i+13,:2])
            contact.append(series[i+8:i+13,2])
            y1.append(int(1 in labels))
x1 = np.array(x1)
print(x1.shape)
y1 = np.array(y1)
print(y1.shape)
lstm_predictions = np.array(lstm_predictions)
print(lstm_predictions.shape)
print(np.array(contact).shape)

xtest1 = x1/m



m

y_pred = model.predict(xtest1)
print("y_pred:", y_pred.shape)
probs = y_pred[:,1]

print(y1)
np.set_printoptions(suppress=True) #Clean up probabilities
print(np.round(probs, 4))


lstmx = lstm_predictions[0][:5, 0]
lstmy = lstm_predictions[0][:5, 1]
realx = true_positions[0][:5, 0]
realy = true_positions[0][:5, 1]
contactx = []
contacty = []
for i in range(len(contact[0])):
    if contact[0][i] == 1:
        contactx.append(realx[i])
        contacty.append(realy[i])
    
plt.figure()
plt.title('True and Predicted Contact')
plt.plot(lstmx, lstmy, label='Predicted')
#plt.scatter(predx, predy, label = 'Predicted Contact')
plt.plot(realx, realy, label='True')
plt.scatter(contactx, contacty, label='True Contact')
plt.legend()
plt.show()