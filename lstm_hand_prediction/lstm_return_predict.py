import tensorflow as tf
#Fixes "Fail to find the dnn implementation." error on my computer
physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import os


def normalize(data_points, max):
    start_point = data_points[0]
    norm_points = np.subtract(data_points, start_point)
    norm_points = np.divide(norm_points,max)
    return norm_points, start_point

def undo_normalize(norm_points, start_point, max):
    norm_points = np.transpose(norm_points)
    data_points = np.zeros(np.shape(norm_points))
    for i in range(0,len(norm_points)):
        data_points[i][0] = norm_points[i][0] * max[0]
        data_points[i][1] = norm_points[i][1] * max[1]
    #data_points = np.transpose(data_points)
    data_points = np.add(data_points, start_point)
    return data_points

#takes in previous 8 points to calculate prediction
def lstm_return_predict(prev_8, num_steps_pred):
    assert len(prev_8) == 8
    #max values from training
    max = [853.01153564, 707.01364136]
    norm_data, start_point = normalize(prev_8, max)

    #Right now the LSTM is trained to predict one step ahead based off of previous 8 timesteps
    #If you want this changed I can try training off of more or fewer
    model = tf.keras.models.load_model('lstm_model_best')

    prev_8_new = np.reshape(norm_data, (1,len(norm_data),2))
    prediction = np.zeros((num_steps_pred,2))
    prediction[0] = model.predict(prev_8_new)
    for j in range(1,num_steps_pred):
        prev_8_new = np.delete(prev_8_new,0,1)
        prev_8_new = np.concatenate((prev_8_new,np.reshape(prediction[j-1],(1,1,2))),1)
        prediction[j] = model.predict(prev_8_new)
    prediction = np.transpose(prediction)

    final_pred = undo_normalize(prediction, start_point, max)
    print(final_pred)

    plot_final = np.transpose(final_pred)
    plot_prev = np.transpose(prev_8)
    plt.clf()
    plt.plot(plot_final[0],plot_final[1], '--')
    plt.plot(plot_prev[0],plot_prev[1])
    plt.savefig('prediction.png')
    #plt.savefig('prediction{0}.png'.format(number))
    return final_pred
