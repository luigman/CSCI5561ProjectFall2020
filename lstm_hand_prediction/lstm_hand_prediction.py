import tensorflow as tf
#Fixes "Fail to find the dnn implementation." error on my computer
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import os

# Needs tensorflow to run
# If you don't want to install tf you can run in google colab
LOAD = False

#To train lstm im creating fake data for now that is points along a quarter ellipse
#Meant to imitate initiale direction of hand but also potential curved movement
#Data in form (num_trajectories, num_datapoints_per_traj, xy_vals(2))
def create_fake_data(n_eq, n_p):
    num_equations = n_eq
    num_points = n_p
    h = 0
    fake_data = np.zeros((num_equations, int(num_points), 2))
    t_vals = np.zeros(int(num_points))
    for eq in range(0,num_equations):
        a = random.uniform(0.01,1)
        b = random.uniform(0.01,1)
        if random.random() >= 0.5:
            w = a
        else:
            w = -a

        if w > 0:
            #we want left side of ellipse
            #pi -> pi/2
            increment = (math.pi/2) / num_points
            for i in range(0,int(num_points)):
                t_vals[i] = math.pi - (i * increment)
        else:
            #we want right side of ellipse
            #0 -> pi/2
            increment = (math.pi/2) / num_points
            for i in range(0,int(num_points)):
                t_vals[i] = 0 + (i * increment)

        for t in range(0,int(num_points)):
            x = w + a * math.cos(t_vals[t])
            y = h + b * math.sin(t_vals[t])
            fake_data[eq,t,0] = x
            fake_data[eq,t,1] = y

    #Plot 100 lines from fake data
    fake_data = np.transpose(fake_data,(0,2,1))
    for i in range(0,100):
        plt.plot(fake_data[i][0],fake_data[i][1])
    plt.savefig('fake_hand_movement_example.png')

    fake_data = np.transpose(fake_data,(0,2,1))
    return fake_data

def loadData():
    parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    series_path = os.path.join(parent_dir, "hand_object_detector/maskrcnn_det/")
    series = []
    for root, directories, filenames in os.walk(series_path):
        #vidName = root.split('/')[-1]
        if 'P97' in root: #only load "nice" data
            for filename in filenames:
                if (filename.endswith("stab.npy")):
                    print("  "+filename)
                    seriesVid = np.load(os.path.join(root,filename), allow_pickle=True)
                    series.append(seriesVid)
    data = np.concatenate(series)
    print(np.shape(data))
    return data
    #return np.array(series).reshape(-1,20,2)

#center starting point at 0,0 and make furthest point within 1,1
#(427,20,2)
#return normalized data, old starting points, and max x,y to denormalize later
def normalize_data(data):
    start_point = np.full((len(data),2),-1.0)
    max = np.full((len(data),2),-1.0)
    for i in range(0,len(data)):
        start_point[i] = data[i][0]
        #center at 0
        for j in range(0,len(data[0])):
            data[i][j] = data[i][j] - start_point[i]
        max[i] = np.max(np.absolute(data[i]),0)
    new_max = np.max(max,0)
    for i in range(0,len(data)):
        data[i] = np.divide(data[i],new_max)
    return data

def sample(data, num_samples, num_timesteps, desired_predicted_label):
    samples_x = np.zeros((num_samples,num_timesteps,2))
    samples_y = np.zeros((num_samples,1,2))
    print(len(data))
    for i in range(0,num_samples):
        rand_data_set = random.randint(0,len(data)-1)
        rand_starting_timestep = random.randint(0,len(data[0])-(num_timesteps+desired_predicted_label))
        samples_x[i] = data[rand_data_set][rand_starting_timestep:rand_starting_timestep+num_timesteps]
        samples_y[i] = data[rand_data_set][rand_starting_timestep+num_timesteps+desired_predicted_label-1]
    return samples_x, samples_y


def visualize_one_step(samples, model):
    plt.clf()
    for i in range(0,len(samples)):
        sample = samples[i]
        sample = sample[0:8]
        sample = np.reshape(sample, (1,len(sample),2))
        prediction = model.predict(sample)
        one_step_compare = np.transpose(sample)

        plt.plot([prediction[0][0]], [prediction[0][1]], marker='o', markersize=3, color="red")
        plt.plot(one_step_compare[0],one_step_compare[1])
    plt.savefig('one_step.png')


def visualize_multi_step(samples, model):
    plt.clf()
    #number of test samples to be plotted
    for i in range(0,len(samples)):
        sample_full = samples[i]
        #update prediction along line with more real data
        for k in range(0,len(sample_full)-8,2):
            sample = sample_full[k:k+8]
            print("here")
            print(np.shape(sample))
            sample = np.reshape(sample, (1,len(sample),2))
            prediction = np.zeros((12,2))
            prediction[0] = model.predict(sample)
            #number of predictions in future
            for j in range(1,12-k):
                sample = np.delete(sample,0,1)
                sample = np.concatenate((sample,np.reshape(prediction[j-1],(1,1,2))),1)
                prediction[j] = model.predict(sample)
            prediction = np.transpose(prediction)
            compare = np.transpose(samples[i])
            print(prediction[0][:12-k])
            plt.plot(prediction[0][:12-k],prediction[1][:12-k], '--')
            plt.plot(compare[0],compare[1])
    plt.savefig('multi_step.png')


def lstm_hand_prediction():
    #data = create_fake_data(2000, 20)
    data = loadData()
    norm_data = normalize_data(data)
    np.random.shuffle(norm_data)

    #Right now the LSTM is trained to predict one step ahead based off of previous 8 timesteps
    TT_SPLIT = norm_data.shape[0]//5*4 #80-20 split
    train_x, test_x = np.split(norm_data, [TT_SPLIT], 0)
    train_sample_x, train_sample_y = sample(train_x, 2000, 8, 1)
    test_sample_x, test_sample_y = sample(test_x, 200, 8, 1)

    #Create model
    if LOAD == False:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(5, input_shape=(len(train_sample_x[0]), 2), return_sequences=True))
        #model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(4, return_sequences=False))
        #model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(2))


        model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(0.001), metrics = ['accuracy'])

        print(model.summary())

        model.fit(train_sample_x, train_sample_y, epochs=30, batch_size = 1, validation_split = 0.1)
        results = model.evaluate(test_sample_x, test_sample_y)

        model.save('lstm_model')
    else:
        model = tf.keras.models.load_model('lstm_model')



    # VISUALIZE RESULTS
    visualize_one_step(test_x[0:1], model)
    visualize_multi_step(test_x[0:1], model)


if __name__ == '__main__':
    lstm_hand_prediction()
