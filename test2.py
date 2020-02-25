import tensorflow as tf
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import csv
def model():
    inp = tf.keras.layers.Input((1200,))
    x = tf.keras.layers.Reshape((1200,1))(inp)
    x = tf.keras.layers.LSTM(1)(x)

    out = x
    model = tf.keras.Model(inp,out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),loss='hinge')
    model.summary()
    return model
def plot(x,y,predict):
    plt.clf()
    plt.plot(x)
    plt.plot(range(1200,1460),y)
    plt.scatter(1460,predict)
    plt.draw()
    plt.pause(0.01)     
model = model()
data = pd.read_csv('AAPL.csv', usecols=[4])
data = list(np.round(data.Close.values,2))
for z in range(2):
    memory=list()
    for i in range(365,len(data)-1200-260):
        plt.ion()
        inp = tf.expand_dims(data[i:i+1200]-data[i+1200], axis=0)
        predict_s = float(model.predict(inp))
        predict_p = predict_s*1+data[i+1200]
        target = np.expand_dims([(data[i+1460]-data[i+1200])/10], axis=0)
        
    
        
        loss = model.test_on_batch(inp, target)
        print('step {} target{} predict {} loss {}'.format(i,target,predict_s,loss))	
        memory.append([inp, target])
        if z > 0:
            plot(data[i:i+1200],data[i+1200:i+1460],predict_p)
       
    random.shuffle(memory)
    for x in range(len(memory)):
        loss = model.train_on_batch(memory[x][0], memory[x][1])
        print('loss {}'.format(loss))	
