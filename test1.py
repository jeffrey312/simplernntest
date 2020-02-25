import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

def var(inp):
    print(inp)
    x = np.arange(0,100000*np.pi,inp)    
    y = tf.math.sin(x)
    return y
    
def plot(inp_step,pred):
    plt.clf()
    plt.plot(inp_step)
    plt.plot(range(1200,1460),pred)
    plt.draw()
    plt.pause(0.01) 

def build_model():
    def mish(x):
        return x * tf.math.tanh(tf.math.softplus(x))
    def lrelu(x):
        return tf.keras.activations.relu(x, alpha=0.3)
    def res_block(input_layer, filters, conv_size,act_func):
        x = tf.keras.layers.Conv1D(filters,conv_size,padding='same',activation=act_func)(input_layer)
        x = tf.keras.layers.Conv1D(filters,conv_size,padding='same',activation=act_func)(x)
        return tf.keras.layers.Add()([x,input_layer])
    inp = tf.keras.layers.Input((1200,))
    x = tf.keras.layers.Dense(65)(inp)
    x = tf.keras.layers.Reshape((65,1))(x)
    
    x = tf.keras.layers.Conv1D(128,130,padding='same', strides=1,activation=mish)(x)
    for i in range(4):
        x = res_block(x,128,130,mish)
        
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1D(64,130,padding='same', strides=1,activation=mish)(x)
    for i in range(4):
        x = res_block(x,64,130,mish)
      
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1D(32,130,padding='same', strides=1,activation=mish)(x)
    for i in range(4):
        x = res_block(x,32,130,mish)
        
    x = tf.keras.layers.Conv1D(8,130,padding='same', strides=1,activation=mish)(x) 
    x = tf.keras.layers.Conv1D(1,130,padding='same', strides=1,activation='tanh')(x) 
    
    out = x
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),loss='logcosh')
    model.summary()
    return model
    
def train(model):
    plt.ion()
    inp = var(random.random()/20+0.05)
    for i in range(5):
        w = random.randint(0,1000)
        z = i+w
        inp_step = inp[z:z+1200]
        tar_step = inp[z+1200:z+1200+260]
        pred = tf.squeeze(model.predict_on_batch(tf.expand_dims(inp_step,0))) #option for off
        loss = model.train_on_batch(tf.expand_dims(inp_step,0),tf.expand_dims(tar_step,0))
        print('step {} loss {}'.format(i, loss))	
        plot(inp_step,pred) #option for off
    return model
        
def test(model):  
    print('start test')
    plt.ion()
    inp = var(random.random()/20+0.05)
    for i in range(100):
        inp_step = inp[i:i+1200]
        pred = tf.squeeze(model.predict_on_batch(tf.expand_dims(inp_step,0)))
        plot(inp_step,pred)
        
model = build_model()       
for i in range(2000): 
    print(i)
    train(model)
for i in range(10): 
    print(i)
    test(model)