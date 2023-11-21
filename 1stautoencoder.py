import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model


(x_train,_),(x_test,_)=fashion_mnist.load_data()

x_train=x_train.astype('float32')/255 #reducing the variability of the color to obtain an easier dataset
x_test=x_test.astype('float32')/255

print (x_train.shape)
print (x_test.shape)

latent_dim=100

class Autoencodeur(Model):
    def __init__(self,latent_dim):
        super(Autoencodeur,self).__init__()
        self.latent_dim=latent_dim
        self.encoder=tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim,activation='relu')
            ])
        self.decoder=tf.keras.Sequential([
            layers.Dense(784,activation='sigmoid'),
            layers.Reshape((28,28))
            ])
    def call(self,x):
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return decoded
    
autoencodeur=Autoencodeur(latent_dim)

autoencodeur.compile(optimizer='adam',loss=losses.MeanSquaredError())

autoencodeur.fit(x_train,x_train,epochs=10,shuffle=True,
                 validation_data=(x_test,x_test))

encoded=autoencodeur.encoder(x_test).numpy()
decoded=autoencodeur.decoder(encoded).numpy()

n=10
plt.figure(figsize=(20,4))
for i in range(n):
        #afficher l'originale
        ax=plt.subplot(2,n,i+1)
        plt.imshow(x_test[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
            
        #afficher la reconstruction
        ax=plt.subplot(2,n,i+1+n)
        plt.imshow(decoded[i])
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.show()





        
