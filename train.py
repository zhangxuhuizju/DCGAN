from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import ZeroPadding2D
from keras.layers import LeakyReLU
from keras.layers import UpSampling2D
from keras.layers import Conv2D
from keras.models import Sequential
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam, RMSprop
from keras.preprocessing import image
import os, sys
import tensorflow as tf

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

class DCGAN():
    def __init__(self):
        self.image_rows = 96
        self.image_cols = 96
        self.channels = 3
        self.image_shape = (self.image_rows, self.image_cols, self.channels)
        self.latent_dim = 100
        
        optimizer = Adam(0.0002, 0.5)
        optimizerD = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
        optimizerG = RMSprop(lr=0.0004, clipvalue=1.0, decay=6e-8)
        
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        self.generator = self.build_generator()
        
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        
        #对生成器进行训练即可
        self.discriminator.trainable = False
        
        #模型输入z，输出valid
        valid = self.discriminator(img)
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        
    def build_generator(self):
        model = Sequential()
        model.add(Dense(512*6*6, activation='relu', input_dim=self.latent_dim))
        model.add(Reshape((6,6,512)))  #reshape成 6*6*512
        model.add(UpSampling2D()) #upsampling后成为 12*12*512
        model.add(Conv2D(256, kernel_size=5, padding='same'))  #转成12*12*256
        model.add(BatchNormalization(momentum=0.8)) #控制过拟合
        model.add(Activation('relu')) #激活函数relu
        model.add(UpSampling2D())  #24*24*256
        model.add(Conv2D(128, kernel_size=5, padding='same'))  #转成24*24*128
        model.add(BatchNormalization(momentum=0.8)) #控制过拟合
        model.add(Activation('relu')) #激活函数relu
        model.add(UpSampling2D())  #48*48*256
        model.add(Conv2D(64, kernel_size=5, padding='same'))  #转成48*48*64
        model.add(BatchNormalization(momentum=0.8)) #控制过拟合
        model.add(Activation('relu')) #激活函数relu
        model.add(UpSampling2D())  #96*96*64
        model.add(Conv2D(self.channels, kernel_size=5, padding='same'))  #转成96*96*3,即最后生成的图片size
        model.add(Activation('tanh'))
        
        model.summary()
        
        noise = Input(shape=(self.latent_dim, ))
        img = model(noise)
        return Model(noise, img)  #输入噪声，输出img
    
    def build_discriminator(self):
        depth = 32
        model = Sequential()
        model.add(Conv2D(64, kernel_size=5, strides=2, input_shape=self.image_shape, padding='same'))  #48*48*64
        model.add(LeakyReLU(alpha=0.2)) 
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))  #24*24*128
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=5, strides=2, padding="same")) #12*12*256
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(512, kernel_size=5, strides=1, padding="same"))  #6*6*512
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        
        model.summary()
        
        img = Input(shape=self.image_shape)
        score = model(img)
        
        return Model(img, score)
    
    def load_batch_images(self, batch_size, dirName):
        image_names = np.array(os.listdir(os.path.join(dirName)))
        index = np.random.randint(0, image_names.shape[0], batch_size)
        image_names = image_names[index]
        imgs = []
        
        for i in range(len(image_names)):
            img = image.load_img(os.path.join(dirName, image_names[i]), target_size=(96, 96))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            imgs.append(img)
            
        imgs = np.concatenate([img for img in imgs])
        imgs = imgs / 127.5 - 1 #正则化，[-1,1]
        return imgs
    
    def save_imgs(self, epoch):
        row = 5
        col = 5
        noise = np.random.normal(0, 1, (row*col, self.latent_dim))
        gen_imgs = self.generator.predict(noise) #此时数据[-1,1]范围内
        gen_imgs = 0.5*gen_imgs + 0.5 #恢复成[0, 1]
        gen_imgs = gen_imgs.astype(float32)
        
        fig, axs = plt.subplots(row, col)
        count = 0
        for i in range(row):
            for j in range(col):
                axs[i, j].imshow(gen_imgs[count, :, :, :])
                axs[i, j].axis('off')
                count += 1
        fig.savefig("results/epoch_%d.png" %epoch)
        plt.close()
        
    
    def train(self, epochs, batch_size=256, save_steps=50):
        true_img = np.ones((batch_size, 1))
        fake_img = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            imgs = self.load_batch_images(batch_size, 'cartoonFaces')
            noise = np.random.normal(0,1,(batch_size, self.latent_dim))
            fake_imgs = self.generator.predict(noise)
            
            d_loss_real = self.discriminator.train_on_batch(imgs, true_img)
            d_loss_fake = self.discriminator.train_on_batch(fake_imgs, fake_img)
            d_loss = 0.5 * np.add(d_loss_fake, d_loss_fake)
            
            g_loss = self.combined.train_on_batch(noise, true_img)
            
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            
            if epochs % save_steps == 0:
                self.combined.save('./model/combined_model_%d.h5' %epoch)
                self.discriminator.save('./model/discriminator_model_%d.h5' %epoch)
                self.save_imgs(epoch)

if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=50000, save_steps=20)