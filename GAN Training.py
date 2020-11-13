#this code may not be perfect or the most effecient but it does the job.
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import layers
import time
from IPython import display
from keras import backend as K
#change datadir to the folder of the dataset
DATADIR = "C:/Users/Jimi/Desktop/Anime AI/datasets/AnimeGirls"

CATAGORIES = ["1"] #just used for counting in a for loop. i was too lazy to get rid of it and change the for loop.
training_data = []
#image sizes but make sure to adjust the generator output and disriminator inpput
a = 128
b = 32
c = 256
Image_size = a
#loads and resizes training data from dataset folder location
def create_training_data():
    for c in CATAGORIES:
        classN = CATAGORIES.index(c)
        
        for img in os.listdir(DATADIR):
            try:
                img_array = cv2.imread(os.path.join(DATADIR,img))
                new_array = cv2.cvtColor(cv2.resize(img_array, (Image_size, Image_size)), cv2.COLOR_BGR2RGB)
                training_data.append(new_array)
            except Exception as e:
                pass
create_training_data()
# reshape traing data then convert it from 0-255 to (-1) - 1
training_data = np.array(training_data).reshape(-1, Image_size, Image_size, 3)
training_data = (np.array(training_data) - 127.5) / 127.5
random.shuffle(training_data)

BATCH_SIZE = 32

#builds traing dataset
train_dataset = tf.data.Dataset.from_tensor_slices(training_data).batch(BATCH_SIZE)



def make_generator_model():
    model = tf.keras.Sequential()
    #noise input and projection
    model.add(layers.Dense(8*8*512, use_bias=False, input_shape=(512,)))
    model.add(layers.BatchNormalization())
    
    #reshape input
    model.add(layers.Reshape((8, 8, 512)))
    assert model.output_shape == (None, 8, 8, 512)
    
    #conv2DTranspose layers upscale images and work the opposite way to convolutional layers
    model.add(layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 512)
    model.add(layers.BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 256)
    model.add(layers.BatchNormalization())
    model.add(Activation("relu"))
    
    
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 128)
    model.add(layers.BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(layers.Conv2DTranspose(64, (6, 6), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 128, 128, 64)
    model.add(layers.BatchNormalization())
    model.add(Activation("relu"))
    
    #three filters because the image is in colour if you have a black and white image dataset set this to one also make sure to update the discriminator input and whre the dataset is reshaped.
    model.add(layers.Conv2DTranspose(3, (6, 6), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 128, 128, 3)
    #output layer doesnt need batch norm.
    
    
    
    return model

generator = make_generator_model()

noise = tf.random.normal([1, 512])
generated_image = generator(noise, training=False)
generated_image = generated_image
# make sure the discriminaor and generator are balanced best way to do this is by makingthem mirror each other
def make_discriminator_model():
    model = tf.keras.Sequential()
    
    #use convolutional strides instead of max pooling
    model.add(layers.Conv2D(64, (6, 6), strides=(2, 2), padding='same', input_shape=[128, 128, 3]))
    model.add(layers.LeakyReLU(0.2))
    #input layer doesnt need batch norm.
    
    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    
    model.add(layers.Dropout(0.3))
    #dropout layers help avoid overfitting
    
    model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    
    model.add(layers.Dropout(0.3))
    
    #best not to use interconnected dense layers and just go straight to the output
    model.add(Flatten())
    model.add(Dense(1))

    return model
    
discriminator = make_discriminator_model()
decision = discriminator(generated_image)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
    
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
    
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 610 #how long you want to train
noise_dim = 512 #how much noise do you give the generator make sure sure to change the input of the generator to the same value.
num_examples_to_generate = 36


seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)
      
      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm)
  
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(6,6))

  for i in range(predictions.shape[0]):
      plt.subplot(6, 6, i+1)
      #this is isnt my code I dont know why it wont show the images in proper colour. Use ganTest.py to get color images from the generaotr.
      plt.imshow(predictions[i, :, :, 0] * 127) 
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
  
def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)


    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator, epochs, seed)
  
train(train_dataset, EPOCHS)
# you can name the model whatever you like.
generator.save('AnimeGirl.model')
print("saved the model")
