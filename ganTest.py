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



#ask user which model they want to test or use.
ModelName = input('which model? ')
Model = ModelName + '.model'
Gen = tf.keras.models.load_model(Model)



noise = tf.random.normal([1, 512])
#create an image change it from (-1) - 1 to 0-255 and show it.
prediction = (Gen.predict(noise))
a = ((np.array(prediction[0])*127)+127).astype(np.uint8)
plt.imshow(a)
plt.show()
#ask user if the wanto save if yes it will saveto the same folder as this script.
should_save = input("Do you want to save? ")
if should_save == "yes" or should_save == "Yes":
    NameInput = input("What do you want to name the file? ")
    Name = NameInput + ".jpg"
    a = np.array(a)
    cv2.imwrite(Name, cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
    print("saved!")
else:
    print("didn't save :(")
    
