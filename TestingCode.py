#testing code

import tensorflow as tf
from keras.preprocessing import image
import numpy as np


model=tf.keras.models.load_model("C:/Users/Sreehasa/Desktop/Miniproj/A-DogBreed/dogbreed2_model.h5")
path = 'C:/Users/Sreehasa/Desktop/gold.jpg' 
img = image.load_img(path, target_size=(150,150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes)