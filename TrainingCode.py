#training code
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = 'C:/Users/Sreehasa/Desktop/Miniproj/A-DogBreed/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

# pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output


from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
#x = layers.Dense  (1, activation='sigmoid')(x)
x=layers.Dense(3, activation='softmax')(x)

model = Model( pre_trained_model.input, x)

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])





from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir="C:/Users/Sreehasa/Desktop/Miniproj/A-DogBreed/DataSet-DogBreed/classes/"

'''
# Define our example directories and files
train_dingo_dir = os.path.join(base_dir+'train/dingo')
train_golden_dir = os.path.join(base_dir+'train/golden_retriever')
train_pug_dir = os.path.join(base_dir+'train/pug')

valid_dingo_dir = os.path.join(base_dir+'valid/dingo')
valid_golden_dir = os.path.join(base_dir+'valid/golden_retriever')
valid_pug_dir = os.path.join(base_dir+'valid/pug')
'''

trainimgs = 0
for root, dirs, files in os.walk(base_dir+'train'):
    trainimgs += len(files)

validimgs = 0
for root, dirs, files in os.walk(base_dir+'valid'):
    validimgs += len(files)

#steps_per_epoch = len(X_train)//batch_size
#validation_steps = len(X_test)//batch_size
steps_per_epoch = 194//20
validation_steps = 47//20 



# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(base_dir+'train/',
                                                    batch_size = 20,
                                                    class_mode = 'categorical',
                                                    target_size = (150, 150))
# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory( base_dir+'valid/',
                                                          batch_size  = 20,
                                                          class_mode  = 'categorical',
                                                          target_size = (150, 150))

history = model.fit(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = steps_per_epoch,
            epochs = 20,
            validation_steps = validation_steps,
            verbose = 2)





history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)

model.save("C:/Users/Sreehasa/Desktop/Miniproj/A-DogBreed/dogbreed2_model.h5")



import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()