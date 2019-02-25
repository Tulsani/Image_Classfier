from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#intializing the convolution neural network
classfier=Sequential()

#step 1 - Convultion
classfier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
#Step 2 - MaxPooling
classfier.add(MaxPooling2D(pool_size=(2,2)))
#Step 3 - Flattening
classfier.add(Flatten())

#Step 4 - Full Connection - connecting the convolution network to neural network
classfier.add(Dense(output_dim=128,activation='relu'))
classfier.add(Dense(output_dim=1,activation='sigmoid'))

#Compiling the CNN
classfier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Data augumentation - since neural net is not smart, even slight difference in images
#shall be considered as seperate images thus to increase our training set size we shall
# provide slight changes to images

#part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)
test_datagen= ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64,64),
                                                batch_size=32,
                                                class_mode='binary')

test_set=test_datagen.flow_from_directory('dataset/test_set',
                                        target_size=(64,64),
                                        batch_size=32,
                                        class_mode='binary')

from IPython.display import display
from PIL import Image

classfier.fit_generator(
         training_set,
         steps_per_epoch=8000,
         epochs=10,
         validation_data=test_set,
         validation_steps=800)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dog.jpg',target_size=(64,64))
test_image = image.image_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = classfier.predict(test_image)
training_set.class_indices
if result[0][0]>=0.5:
    prediction='dog'
else:
    prediction='cat'
print(prediction)
