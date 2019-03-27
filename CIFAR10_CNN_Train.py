# Import Library
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from pathlib import Path

tf.reset_default_graph()

# Loading the data set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalizing the data set to make values of individual pixel in range 0-1
  
  # ..Converting Training & Testing input data to float type 
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
  
  # ..Normalizing train and test input data
x_train = x_train/255
x_test = x_test/255

# Define output class vector [e.g how many different class to be defined for categorizing output]
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)

# Creating deep learning model
model = Sequential()

# Adding convolution layers to model
  # ..Layer 1 details
model.add(Conv2D(32,(3,3), padding = "same", activation = "relu", input_shape = (32,32,3)))
model.add(Conv2D(32,(3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
  # ..Layer 2 details [Input image is not required as this is 2nd layer
model.add(Conv2D(64,(3,3), padding ="same", activation= "relu"))
model.add(Conv2D(64,(3,3), activation= "relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
  #..Last layer to be fully connected or flatten
model.add(Flatten())

# Adding dense layers to the model
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

# Compiling the model
model.compile(
	loss = "categorical_crossentropy",
	optimizer = "adam",
	metrics = ["accuracy"]
)

# Training the model
model.fit(
	x_train,
	y_train,
	batch_size = 32,
	epochs = 30,
	validation_data = (x_test, y_test),
	shuffle = True
)

#Saving deep learning network structure
'''
model_structure = model.to_json()
f = Path("model_structure.json")
f.write_text(model_structure)

# Saving deep learning's training weights
model.save_weights("model_weights.h5")
'''
model.save('C:/PRANAVRAJ/PythonScripts/DeepLearning/model_structure_epoch40.h5')

# For Tensorboard vizualization
merged = tf.summary.merge_all()
saver = tf.train.Saver()
sess = tf.Session()
train_writer = tf.summary.FileWriter('C:/PRANAVRAJ/PythonScripts/DeepLearning', sess.graph)

# Printing/Displaying model summary
model.summary()
val_loss, val_acc = model.evaluate(x_test, y_test)
print ("Model validation loss is :{} - Validation accuracy is: {}".format(val_loss, val_acc))
