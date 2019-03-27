# import library
from keras.models import model_from_json, load_model
from pathlib import Path
from keras.preprocessing import image
import numpy as np

# Defining output class for recognition
class_labels = [
	"Plane",
	"Car",
	"Bird",
	"Cat",
	"Deer",
	"Dog",
	"Frog",
	"Horse",
	"Ship",
	"Truck"
]

model = load_model('C:/PRANAVRAJ/PythonScripts/DeepLearning/model_structure.h5')
'''
# Loading the model output file (json file)
f = Path("model_structure.json")
model_structure = f.read_text()

# Re-creating the keras model from the stored file (json file)
model = model_from_json(model_structure)

# Reloading the weights from the trained model
model.load_weights("model_weights.h5")
'''

# Load testing image & convert to numpy array
img = image.load_img("C:\\PRANAVRAJ\\PythonScripts\\DeepLearning\\TestImage\\horse2.png", target_size = (32,32))
image_to_test = image.img_to_array(img)

# Adding 4th dimension to the testing image [as keras expect a list of image]
list_of_images = np.expand_dims(image_to_test, axis = 0)

# Make the prediction with trained model
results = model.predict(list_of_images)
single_result = results[0] # since we have only one image for the test from list of images in Results list

# Calculating likelyhood of all the defined classes out of 10
most_likely_class_index = int(np.argmax(single_result))
class_likelihood = single_result[most_likely_class_index]

# Getting the name of most likelyhood class
class_label = class_labels[most_likely_class_index]

# Finally print the result of prediction
print ('\n')
print ('\n')
print("This image is a {} - Likelihood: {:2f}".format(class_label, class_likelihood))

