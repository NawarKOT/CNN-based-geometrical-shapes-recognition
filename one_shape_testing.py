import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow import keras
from PIL import Image

model_path = "model1.h5"
loaded_model = keras.models.load_model(model_path)


class_names = ['Circle', 'Heptagon', 'Hexagon', 'Nanogon', 'Octagon', 'Pentagon', 'Square', 'Star', 'Triangle']
image = cv2.imread("Test.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_fromarray = Image.fromarray(image)
resize_image = image_fromarray.resize((128, 128))

plt.imshow(resize_image)
expand_input = np.expand_dims(resize_image,axis=0)
input_data = np.array(expand_input)
input_data = input_data/255

pred = loaded_model.predict(input_data)
print(pred)
result = pred.argmax()
print(result)
class_names[result]