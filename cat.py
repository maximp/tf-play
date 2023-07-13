import tensorflow as tf
import numpy as np
import keras
import time


IMAGE_SIZE = 224
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
resnet = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE)

img = tf.keras.preprocessing.image.load_img('cat.png', target_size=(IMAGE_SIZE, IMAGE_SIZE))
t_start = time.time()
img_data = tf.keras.preprocessing.image.img_to_array(img)
x = tf.keras.applications.resnet50.preprocess_input(np.expand_dims(img_data, axis=0))
probabilities = resnet.predict(x)

print(tf.keras.applications.resnet50.decode_predictions(probabilities, top=5))
print("dT", time.time() - t_start)