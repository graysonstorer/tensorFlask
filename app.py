import os
from flask import *
from PIL import Image as im
from matplotlib import cm


import requests
from io import BytesIO



app = Flask(__name__)
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds
import os
from numpy import asarray

keras = tf.keras

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # split data manually

    # get_label_name = metadata.features['label'].int2str  # creates a string label of the picture i think

    # images need to be reformatted

IMG_SIZE = 160

    # def format_example(image, label):
    #     # returns image reshaped to image_size
    #     image = tf.cast(image, tf.float32)
    #     image = image / 127.5 - 1
    #     image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    #     return image, label

for image in train_images:
    image = cv2.resize(image, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)

for image in test_images:
    image = cv2.resize(image, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)

train_images, test_images = train_images / 255, test_images / 255

print(train_images)
    # test_img = train_images[14]
    # test_img = tf.cast(image, tf.float32)
    # test_img = test_img / 127.5 - 1
    # test_img = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    # img = image.img_to_array(test_img)
    # img = img.reshape((1,) + img.shape)
    #
    # i = 0
    # #
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((1, 1)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10))

print(model.summary())

model.compile(
    optimizer='Adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics='accuracy'
)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)


history = model.fit(train_images, train_labels, epochs=15, validation_data=(test_images, test_labels),
                    callbacks=[cp_callback])

# model.load_weights(checkpoint_path)

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))



@app.route('/')
def index():
    return render_template('index.html', prompt='welcome')




@app.route('/trainAndPredict', methods=(['POST']))
def actionBaby():
    index = (request.form.get('imageIndex'))

    response = requests.get(index)
    img = im.open(BytesIO(response.content))

    # img.save("static/imageOne.jpeg")


    # img = im.open('static/'+str(index)+".jpeg")

    data = asarray(img)


    data = cv2.resize(data, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
    data = data / 255

    data = tf.reshape(data, [1, 32, 32, 3])

    predictions = model.predict(data)

    img.save('static/imageOne.png')

    return render_template('result.html', result=class_names[np.argmax(predictions)])



# index = int(input("enter the index"))
#
# (model)
# (test_images)
# (class_names)
#
# predictions = model.predict(test_images)
# print('prediction is', class_names[np.argmax(predictions[index])])
# print('with a confidence of ', predictions[index][np.argmax(predictions[index])])

# plt.imshow(test_images[index], cmap=plt.cm.binary)
# plt.xlabel(class_names[test_labels[index][0]])
# plt.show()



# image = im.fromarray(np.uint8(cm.gist_earth(test_images)*255))
# image.save('image.png')
#
# return render_template('index.html', result = class_names[np.argmax(predictions[index])], image = 'image.png')









