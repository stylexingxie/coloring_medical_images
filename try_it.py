import tensorflow as tf
import pix2pix
import os
import time
import matplotlib.pyplot as plt
import cv2
import pix2pix
import tensorflow as tf

AUTOTUNE=tf.data.AUTOTUNE
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

generator_g=pix2pix.unet_generator(3,norm_type='instancenorm')
generator_g.load_weights('./generatior_weight/')

def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

def preprocess_input_test(image):
    image=tf.image.decode_jpeg(image,channels=3)
    image=tf.image.resize(image,[256,256])
    image=normalize(image)
    return image


testA=tf.data.TFRecordDataset('./testA.tfrec')
testInputData=testA.cache().map(preprocess_input_test,num_parallel_calls=AUTOTUNE).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def generate_images(model, test_input):
    prediction = model(test_input)

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

# Run the trained model on the test dataset
for inp in testInputData.take(5):
    generate_images(generator_g, inp)
