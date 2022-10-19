import tensorflow as tf
import cv2
import pix2pix
import matplotlib.pyplot as plt
from tensorflow import keras
IMG_WIDTH = 256
IMG_HEIGHT = 256
AUTOTUNE=tf.data.experimental.AUTOTUNE
BATCH_SIZE=1
BUFFER_SIZE = 1000
# generator_f=pix2pix.unet_generator(3,norm_type='instancenorm')
# generator_g=pix2pix.unet_generator(3,norm_type='instancenorm')
# discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
# downple=keras.Sequential()
# downple.add(
#     keras.layers.Conv2D(128,4,2,padding='same',kernel_initializer=tf.random_normal_initializer(0.,0.02),use_bias=False)
# )
# downple.add(pix2pix.InstanceNormalization())
# downple.add(keras.layers.LeakyReLU())

# keras.utils.plot_model(generator,'./gen.png',show_shapes=True)
# keras.utils.plot_model(discriminator_x,'./disc.png',show_shapes=True)
# keras.utils.plot_model(downple,'./paper/downsample.png',show_shapes=True)

def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def random_crop(image):
    cropped_image = tf.image.random_crop(
        image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image

def random_jitter(image):
    # resizing to 286 x 286 x channels
    image = tf.image.resize(image, [286, 286],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # randomly cropping to 256 x 256 x channels
    image = random_crop(image)
    # random mirroring
    image = tf.image.random_flip_left_right(image)
    return image

trainA=tf.data.TFRecordDataset('./trainA.tfrec')
trainB=tf.data.TFRecordDataset('./trainB.tfrec')
def preprocess_input_train(image):
    image=tf.image.decode_jpeg(image,channels=3)
    image=random_jitter(image)
    image=normalize(image)
    return image

trainInputData=trainA.cache().map(preprocess_input_train,num_parallel_calls=AUTOTUNE).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
trainTargetData=trainB.cache().map(preprocess_input_train,num_parallel_calls=AUTOTUNE).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
sample_train=next(iter(trainInputData))
sample_target=next(iter(trainTargetData))

# to_target=generator_g(sample_train)
# to_input=generator_f(sample_target)
# plt.figure(figsize=(8,8))
# contrast = 8
# imgs=[sample_train,to_target,sample_target,to_input]
# title=['input','to Target','target','to Input']
# for i in range(len(imgs)):
#     plt.subplot(2, 2, i+1)
#     plt.title(title[i])
#     if i % 2 == 0:
#         plt.imshow(imgs[i][0] * 0.5 + 0.5)
#     else:
#         plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
# plt.show()

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)
plt.figure(figsize=(8, 8))

plt.subplot(121)
plt.title('discriminator_y?')
plt.imshow(discriminator_y(sample_target)[0, ..., -1], cmap='RdBu_r')

plt.subplot(122)
plt.title('discriminator_x?')
plt.imshow(discriminator_x(sample_train)[0, ..., -1], cmap='RdBu_r')

plt.show()
# plt.figure()
# plt.subplot(121)
# plt.title('image')
# plt.imshow(sample_train[0] * 0.5 + 0.5)

# plt.subplot(122)
# plt.title('image with random jitter')
# plt.imshow(random_jitter(sample_train[0],1) * 0.5 + 0.5)
# plt.show()

