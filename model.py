import tensorflow as tf
from tensorflow import keras


class InstanceNormalization(keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


def downsample_gg(filters, size,apply_norm=True):

    initializer = tf.random_normal_initializer(0., 0.02)

    result = keras.Sequential()
    result.add(
        keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))
    
    if apply_norm:
        result.add(InstanceNormalization())
    result.add(keras.layers.LeakyReLU())

    return result

def downsample_gf(filters, size, apply_norm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))

    if apply_norm:
        result.add(InstanceNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample_gg(filters, size, apply_dropout=False):

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(InstanceNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def upsample_gf(filters, size, apply_dropout=False):
  
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
    result.add(InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def downsample_dx(filters, size, apply_norm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))

    if apply_norm:
        result.add(InstanceNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def downsample_dy(filters, size, apply_norm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))

    if apply_norm:
        result.add(InstanceNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def generator_g():
    down_stack = [
      downsample_gg(64, 4, apply_norm=False),  # (bs, 128, 128, 64)
      downsample_gg(128, 4),  # (bs, 64, 64, 128)
      downsample_gg(256, 4),  # (bs, 32, 32, 256)
      downsample_gg(512, 4),  # (bs, 16, 16, 512)
      downsample_gg(512, 4),  # (bs, 8, 8, 512)
      downsample_gg(512, 4),  # (bs, 4, 4, 512)
      downsample_gg(512, 4),  # (bs, 2, 2, 512)
      downsample_gg(512, 4),  # (bs, 1, 1, 512)
  ]

    up_stack = [
        upsample_gg(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample_gg(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample_gg(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample_gg(512, 4),  # (bs, 16, 16, 1024)
        upsample_gg(256, 4),  # (bs, 32, 32, 512)
        upsample_gg(128, 4),  # (bs, 64, 64, 256)
        upsample_gg(64, 4),  # (bs, 128, 128, 128)
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
            3, 4, strides=2,
            padding='same', kernel_initializer=initializer,
            activation='tanh')
    concat=keras.layers.Concatenate()
    inp=keras.layers.Input(shape=[None, None, 1])
    x=inp
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)

    return keras.Model(inputs=inp, outputs=x)

def generator_f():
    down_stack = [
        downsample_gf(64, 4, apply_norm=False),  # (bs, 128, 128, 64)
        downsample_gf(128, 4),  # (bs, 64, 64, 128)
        downsample_gf(256, 4),  # (bs, 32, 32, 256)
        downsample_gf(512, 4),  # (bs, 16, 16, 512)
        downsample_gf(512, 4),  # (bs, 8, 8, 512)
        downsample_gf(512, 4),  # (bs, 4, 4, 512)
        downsample_gf(512, 4),  # (bs, 2, 2, 512)
        downsample_gf(512, 4),  # (bs, 1, 1, 512)
  ]

    up_stack = [
        upsample_gf(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample_gf(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample_gf(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample_gf(512, 4),  # (bs, 16, 16, 1024)
        upsample_gf(256, 4),  # (bs, 32, 32, 512)
        upsample_gf(128, 4),  # (bs, 64, 64, 256)
        upsample_gf(64, 4),  # (bs, 128, 128, 128)
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
            1, 4, strides=2,
            padding='same', kernel_initializer=initializer,
            activation='tanh')
    concat=keras.layers.Concatenate()
    inp=keras.layers.Input(shape=[None, None, 3])
    x=inp
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)

    return keras.Model(inputs=inp, outputs=x)

def discriminator_x():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, 1], name='input_image')
    x = inp

    down1 = downsample_dx(64, 4,  False)(x)  # (bs, 128, 128, 64)
    down2 = downsample_dx(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample_dx(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer,
        use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    norm1 = InstanceNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(
        1, 4, strides=1,
        kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=inp, outputs=last)

def discriminator_y():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
    x = inp

    down1 = downsample_dx(64, 4,  False)(x)  # (bs, 128, 128, 64)
    down2 = downsample_dx(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample_dx(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer,
        use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    norm1 = InstanceNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(
        1, 4, strides=1,
        kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=inp, outputs=last)
