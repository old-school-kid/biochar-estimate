import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend

class AttentionBlock(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.norm = keras.layers.GroupNormalization(epsilon=1e-5)
        self.q = layers.Conv2D(output_dim, 1)
        self.k = layers.Conv2D(output_dim, 1)
        self.v = layers.Conv2D(output_dim, 1)
        self.proj_out = layers.Conv2D(output_dim, 1)

    def call(self, inputs):
        x = self.norm(inputs)
        q, k, v = self.q(x), self.k(x), self.v(x)

        # Compute attention
        shape = tf.shape(q)
        h, w, c = shape[1], shape[2], shape[3]
        q = tf.reshape(q, (-1, h * w, c))  # b, hw, c
        k = tf.transpose(k, (0, 3, 1, 2))
        k = tf.reshape(k, (-1, c, h * w))  # b, c, hw
        y = q @ k
        y = y * 1 / tf.sqrt(tf.cast(c, self.compute_dtype))
        y = keras.activations.softmax(y)

        # Attend to values
        v = tf.transpose(v, (0, 3, 1, 2))
        v = tf.reshape(v, (-1, c, h * w))
        y = tf.transpose(y, (0, 2, 1))
        x = v @ y
        x = tf.transpose(x, (0, 2, 1))
        x = tf.reshape(x, (-1, h, w, c))
        return self.proj_out(x) + inputs

def conv_block(x, growth_rate, name):
    """A building block for a dense block.
    Args:
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    Returns:
        Output tensor for the block.
    """
    bn_axis = 3
    x1 = layers.LayerNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(
          x)
    x1 = layers.Activation('gelu', name=name + '_0_gelu')(x1)
    x1 = layers.Conv2D(
      4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(
          x1)
    x1 = layers.LayerNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(
          x1)
    x1 = layers.Activation('gelu', name=name + '_1_gelu')(x1)
    x1 = layers.Conv2D(
      growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(
          x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

def transition_block(x, reduction, name):
    """A transition block.
    Args:
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    Returns:
        output tensor for the block.
    """
    bn_axis = 3
    x = layers.LayerNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(
          x)
    x = layers.Activation('gelu', name=name + '_gelu')(x)
    x = layers.Conv2D(
      int(backend.int_shape(x)[bn_axis] * reduction),
      1,
      use_bias=False,
      name=name + '_conv')(
          x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def dense_block(x, blocks, name):
    for i in range(blocks):
        x = conv_block(x, 16, name=name + '_block' + str(i + 1))   # original= conv_block(x, 32, name=name + '_block' + str(i + 1))
    x= layers.Dropout((blocks/100), name= name+'_drop')(x)      # wasnt there
    return x

def encoder(
    blocks= [3, 6, 12, 8],
    input_shape=(512, 512, 3),
    weights="encoder.h5"):

    img_input = layers.Input(shape=input_shape)

    bn_axis = 3

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = layers.LayerNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(
            x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    layer_names= ["conv1/relu", "pool2_conv", "pool3_conv", "pool4_conv", "relu"]
    o256= x
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    o128= x
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    o64= x
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    o32= x
    x = transition_block(x, 0.5, name='pool4')
    x = layers.Conv2D(256, 1, name='project_encoder_attention')(x)
    x = AttentionBlock(256)(x)
    x = dense_block(x, blocks[3], name='conv5')
    x = layers.Activation('gelu', name='relu')(x)
    o16= x

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    inputs = img_input

    model = tf.keras.Model(inputs, [o256, o128, o64, o32, o16], name='encoder')
    model.load_weights(weights)
    return model

def head(weights= "head.h5"):
    inp= layers.Input(shape=(16, 16, 384))
    x = layers.GlobalMaxPooling2D(name='avg_pool')(inp)
    x= layers.Flatten()(x)
    x= layers.Dense(384, activation='gelu', name='dense')(x)
    x= layers.Dense(192, activation='gelu', name='penultimate_dense')(x)
    x_cat= layers.Dense(12, activation='softmax', name='vol_help_dense')(x)
    x= layers.Concatenate(axis=-1)([x_cat, x, x_cat])
    x= layers.Dense(24, activation='gelu', name='penultimate_layer')(x)
    out= layers.Dense(1, activation='relu', name='output_layer')(x)
    model= tf.keras.Model(inp, [out, x_cat], name='head')
    model.load_weights(weights)
    return model

def decoder_layer(x, concat, name, filters):

    o = layers.Conv2DTranspose(filters, (2, 2), name=f"upscale_{name}", strides=(2, 2), padding='same')(x)
    o = layers.concatenate([o, concat], axis=-1, name=f"concat_{name}")
    o = layers.ZeroPadding2D((1, 1), name=f"pad_{name}")(o)
    o = layers.Conv2D(filters*4, (3, 3), padding='valid', activation='gelu', name=f"decoder_conv_1_{name}")(o)
    o = layers.LayerNormalization()(o)
    o = layers.Conv2D(filters, (3, 3), padding='same', activation='gelu', use_bias=False, name=f"decoder_conv_2_{name}")(o)
    o = layers.LayerNormalization()(o)
    o = layers.Add()([o, concat])
    o = layers.Conv2D(2*filters, (3, 3), padding='same', activation='gelu', name=f"decoder_conv_{name}")(o)
    return o

def decoder(weights= "decoder.h5"):

    o256, o128, o64, o32, o16= layers.Input(shape=(256, 256, 64)), layers.Input(shape=(128, 128, 112)), layers.Input(shape=(64, 64, 152)), layers.Input(shape=(32, 32, 268)), layers.Input(shape=(16, 16, 384))
    x= decoder_layer(o16, o32, "32", 268)
    x = layers.Conv2D(256, 1, name='project_decoder_attention')(x)
    x = AttentionBlock(256)(x)
    x= decoder_layer(x, o64, "64", 152)
    out64= layers.Conv2D(4, 1, name='decoder_output_64_proj')(x)
    out64= layers.Conv2D(2, 1, activation='sigmoid', name='decoder_output_64')(out64)
    x= decoder_layer(x, o128, "128", 112)
    out128= layers.Conv2D(4, 1, name='decoder_output_128_proj')(x)
    out128= layers.Conv2D(2, 1, activation='sigmoid', name='decoder_output_128')(out128)
    x= decoder_layer(x, o256, "256", 64)
    out256= layers.Conv2D(4, 1, name='decoder_output_256_proj')(x)
    out256= layers.Conv2D(2, 1, activation='sigmoid', name='decoder_output_256')(out256)

    x= layers.Conv2DTranspose(8, (2, 2), name=f"upscale_final", strides=(2, 2), padding='same')(x)
    x3 = layers.ZeroPadding2D((1, 1), name=f"pad_final3")(x)
    x5 = layers.ZeroPadding2D((2, 2), name=f"pad_final5")(x)
    x3 = layers.Conv2D(4, (3, 3), padding='valid', activation='linear', kernel_initializer = 'he_normal', name=f"decoder_conv_final3")(x3)
    x5 = layers.Conv2D(4, (5, 5), padding='valid', activation='linear', kernel_initializer = 'he_normal', name=f"decoder_conv_final5")(x5)
    x= layers.concatenate([x3, x5], axis=-1, name=f"concat_final")
    out= layers.Conv2D(2, (1, 1), padding='valid', activation='sigmoid', name=f"output")(x)

    model = tf.keras.Model([o256, o128, o64, o32, o16], [out, out256, out128, out64], name='decoder')
    model.load_weights(weights)
    return model