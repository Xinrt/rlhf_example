"""
Core network which predicts rewards from frames,
for gym-moving-dot and Atari games.
"""

import tensorflow as tf

from nn_layers import dense_layer, conv_layer



def get_dot_position(s):
    """
    Estimate the position of the dot in the gym-moving-dot environment.
    """
    # s is (?, 84, 84, 4)
    s = s[..., -1]  # select last frame; now (?, 84, 84)

    x = tf.reduce_sum(s, axis=1)  # now (?, 84)
    x = tf.argmax(x, axis=1)

    y = tf.reduce_sum(s, axis=2)
    y = tf.argmax(y, axis=1)

    return x, y


# def net_moving_dot_features(s, batchnorm, dropout, training, reuse):
#     # Action taken at each time step is encoded in the observations by a2c.py.
#     a = s[:, 0, 0, -1]
#     a = tf.cast(a, tf.float32) / 4.0

#     xc, yc = get_dot_position(s)
#     xc = tf.cast(xc, tf.float32) / 83.0
#     yc = tf.cast(yc, tf.float32) / 83.0

#     features = [a, xc, yc]
#     x = tf.stack(features, axis=1)

#     x = dense_layer(x, 64, "d1", reuse, activation='relu')
#     x = dense_layer(x, 64, "d2", reuse, activation='relu')
#     x = dense_layer(x, 64, "d3", reuse, activation='relu')
#     x = dense_layer(x, 1, "d4", reuse, activation=None)
#     x = x[:, 0]

#     return x

def net_moving_dot_features(s, batchnorm, dropout, training, reuse):
    # Action taken at each time step is encoded in the observations by a2c.py.
    a = s[:, 0, 0, -1]
    a = tf.cast(a, tf.float32) / 4.0

    xc, yc = get_dot_position(s)
    xc = tf.cast(xc, tf.float32) / 83.0
    yc = tf.cast(yc, tf.float32) / 83.0

    # Add print operations
    a_print = tf.print("Action (a):", a, summarize=-1)
    xc_print = tf.print("X coordinate (xc):", xc, summarize=-1)
    yc_print = tf.print("Y coordinate (yc):", yc, summarize=-1)

    # Use tf.control_dependencies to ensure prints are executed
    with tf.control_dependencies([a_print, xc_print, yc_print]):
        features = [a, xc, yc]
        print("Features:", features)
        x = tf.stack(features, axis=1)

    x = dense_layer(x, 64, "d1", reuse, activation='relu')
    x = dense_layer(x, 64, "d2", reuse, activation='relu')
    x = dense_layer(x, 64, "d3", reuse, activation='relu')
    x = dense_layer(x, 1, "d4", reuse, activation=None)
    x = x[:, 0]

    # Add print operation for the reward
    reward_print = tf.print("Predicted Reward:", x, summarize=-1)

    # Use tf.identity with control dependencies to ensure the reward print is executed
    with tf.control_dependencies([reward_print]):
        x = tf.identity(x)

    return x


def net_cnn(s, batchnorm, dropout, training, reuse):
    x = s / 255.0
    # Page 15: (Atari)
    # "[The] input is fed through 4 convolutional layers of size 7x7, 5x5, 3x3,
    # and 3x3 with strides 3, 2, 1, 1, each having 16 filters, with leaky ReLU
    # nonlinearities (α = 0.01). This is followed by a fully connected layer of
    # size 64 and then a scalar output. All convolutional layers use batch norm
    # and dropout with α = 0.5 to prevent predictor overfitting"
    x = conv_layer(x, 16, 7, 3, batchnorm, training, "c1", reuse, 'relu')
    x = tf.layers.dropout(x, dropout, training=training)
    x = conv_layer(x, 16, 5, 2, batchnorm, training, "c2", reuse, 'relu')
    x = tf.layers.dropout(x, dropout, training=training)
    x = conv_layer(x, 16, 3, 1, batchnorm, training, "c3", reuse, 'relu')
    x = tf.layers.dropout(x, dropout, training=training)
    x = conv_layer(x, 16, 3, 1, batchnorm, training, "c4", reuse, 'relu')

    w, h, c = x.get_shape()[1:]
    x = tf.reshape(x, [-1, int(w * h * c)])

    x = dense_layer(x, 64, "d1", reuse, activation='relu')
    x = dense_layer(x, 1, "d2", reuse, activation=None)
    x = x[:, 0]

    return x




def net_mlp(s, batchnorm, dropout, training, reuse):
    """
    MLP network for Hopper-v4 environment.
    
    Args:
    s: Input tensor (state)
    batchnorm: Boolean, whether to use batch normalization
    dropout: Float, dropout rate
    training: Boolean, whether in training mode
    reuse: Boolean, whether to reuse variables
    
    Returns:
    x: Output tensor (predicted reward)
    """
    x = s  # Input is already normalized in Hopper-v4

    # MLP architecture with 3 hidden layers
    x = dense_layer(x, 64, "d1", reuse, activation='relu')
    if batchnorm:
        x = tf.layers.batch_normalization(x, training=training, reuse=reuse, name="bn1")
    x = tf.layers.dropout(x, rate=dropout, training=training)

    x = dense_layer(x, 64, "d2", reuse, activation='relu')
    if batchnorm:
        x = tf.layers.batch_normalization(x, training=training, reuse=reuse, name="bn2")
    x = tf.layers.dropout(x, rate=dropout, training=training)

    x = dense_layer(x, 64, "d3", reuse, activation='relu')
    if batchnorm:
        x = tf.layers.batch_normalization(x, training=training, reuse=reuse, name="bn3")
    x = tf.layers.dropout(x, rate=dropout, training=training)

    x = dense_layer(x, 1, "d4", reuse, activation=None)
    x = x[:, 0]  # Flatten to a 1D tensor

    return x

