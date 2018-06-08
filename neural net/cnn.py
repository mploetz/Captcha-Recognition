from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

# set up logging
# INFO sends msg to log file | default is stderr | ie "Waiting for data to be loaded"
tf.logging.set_verbosity(tf.logging.INFO)

# App logic | Run app
if __name__ == "__main__":
    tf.app.run()

def cnn_model_fn(features, labels, mode):
    # NOTE: Model function for CNN

    # Input layer
    # NOTE: Reshapes a tensor. 
    # Given tensor, this operation returns a tensor that has the same values as tensor with shape shape.
    # If one component of shape is the special value -1, the size of that dimension is computed so that the total size remains constant.
    #  In particular, a shape of [-1] flattens into 1-D. At most one component of shape can be -1
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convulutional Layer 1
    # NOTE: conv2d(). Constructs a two-dimensional convolutional layer. 
    # Takes number of filters, filter kernel size, padding, and activation function as arguments.
    conv1 = tf.layers.Conv2D(
        inputs=input_layer,
        filters=32,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer 1
    # NOTE: max_pooling2d(). Constructs a two-dimensional pooling layer using the max-pooling algorithm.
    #  Takes pooling filter size and stride as arguments.
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2,2],
        strides=2 # strides are how the conv moves in the matrix of pixels. ie by what amount
    )

    # Convolutional Layer 2 and Pooling Layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu
    )

    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2,2],
        strides=2
    )

    # Dense layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=tf.nn.relu
    )
    # NOTE: Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting. 
    # The units that are kept are scaled by 1 / (1 - rate), so that their sum is unchanged at training time and inference time.
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # Logits Layer
    logits = tf.layers.dense(
        inputs=dropout,
        units=10
    )

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add 'softmax_tensor' to the graph. 
        # It is used for PREDICT and by the 'logging_hook'
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits
    )

    # Configure the training op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    


