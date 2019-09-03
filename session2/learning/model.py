import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

L2_LAMBDA = 1e-04
def _residual_block(x, size, dropout=False, dropout_prob=0.5, seed=7):
    residual = keras.layers.BatchNormalization()(x)  
    residual = keras.activations.relu(residual)
    residual = keras.layers.Conv2D(filters=size, kernel_size=3, strides=2, padding='same',
                                kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                                kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA))(residual)
    if dropout:
        residual = keras.layers.Dropout(dropout_prob, seed=seed)(residual)
        
    residual = keras.layers.BatchNormalization()(residual)
    residual = keras.activations.relu(residual)
    residual = keras.layers.Conv2D(filters=size, kernel_size=3, padding='same',
                                kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                                kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA))(residual)
    if dropout:
        residual = keras.layers.Dropout(dropout_prob, seed=seed)(residual)

    return residual


def one_residual(x, keep_prob=0.5, seed=7):
    
    nn = keras.layers.Conv2D(filters=32, kernel_size=5, strides=2, padding='same',
                          kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                          kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA))(x)
    
#     nn = keras.layers.MaxPooling2D(pool_size=3, strides=2)(nn)

    rb_1 = _residual_block(nn, 32, dropout=True,dropout_prob=keep_prob, seed=seed)

    nn = keras.layers.Conv2D(filters=32, kernel_size=1, strides=2, padding='same',
                          kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                          kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA))(nn)
    
    nn = keras.layers.Add()([rb_1, nn])

#     nn = keras.layers.Flatten()(nn)

    return nn

def MODEL(image_shape):
    inputs = keras.layers.Input(shape=image_shape)
    
    outputs = one_residual(inputs, seed=7)
    outputs = one_residual(outputs, seed=7)
    outputs = one_residual(outputs, seed=7)
    
    outputs = keras.layers.Flatten()(outputs)
    outputs = keras.layers.Dense(units=64,
                               activation="relu",
                               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=7),
                               bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=7))(outputs)
    
    outputs = keras.layers.Dense(units=32,
                               activation="relu",
                               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=7),
                               bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=7))(outputs) 
    
    outputs = keras.layers.Dense(units=4,activation="softmax")(outputs)
    

    return keras.models.Model(inputs=inputs,outputs=outputs)