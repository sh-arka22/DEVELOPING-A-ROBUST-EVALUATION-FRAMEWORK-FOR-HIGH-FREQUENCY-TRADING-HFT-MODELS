# filepath: /Users/arkajyotisaha/Desktop/My-Thesis/code/src/model/lstm_tf.py -->
import tensorflow as tf
from tensorflow.keras import layers, models

def build(input_shape, num_classes=3, units=(64, 64), dropout=0.2):
    x = inp = layers.Input(input_shape)
    for i, u in enumerate(units):
        x = layers.LSTM(u, return_sequences=i < len(units)-1)(x)
        if dropout:
            x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inp, out, name="LSTM_TF")
    model.compile("adam", "sparse_categorical_crossentropy", ["accuracy"])
    return model
