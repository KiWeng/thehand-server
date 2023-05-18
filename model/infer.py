import numpy as np
import tensorflow as tf


class EMGModel:
    def __init__(self, model_path='../assets/saved_model/base'):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, data):
        return self.model.predict(data)


if __name__ == "__main__":
    new_model = EMGModel()
    new_model.model.summary()
