import numpy as np
import tensorflow as tf


class EMGModel:
    def __init__(self, model_path='../assets/saved_model/finetuned'):
        self.current_model_path = None
        self.model = None
        self.stds = None
        self.reload_model(model_path)

    def reload_model(self, model_path):
        if model_path != self.current_model_path:
            self.current_model_path = model_path

            # custom_objects = {'learning_rate': 5e-6}

            # self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            self.model = tf.keras.models.load_model(model_path)
            self.stds = np.genfromtxt(f"{model_path}/stds.csv", delimiter=',')
            print(f'Loading model from {model_path}')
        return self.stds

    def predict(self, data, verbose=1):
        return self.model.predict(data, verbose)

    def calibrate(self, calibrate_ds, new_model_path):
        self.model.fit(calibrate_ds, epochs=10)
        self.model.save(new_model_path)

        # ri_counterpart = tf.keras.models.load_model(
        #     filepath='../assets/saved_model/random_inited')
        # ri_counterpart.fit(calibrate_ds, epochs=10)
        # ri_counterpart.save(new_model_path + "ri")
        #
        return new_model_path
