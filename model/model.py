import tensorflow as tf


class EMGModel:
    def __init__(self, model_path='../assets/saved_model/finetuned'):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, data):
        return self.model.predict(data)

    def calibrate(self, calibrat_ds, new_model_path):
        self.model.fit(calibrat_ds, epochs=5)
        self.model.save(new_model_path)
        return new_model_path
