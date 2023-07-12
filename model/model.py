import tensorflow as tf


class EMGModel:
    def __init__(self, model_path='../assets/saved_model/finetuned'):
        self.current_model_path = None
        self.model = None

        self.reload_model(model_path)

    def reload_model(self, model_path):
        if model_path == self.current_model_path:
            return
        else:
            self.current_model_path = model_path

            custom_objects = {'learning_rate': 5e-6}

            self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            print(f'Loading model from {model_path}')

    def predict(self, data):
        return self.model.predict(data)

    def calibrate(self, calibrat_ds, new_model_path):
        self.model.fit(calibrat_ds, epochs=20)
        self.model.save(new_model_path)
        return new_model_path
