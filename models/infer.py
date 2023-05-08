import tensorflow as tf


def load_model(model_path='../assests/saved_model/base'):
    return tf.keras.models.load_model(model_path)


def infer(model, data):
    return model.fit(data)


if __name__ == "__main__":
    new_model = load_model()
    new_model.summary()
