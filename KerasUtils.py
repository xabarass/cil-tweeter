from keras.models import model_from_json
from keras.models import Sequential

# Utilities for loading and saving a Keras sequential model

def save_model(model,model_save_path):
    assert isinstance(model, Sequential)
    print("Saving model...")
    model_json = model.to_json()
    with open(model_save_path + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(model_save_path + ".h5")
    print("Saved model to disk at %s.(json,h5)" % model_save_path)


def load_model(model_json_file, model_h5_file):
    print("Loading model...")
    with open(model_json_file, 'r') as structure_file:
        loaded_model_json = structure_file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(model_h5_file)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    assert isinstance(model, Sequential)

    print("Loaded model from disk!")
