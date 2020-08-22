import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import pickle

import kerastuner
from tensorflow.keras.datasets import fashion_mnist

from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

import time
LOG_DIR = f"{int(time.time())}"


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#print(kerastuner.__version__)


x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

def build_model(hp):  
    model = keras.models.Sequential()

    model.add(Conv2D(hp.Int('input_units',
                                min_value=32,
                                max_value=256,
                                step=32), (3, 3), input_shape=x_train.shape[1:]))

    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(hp.Int('n_layers', 1, 4)): 
        model.add(Conv2D(hp.Int(f'conv_{i}_units',
                                min_value=32,
                                max_value=256,
                                step=32), (3, 3)))
        model.add(Activation('relu'))

    model.add(Flatten()) 
    model.add(Dense(10))
    model.add(Activation("softmax"))

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    return model


tuner = RandomSearch(
	build_model,
	objective = 'val_accuracy', #I could also use val_loss 
	max_trials = 1,
	executions_per_trial = 1,
	directory = LOG_DIR)

tuner.search(x =x_train,
			y=y_train,
			verbose=2, 
			epochs = 1, 
			batch_size = 64,
			validation_data=(x_test, y_test))


print(tuner.get_best_hyperparameters()[0].values)
print(tuner.get_best_models()[0].summary())

with open(f"tuner_{int(time.time())}.pkl", "wb") as f:
    pickle.dump(tuner, f)

#if you want to load it directly, open your folder and replace with your pickle name :)

# bestmodel = pickle.load(open("tuner_1598111441.pkl","rb"))
# print(tuner.get_best_hyperparameters()[0].values)
# print(tuner.results_summary())