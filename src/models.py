from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
import keras.backend as K


def get_model(model_id, input_shape, nb_actions):
    if model_id == 'standard':
        return standard(input_shape, nb_actions)
    elif model_id == 'large':
        return large(input_shape, nb_actions)
    elif model_id == 'recurrent':
        return recurrent(input_shape, nb_actions)
    elif model_id == 'small':
        return small(input_shape, nb_actions)
    else:
        print("Unknown model id, using standard")
        return standard(input_shape, nb_actions)


def standard(input_shape, nb_actions):
    model = Sequential()
    if K.image_dim_ordering() == 'tf':
        # (width, height, channels)
        model.add(Permute((2, 3, 1), input_shape=input_shape))
    elif K.image_dim_ordering() == 'th':
        # (channels, width, height)
        model.add(Permute((1, 2, 3), input_shape=input_shape))
    else:
        raise RuntimeError('Unknown image_dim_ordering.')
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    # print(model.summary())
    return model


def large(input_shape, nb_actions):
    return standard(input_shape, nb_actions)


def recurrent(input_shape, nb_actions):
    return standard(input_shape, nb_actions)


def small(input_shape, nb_actions):
    return standard(input_shape, nb_actions)
