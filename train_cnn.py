from os.path import join
import os
#import keras.backend as K
#K.set_image_dim_ordering('')
from keras.callbacks import ModelCheckpoint #new
from keras.layers import Conv2D, Dense, Activation, MaxPooling2D, Lambda #lambda is new
from keras.layers.merge import concatenate #new
from keras.models import Model #new
from keras.layers import Flatten, BatchNormalization, Dropout
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import multi_gpu_model
from keras.callbacks import Callback
from data_provider import MODELS_DIR
from data_provider import load_organized_data_info
from data_provider import train_val_dirs_generators
import tensorflow as tf
#convert to channels last
#IMGS_DIM_3D = (3, 256, 256)
IMGS_DIM_3D = (256,256,3)
#Using this for testing multi-gpu models
CNN_MODEL_FILE = join(MODELS_DIR,'cnn.h5')
#CNN_MODEL_FILE = join(MODELS_DIR, 'cnn.h5')
MAX_EPOCHS = 500
BATCH_SIZE = 96
#BATCH_SIZE = 5 used for testing
L2_REG = 0.003
W_INIT = 'he_normal'
LAST_FEATURE_MAPS_LAYER = 46
LAST_FEATURE_MAPS_SIZE = (128, 8, 8)
PENULTIMATE_LAYER = 51
PENULTIMATE_SIZE = 2048
SOFTMAX_LAYER = 55
SOFTMAX_SIZE = 1584
KERNAL_SIZE = 3


def _train_model():
    # Change IMGS_DIM_3D to 0 instead of index 2 because we reversed order for the tensorflow switch
    data_info = load_organized_data_info(IMGS_DIM_3D[0])
    dir_tr = data_info['dir_tr']
    dir_val = data_info['dir_val']

    #just threw this in here from  https://github.com/keras-team/keras/issues/8649 flagged in git issue 6. 
    # currently it just saves every epoch. can probably modify to only save if highest accuracy. 
    class MyCbk(Callback):

        def __init__(self, model):
             self.model_to_save = model

        def on_epoch_end(self, epoch, logs=None):
            self.model_to_save.save('model_at_epoch_%d.h5' % epoch)

    gen_tr, gen_val = train_val_dirs_generators(BATCH_SIZE, dir_tr, dir_val)


    # with tf.device('/cpu:0'): # important!!!
    #     model = _cnn(IMGS_DIM_3D)
    #     print('Single tower model:')
    #     model.summary()
    #
    #     if gpu_count > 1:
    #         model = make_parallel(model, gpu_count)
    #
    #         print('Multi-GPU model:')
    #         model.summary()
    #
    #     model = compile_model(model)
    #     model.fit_generator(
    #         generator=gen_tr,
    #         epochs=MAX_EPOCHS,
    #         steps_per_epoch=data_info['num_tr'],
    #         validation_data=gen_val,
    #         validation_steps=data_info['num_val'],
    #         callbacks=[ModelCheckpoint(CNN_MODEL_FILE, save_best_only=True)],
    #         verbose=1)


    model = _cnn(IMGS_DIM_3D) #new

    #model.load_weights("/home/nkim/art_cnn/models/cnn.h5")
    #print("Model weights have been updated!")

    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model = compile_model(parallel_model)
    cbk = MyCbk(model) #new

    parallel_model.fit_generator(
        generator=gen_tr,
        epochs=MAX_EPOCHS,
        steps_per_epoch=data_info['num_tr'],
        validation_data=gen_val,
        validation_steps=data_info['num_val'],
        callbacks=[cbk], #new
        verbose=1)


    # model.fit_generator(
    #     generator=gen_tr,
    #     epochs=MAX_EPOCHS,
    #     steps_per_epoch=data_info['num_tr'],
    #     validation_data=gen_val,
    #     validation_steps=data_info['num_val'],
    #     callbacks=[ModelCheckpoint(CNN_MODEL_FILE, save_best_only=True)],
    #     verbose=1)

# def make_parallel(model, gpu_count):
#     def get_slice(data, idx, parts):
#         shape = tf.shape(data)
#         size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
#         stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
#         start = stride * idx
#         return tf.slice(data, start, size)
#
#     outputs_all = []
#     for i in range(len(model.outputs)):
#         outputs_all.append([])
#
#     #Place a copy of the model on each GPU, each getting a slice of the batch
#     for i in range(gpu_count):
#         with tf.device('/gpu:%d' % i):
#             with tf.name_scope('tower_%d' % i) as scope:
#
#                 inputs = []
#                 #Slice each input into a piece for processing on this GPU
#                 for x in model.inputs:
#                     input_shape = tuple(x.get_shape().as_list())[1:]
#                     slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
#                     inputs.append(slice_n)
#
#                 outputs = model(inputs)
#
#                 if not isinstance(outputs, list):
#                     outputs = [outputs]
#
#                 #Save all the outputs for merging back together later
#                 for l in range(len(outputs)):
#                     outputs_all[l].append(outputs[l])
#
#     # merge outputs on CPU
#     with tf.device('/cpu:0'):
#         merged = []
#         for outputs in outputs_all:
#             merged.append(concatenate(outputs, axis=0))
#
#         return Model(inputs=model.inputs, outputs=merged)



def _cnn(imgs_dim, compile_=True):
    model = Sequential()

    model.add(_convolutional_layer(nb_filter=16, input_shape=imgs_dim))
    model.add(BatchNormalization(axis=-1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=16))
    model.add(BatchNormalization(axis=-1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(_convolutional_layer(nb_filter=32))
    model.add(BatchNormalization(axis=-1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=32))
    model.add(BatchNormalization(axis=-1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=32))
    model.add(BatchNormalization(axis=-1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(_convolutional_layer(nb_filter=64))
    model.add(BatchNormalization(axis=-1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=64))
    model.add(BatchNormalization(axis=-1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=64))
    model.add(BatchNormalization(axis=-1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(_convolutional_layer(nb_filter=128))
    model.add(BatchNormalization(axis=-1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=128))
    model.add(BatchNormalization(axis=-1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=128))
    model.add(BatchNormalization(axis=-1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(_convolutional_layer(nb_filter=256))
    model.add(BatchNormalization(axis=-1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=256))
    model.add(BatchNormalization(axis=-1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=256))
    model.add(BatchNormalization(axis=-1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.5))

    model.add(Flatten())
    model.add(_dense_layer(output_dim=PENULTIMATE_SIZE))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=W_INIT))

    if compile_:
        model.add(Dropout(rate=0.5))
        model.add(_dense_layer(output_dim=SOFTMAX_SIZE))
        model.add(BatchNormalization())
        model.add(Activation(activation='softmax'))
        return compile_model(model)

    return model


def _convolutional_layer(nb_filter, input_shape=None):
    if input_shape:
        return _first_convolutional_layer(nb_filter, input_shape)
    else:
        return _intermediate_convolutional_layer(nb_filter)


def _first_convolutional_layer(nb_filter, input_shape):
    return Conv2D(nb_filter, (KERNAL_SIZE ,KERNAL_SIZE), input_shape=input_shape,
        padding='same', kernel_initializer=W_INIT, kernel_regularizer=l2(l=L2_REG))


def _intermediate_convolutional_layer(nb_filter):
    return Conv2D(nb_filter,(KERNAL_SIZE, KERNAL_SIZE), padding='same',
        kernel_initializer=W_INIT, kernel_regularizer=l2(l=L2_REG))


def _dense_layer(output_dim):
    return Dense(units=output_dim, kernel_regularizer=l2(l=L2_REG), kernel_initializer=W_INIT)


def compile_model(model):
    adam = Adam(lr=0.000074)
    #adam = Adam(lr=0.0007)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=adam,
        metrics=['accuracy'])
    return model


def load_trained_cnn_feature_maps_layer(model_path):
    return _load_trained_cnn_layer(model_path, LAST_FEATURE_MAPS_LAYER)


def load_trained_cnn_penultimate_layer(model_path):
    return _load_trained_cnn_layer(model_path, PENULTIMATE_LAYER)


def load_trained_cnn_softmax_layer(model_path):
    return _load_trained_cnn_layer(model_path, SOFTMAX_LAYER)


def _load_trained_cnn_layer(model_path, layer_index):
    model = load_model(model_path)
    dense_output = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[layer_index].output])
    # output in test mode = 0
    return lambda X: dense_output([X, 0])[0]


if __name__ == '__main__':
    #gpu_count = len([dev for dev in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if len(dev.strip()) > 0])
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #sess = tf.Session(config = config)
    _train_model()
