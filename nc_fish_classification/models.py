import numpy as np

from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D, Convolution2D, AveragePooling2D
from keras.layers import Input, Activation, Lambda
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from keras.applications.resnet50 import identity_block, conv_block
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras import optimizers


px_mean = np.array([123.68, 116.779, 103.939]).reshape((3,1,1))


def preprocess(x):
    x = x - px_mean
    return x[:, ::-1] # reverse axis bgr->rgb


######################################### ----- VGG16 MODEL ----- #####################################################


class Vgg16BN():
    """
    The VGG16 Imagenet model with Batch Normalization for the Dense Layers
    """
    def __init__(self, size=(224, 224), n_classes=2, lr=0.001, batch_size=64, dropout=0.5):
        self.weights_file = 'vgg16.h5'  # download from: http://www.platform.ai/models/
        self.size = size
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout

    def build(self):
        """
        Constructs vgg16 model from keras with batch normalization layers;
        Returns stacked model
        """
        model = self.model = Sequential()
        model.add(Lambda(preprocess, input_shape=(3,)+self.size))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(1000, activation='softmax'))

        model.load_weights(self.weights_file)
        model.pop(); model.pop(); model.pop()

        for layer in model.layers:
            layer.trainable = False

        model.add(Dense(4096, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout))
        model.add(Dense(4096, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout))
        model.add(Dense(self.n_classes, activation='softmax'))

        optimizer = optimizers.Adadelta(lr=self.lr)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
        return model

    def get_datagen(self, aug=False):
        if aug:
            return ImageDataGenerator(rotation_range=10, width_shift_range=0.05, zoom_range=0.05,
                                      channel_shift_range=10, height_shift_range=0.05, shear_range=0.05,
                                      horizontal_flip=True)
        return ImageDataGenerator()

    def fit_val(self, trn_path, val_path, nb_trn_samples, nb_val_samples, nb_epoch=1, callbacks=[], aug=False):
        """Custom fit method for training with validation data and option for data augmentation"""
        train_datagen = self.get_datagen(aug=aug)
        trn_gen = train_datagen.flow_from_directory(trn_path, target_size=self.size, batch_size=self.batch_size,
                                                    class_mode='categorical', shuffle=True)
        val_gen = ImageDataGenerator().flow_from_directory(val_path, target_size=self.size, batch_size=self.batch_size,
                                                           class_mode='categorical', shuffle=True)
        self.model.fit_generator(trn_gen, samples_per_epoch=nb_trn_samples, nb_epoch=nb_epoch, verbose=2,
                                 validation_data=val_gen, nb_val_samples=nb_val_samples, callbacks=callbacks)


    def fit_full(self, trn_path, nb_trn_samples, nb_epoch=1, callbacks=[], aug=False):
        """Custom fit method for training without validation data and option for data augmentation"""
        train_datagen = self.get_datagen(aug=aug)
        trn_gen = train_datagen.flow_from_directory(trn_path, target_size=self.size, batch_size=self.batch_size,
                                                    class_mode='categorical', shuffle=True)
        self.model.fit_generator(trn_gen, samples_per_epoch=nb_trn_samples, nb_epoch=nb_epoch, verbose=2,
                callbacks=callbacks)

    def test(self, test_path, nb_test_samples, aug=False):
        """Custom prediction method with option for data augmentation"""
        test_datagen = self.get_datagen(aug=aug)
        test_gen = test_datagen.flow_from_directory(test_path, target_size=self.size, batch_size=self.batch_size,
                                                    class_mode=None, shuffle=False)
        return self.model.predict_generator(test_gen, val_samples=nb_test_samples), test_gen.filenames


###################################### ----- INCEPTION V3 MODEL ----- #################################################


def inception_preprocess(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


class Inception():
    """The VGG 16 Imagenet model with Batch Normalization for the Dense Layers"""
    def __init__(self, size=(299, 299), n_classes=2, lr=0.001, batch_size=64):
        self.size = size
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size

    def build(self):
        """
            Loads preconstructed inception model from keras without top classification layer;
            Stacks custom classification layer on top;
        """
        img_input = Input(shape=(3, self.size[0], self.size[1]))
        inception = InceptionV3(include_top=False, weights='imagenet', input_tensor=img_input)

        for layer in inception.layers:
            layer.trainable = False

        output = inception.output
        output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
        output = Flatten(name='flatten')(output)
        output = Dense(self.n_classes, activation='softmax', name='predictions')(output)

        model = self.model = Model(inception.input, output)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9, decay=0.0, nesterov=True),
                      metrics=["accuracy"])
        return model

    def get_datagen(self, aug=False):
        if aug:
            return ImageDataGenerator(rotation_range=10, width_shift_range=0.05, zoom_range=0.05,
                                      channel_shift_range=10, height_shift_range=0.05, shear_range=0.05,
                                      horizontal_flip=True, preprocessing_function=inception_preprocess)
        return ImageDataGenerator(preprocessing_function=inception_preprocess)

    def fit_val(self, trn_path, val_path, nb_trn_samples, nb_val_samples, nb_epoch=1, callbacks=[], aug=False):
        """Custom fit method for training with validation data and option for data augmentation"""
        train_datagen = self.get_datagen(aug=aug)
        val_datagen = self.get_datagen(aug=False)

        trn_gen = train_datagen.flow_from_directory(trn_path, target_size=self.size, batch_size=self.batch_size,
                                                    class_mode='categorical', shuffle=True)
        val_gen = val_datagen.flow_from_directory(val_path, target_size=self.size, batch_size=self.batch_size,
                                                  class_mode='categorical', shuffle=True)
        self.model.fit_generator(trn_gen, samples_per_epoch=nb_trn_samples, nb_epoch=nb_epoch, verbose=2,
                                 validation_data=val_gen, nb_val_samples=nb_val_samples, callbacks=callbacks)


    def fit_full(self, trn_path, nb_trn_samples, nb_epoch=1, callbacks=[], aug=False):
        """Custom fit method for training without validation data and option for data augmentation"""
        train_datagen = self.get_datagen(aug=aug)

        trn_gen = train_datagen.flow_from_directory(trn_path, target_size=self.size, batch_size=self.batch_size,
                                                    class_mode='categorical', shuffle=True)
        self.model.fit_generator(trn_gen, samples_per_epoch=nb_trn_samples, nb_epoch=nb_epoch, verbose=2,
                                 callbacks=callbacks)

    def test(self, test_path, nb_test_samples, aug=False):
        """Custom prediction method with option for data augmentation"""
        test_datagen = self.get_datagen(aug=aug)
        test_gen = test_datagen.flow_from_directory(test_path, target_size=self.size, batch_size=self.batch_size,
                                                    class_mode=None, shuffle=False)
        return self.model.predict_generator(test_gen, val_samples=nb_test_samples), test_gen.filenames


####################################### ----- RESNET 50 MODEL ----- ###################################################



class Resnet50():
    """The Resnet 50 Imagenet model"""

    def __init__(self, size=(224, 224), n_classes=2, lr=0.001, batch_size=64, dropout=0.2):
        self.weights_file = 'resnet_nt.h5'  # download from: http://www.platform.ai/models/
        self.size = size
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout

    def get_base(self):
        """Gets base architecture of Resnet 50 Model"""
        input_shape = (3,) + self.size
        img_input = Input(shape=input_shape)
        bn_axis = 1

        x = Lambda(preprocess)(img_input)
        x = ZeroPadding2D((3, 3))(x)
        x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')

        for n in ['b','c','d']:
            x = identity_block(x, 3, [128, 128, 512], stage=3, block=n)

        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')

        for n in ['b','c','d', 'e', 'f']:
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block=n)

        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        self.img_input = img_input
        self.model = Model(self.img_input, x)
        convert_all_kernels_in_model(self.model)
        self.model.load_weights(self.weights_file)

    def build(self):
        """Builds the model and stacks global average pooling layer on top"""
        self.get_base()
        self.model.layers.pop()

        for layer in self.model.layers:
            layer.trainable = False

        m = GlobalAveragePooling2D()(self.model.layers[-1].output)
        m = Dropout(self.dropout)(m)
        Dense(self.n_classes, activation='softmax')
        m = Dense(self.n_classes, activation='softmax')(m)
        self.model = Model(self.model.input, m)
        self.model.compile(optimizer=optimizers.Adam(lr=self.lr), loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model

    def build_precomputed(self):
        """
            Builds the model based on output of last layer of base architecture;
            Used for training with bottleneck features;
        """
        self.get_base()
        self.model = Sequential(
            [GlobalAveragePooling2D(input_shape=self.model.layers[-1].output_shape[1:]),
            Dropout(self.dropout),
            Dense(self.n_classes, activation="softmax"),
            ])
        self.model.compile(optimizer=optimizers.Adam(lr=self.lr), loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model

    def get_datagen(self, aug=False):
        if aug:
            return ImageDataGenerator(rotation_range=10, width_shift_range=0.05, zoom_range=0.05,
                                      channel_shift_range=10, height_shift_range=0.05, shear_range=0.05,
                                      horizontal_flip=True)
        return ImageDataGenerator()

    def fit_val(self, trn_path, val_path, nb_trn_samples, nb_val_samples, nb_epoch=1, callbacks=[], aug=False):
        """Custom fit method for training with validation data and option for data augmentation"""
        train_datagen = self.get_datagen(aug=aug)
        val_datagen = self.get_datagen(aug=False)

        trn_gen = train_datagen.flow_from_directory(trn_path, target_size=self.size, batch_size=self.batch_size,
                                                    class_mode='categorical', shuffle=True)
        val_gen = val_datagen.flow_from_directory(val_path, target_size=self.size, batch_size=self.batch_size,
                                                  class_mode='categorical', shuffle=True)
        self.model.fit_generator(trn_gen, samples_per_epoch=nb_trn_samples, nb_epoch=nb_epoch, verbose=2,
                validation_data=val_gen, nb_val_samples=nb_val_samples, callbacks=callbacks)

    def fit_full(self, trn_path, nb_trn_samples, nb_epoch=1, callbacks=[], aug=False):
        """Custom fit method for training without validation data and option for data augmentation"""
        train_datagen = self.get_datagen(aug=aug)

        trn_gen = train_datagen.flow_from_directory(trn_path, target_size=self.size, batch_size=self.batch_size,
                                                    class_mode='categorical', shuffle=True)
        self.model.fit_generator(trn_gen, samples_per_epoch=nb_trn_samples, nb_epoch=nb_epoch, verbose=2,
                                 callbacks=callbacks)

    def test(self, test_path, nb_test_samples, aug=False):
        """Custom prediction method with option for data augmentation"""
        test_datagen = self.get_datagen(aug=aug)
        test_gen = test_datagen.flow_from_directory(test_path, target_size=self.size, batch_size=self.batch_size,
                                                    class_mode=None, shuffle=False)
        return self.model.predict_generator(test_gen, val_samples=nb_test_samples), test_gen.filenames
