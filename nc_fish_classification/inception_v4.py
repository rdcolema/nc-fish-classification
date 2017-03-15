import warnings

from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers import Input, merge, Dropout, Dense, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import optimizers

from keras import backend as K

from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file


#########################################################################################
# Implements the Inception Network v4 (http://arxiv.org/pdf/1602.07261v1.pdf) in Keras. #
#########################################################################################



def conv2d_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1), bias=False):
    """
    Utility function to apply conv + BN. 
    (Slightly modified from https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py)
    """
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1
    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      border_mode=border_mode,
                      bias=bias)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x


def block_inception_a(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 96, 1, 1)

    branch_1 = conv2d_bn(input, 64, 1, 1)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3)

    branch_2 = conv2d_bn(input, 64, 1, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3)

    branch_3 = AveragePooling2D((3,3), strides=(1,1), border_mode='same')(input)
    branch_3 = conv2d_bn(branch_3, 96, 1, 1)

    x = merge([branch_0, branch_1, branch_2, branch_3], mode='concat', concat_axis=channel_axis)
    return x


def block_reduction_a(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 384, 3, 3, subsample=(2,2), border_mode='valid')

    branch_1 = conv2d_bn(input, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 224, 3, 3)
    branch_1 = conv2d_bn(branch_1, 256, 3, 3, subsample=(2,2), border_mode='valid')

    branch_2 = MaxPooling2D((3,3), strides=(2,2), border_mode='valid')(input)

    x = merge([branch_0, branch_1, branch_2], mode='concat', concat_axis=channel_axis)
    return x


def block_inception_b(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 384, 1, 1)

    branch_1 = conv2d_bn(input, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 224, 1, 7)
    branch_1 = conv2d_bn(branch_1, 256, 7, 1)

    branch_2 = conv2d_bn(input, 192, 1, 1)
    branch_2 = conv2d_bn(branch_2, 192, 7, 1)
    branch_2 = conv2d_bn(branch_2, 224, 1, 7)
    branch_2 = conv2d_bn(branch_2, 224, 7, 1)
    branch_2 = conv2d_bn(branch_2, 256, 1, 7)

    branch_3 = AveragePooling2D((3,3), strides=(1,1), border_mode='same')(input)
    branch_3 = conv2d_bn(branch_3, 128, 1, 1)

    x = merge([branch_0, branch_1, branch_2, branch_3], mode='concat', concat_axis=channel_axis)
    return x


def block_reduction_b(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 192, 1, 1)
    branch_0 = conv2d_bn(branch_0, 192, 3, 3, subsample=(2, 2), border_mode='valid')

    branch_1 = conv2d_bn(input, 256, 1, 1)
    branch_1 = conv2d_bn(branch_1, 256, 1, 7)
    branch_1 = conv2d_bn(branch_1, 320, 7, 1)
    branch_1 = conv2d_bn(branch_1, 320, 3, 3, subsample=(2,2), border_mode='valid')

    branch_2 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(input)

    x = merge([branch_0, branch_1, branch_2], mode='concat', concat_axis=channel_axis)
    return x


def block_inception_c(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 256, 1, 1)

    branch_1 = conv2d_bn(input, 384, 1, 1)
    branch_10 = conv2d_bn(branch_1, 256, 1, 3)
    branch_11 = conv2d_bn(branch_1, 256, 3, 1)
    branch_1 = merge([branch_10, branch_11], mode='concat', concat_axis=channel_axis)


    branch_2 = conv2d_bn(input, 384, 1, 1)
    branch_2 = conv2d_bn(branch_2, 448, 3, 1)
    branch_2 = conv2d_bn(branch_2, 512, 1, 3)
    branch_20 = conv2d_bn(branch_2, 256, 1, 3)
    branch_21 = conv2d_bn(branch_2, 256, 3, 1)
    branch_2 = merge([branch_20, branch_21], mode='concat', concat_axis=channel_axis)

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    branch_3 = conv2d_bn(branch_3, 256, 1, 1)

    x = merge([branch_0, branch_1, branch_2, branch_3], mode='concat', concat_axis=channel_axis)
    return x


def inception_v4_base(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
    net = conv2d_bn(input, 32, 3, 3, subsample=(2,2), border_mode='valid')
    net = conv2d_bn(net, 32, 3, 3, border_mode='valid')
    net = conv2d_bn(net, 64, 3, 3)

    branch_0 = MaxPooling2D((3,3), strides=(2,2), border_mode='valid')(net)

    branch_1 = conv2d_bn(net, 96, 3, 3, subsample=(2,2), border_mode='valid')

    net = merge([branch_0, branch_1], mode='concat', concat_axis=channel_axis)

    branch_0 = conv2d_bn(net, 64, 1, 1)
    branch_0 = conv2d_bn(branch_0, 96, 3, 3, border_mode='valid')

    branch_1 = conv2d_bn(net, 64, 1, 1)
    branch_1 = conv2d_bn(branch_1, 64, 1, 7)
    branch_1 = conv2d_bn(branch_1, 64, 7, 1)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3, border_mode='valid')

    net = merge([branch_0, branch_1], mode='concat', concat_axis=channel_axis)

    branch_0 = conv2d_bn(net, 192, 3, 3, subsample=(2,2), border_mode='valid')
    branch_1 = MaxPooling2D((3,3), strides=(2,2), border_mode='valid')(net)

    net = merge([branch_0, branch_1], mode='concat', concat_axis=channel_axis)

    # 35 x 35 x 384
    # 4 x Inception-A blocks
    for idx in range(4):
        net = block_inception_a(net)

    # 35 x 35 x 384
    # Reduction-A block
    net = block_reduction_a(net)

    # 17 x 17 x 1024
    # 7 x Inception-B blocks
    for idx in range(7):
        net = block_inception_b(net)

    # 17 x 17 x 1024
    # Reduction-B block
    net = block_reduction_b(net)

    # 8 x 8 x 1536
    # 3 x Inception-C blocks
    for idx in range(3):
        net = block_inception_c(net)

    return net


def inception_preprocess(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


class InceptionV4:
    '''
    Creates the inception v4 network

    Args:
        num_classes: number of classes
        dropout_keep_prob: float, the fraction to keep before final layer.
    
    Returns: 
        logits: the logits outputs of the model.
    '''

    def __init__(self, num_classes, dropout_keep_prob, weights, size=(299, 299), batch_size=32):
        self.num_classes = num_classes
        self.dropout_keep_prob = dropout_keep_prob
        self.weights = weights
        self.size = size
        self.batch_size = batch_size

    def build(self):
        # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
        if K.image_dim_ordering() == 'th':
            inputs = Input((3, self.size[0], self.size[1]))
        else:
            inputs = Input((self.size[0], self.size[1], 3))

        # Make inception base
        net = inception_v4_base(inputs)

        # Final pooling and prediction
        # 8 x 8 x 1536
        net = AveragePooling2D((8,8), border_mode='valid')(net)

        # 1 x 1 x 1536
        net = Dropout(self.dropout_keep_prob)(net)
        net = Flatten()(net)

        # 1536
        predictions = Dense(output_dim=1001, activation='softmax')(net)

        inception = Model(inputs, predictions, name='inception_v4')

        # load weights
        weights_path = "inception_v4.h5"
        inception.load_weights(weights_path, by_name=True)
        print("Loaded imagenet weights")

        inception.layers.pop()

        for layer in inception.layers:
            layer.trainable = False

        output = inception.output
        output = Dense(self.num_classes, activation='softmax', name='predictions')(output)

        model = self.model = Model(inception.input, output)

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=1., momentum=0.9, decay=0.01, nesterov=True),
                      metrics=["accuracy"])

        return model

    def get_datagen(self, aug=False):
        if aug:
            return ImageDataGenerator(rotation_range=10, width_shift_range=0.05, zoom_range=0.05,
                                      channel_shift_range=10, height_shift_range=0.05, shear_range=0.05,
                                      horizontal_flip=True, preprocessing_function=inception_preprocess)
        return ImageDataGenerator(preprocessing_function=inception_preprocess)

    def fit_val(self, trn_path, val_path, nb_trn_samples, nb_val_samples, nb_epoch=1, callbacks=[], aug=False):
        """
        Custom fit method for training with validation data and option for data augmentation"
        """
        train_datagen = self.get_datagen(aug=aug)
        val_datagen = self.get_datagen(aug=False)

        trn_gen = train_datagen.flow_from_directory(trn_path, target_size=self.size, batch_size=self.batch_size,
                                                    class_mode='categorical', shuffle=True)
        val_gen = val_datagen.flow_from_directory(val_path, target_size=self.size, batch_size=self.batch_size,
                                                  class_mode='categorical', shuffle=True)
        self.model.fit_generator(trn_gen, samples_per_epoch=nb_trn_samples, nb_epoch=nb_epoch, verbose=2,
                                 validation_data=val_gen, nb_val_samples=nb_val_samples, callbacks=callbacks)


    def fit_full(self, trn_path, nb_trn_samples, nb_epoch=1, callbacks=[], aug=False):
        """
        Custom fit method for training without validation data and option for data augmentation
        """
        train_datagen = self.get_datagen(aug=aug)

        trn_gen = train_datagen.flow_from_directory(trn_path, target_size=self.size, batch_size=self.batch_size,
                                                    class_mode='categorical', shuffle=True)
        self.model.fit_generator(trn_gen, samples_per_epoch=nb_trn_samples, nb_epoch=nb_epoch, verbose=2,
                                 callbacks=callbacks)

    def test(self, test_path, nb_test_samples, aug=False):
        """
        Custom prediction method with option for data augmentation
        """
        test_datagen = self.get_datagen(aug=aug)
        test_gen = test_datagen.flow_from_directory(test_path, target_size=self.size, batch_size=self.batch_size,
                                                    class_mode=None, shuffle=False)
        return self.model.predict_generator(test_gen, val_samples=nb_test_samples), test_gen.filenames
