from keras.layers.core import Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K

def transition_layer(ip, nb_filter, block_idx, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout and Maxpooling2D
    Args:
        ip: keras tensor
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay), name = 'dense_block_{}_trans_bn'.format(block_idx))(ip)
    x = Activation('relu', name = 'dense_block_{}_trans_relu'.format(block_idx))(x)
    x = Convolution2D(int(nb_filter * compression), 1, 1, init="he_uniform", border_mode="same", bias=False,
                      W_regularizer=l2(weight_decay),name = 'dense_block_{}_trans_conv2d'.format(block_idx) )(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2),name = 'dense_block_{}_trans_'.format(block_idx))(x)

    return x
