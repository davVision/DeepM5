from keras.layers.core import  Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers import merge
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K


def conv_block(ip, nb_filter, block_idx, layer_idx, bottleneck=False, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, Relu 3x3, Conv2D, optional bottleneck block and dropout
    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay),name = 'dense_block_{}_layer_{}_bn'.format(block_idx, layer_idx)  )(ip)
    x = Activation('relu', name = 'dense_block_{}_layer_{}_relu'.format(block_idx, layer_idx) )(x)

    if bottleneck:
        inter_channel = nb_filter * 4 # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua

        x = Convolution2D(inter_channel, 1, 1, init='he_uniform', border_mode='same', bias=False,
                          W_regularizer=l2(weight_decay), name = 'dense_block_{}_layer_{}_bottle_conv2d'.format(block_idx, layer_idx) )(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay), name = 'dense_block_{}_layer_{}_bottle_bn'.format(block_idx, layer_idx) )(x)
        x = Activation('relu', name = 'dense_block_{}_bottle_relu'.format(block_idx, layer_idx))(x)

    x = Convolution2D(nb_filter, 3, 3, init="he_uniform", border_mode="same", bias=False,
                      W_regularizer=l2(weight_decay), name = 'dense_block_{}_layer_{}_conv2d'.format(block_idx, layer_idx))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x



def dense_block(x, nb_layers, nb_filter, growth_rate, block_idx, bottleneck=False, dropout_rate=None, weight_decay=1E-4):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with nb_layers of conv_block appended
    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    feature_list = [x]

    for i in range(nb_layers):
        x = conv_block(x, growth_rate, block_idx, i, bottleneck, dropout_rate, weight_decay)
        feature_list.append(x)
        x = merge(feature_list, mode='concat', concat_axis=concat_axis)
        nb_filter += growth_rate

    return x, nb_filter
