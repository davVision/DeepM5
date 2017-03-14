# Keras imports
from keras.models import Model, Sequential
from keras.layers import Input
import keras.backend as K
from keras.layers.convolutional import Convolution2D
from layers.dense_block import dense_block
from layers.transition_layer import transition_layer
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation
from keras.layers.pooling import GlobalAveragePooling2D


# Paper: https://arxiv.org/pdf/1608.06993.pdf


def build_densenetFCN(img_shape=(3, 224, 224), n_classes=1000, weight_decay=1E-4,
                load_pretrained=False, freeze_layers_from='base_model',
                path_weights=None):

    depth = 40
    nb_dense_block = 3
    growth_rate = 12
    nb_filter = -1
    bottleneck = False
    reduction = 0.0
    dropout_rate = None
    verbose = True

    model_input = Input(shape=img_shape)

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    assert (depth - 4) % nb_dense_block == 0, "Depth must be 3 N + 4"

    assert reduction <= 1.0 and reduction >= 0, "Reduction must lie between 0.0 and 1.0"

    # layers in each dense block
    nb_layers = int((depth - 4) / nb_dense_block)

    if bottleneck:
        nb_layers = int(nb_layers // 2)

    # compute initial nb_filter if -1, else accept users initial nb_filter
    if nb_filter <= 0:
        nb_filter = 2 * growth_rate

    # compute compression factor
    compression = 1.0 - reduction

    # Initial convolution
    x = Convolution2D(nb_filter, 3, 3, init="he_uniform", border_mode="same", name="initial_conv2D", bias=False,
                      W_regularizer=l2(weight_decay))(model_input)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, block_idx, bottleneck=bottleneck,
                                   dropout_rate=dropout_rate, weight_decay=weight_decay)
        # add transition_block
        x = transition_layer(x, nb_filter, block_idx, compression=compression, dropout_rate=dropout_rate,
                             weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    # The last dense_block does not have a transition_block
    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, nb_dense_block, bottleneck=bottleneck,
                               dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    base_model = Model(input=model_input, output=x)

    x = Dense(n_classes, activation='softmax', W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(x)

    model = Model(input=base_model.input, output=x)


    # Freeze some layers
    if freeze_layers_from is not None:
        if freeze_layers_from == 'base_model':
            print ('   Freezing base model layers')
            for layer in base_model.layers:
                layer.trainable = False
        else:
            for i, layer in enumerate(model.layers):
                print(i, layer.name)
            print ('   Freezing from layer 0 to ' + str(freeze_layers_from))
            for layer in model.layers[:freeze_layers_from]:
               layer.trainable = False
            for layer in model.layers[freeze_layers_from:]:
               layer.trainable = True


    if verbose:
        if bottleneck and not reduction:
            print("Bottleneck DenseNet-B-%d-%d created." % (depth, growth_rate))
        elif not bottleneck and reduction > 0.0:
            print("DenseNet-C-%d-%d with %0.1f compression created." % (depth, growth_rate, compression))
        elif bottleneck and reduction > 0.0:
            print("Bottleneck DenseNet-BC-%d-%d with %0.1f compression created." % (depth, growth_rate, compression))
        else:
            print("DenseNet-%d-%d created." % (depth, growth_rate))

    return model
