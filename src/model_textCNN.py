import keras
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.models import Model


def CNN_simple(MAX_SEQUENCE_LENGTH, embedding_layer, filter_num, kernal_size, label_dim, dropout_CNN, dropout_full_con):
    """
    Simple CNN network for text classification

    Arguments:
    MAX_SEQUENCE_LENGTH: Positive integer, max sequence #
    embedding_layer -- word embedding layer created by keras.layers.embedding
    filter_num: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)
    kernal_size: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window
    label_dim: # of labels in the classification

    Return:
    model -- a model instance in Keras
    """
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    l_cov1 = Conv1D(filter_num, kernal_size, activation='relu')(embedded_sequences)
    l_pool1 = MaxPooling1D(kernal_size)(l_cov1)
    l_pool1 = Dropout(dropout_CNN)(l_pool1)

    l_cov2 = Conv1D(filter_num, kernal_size, activation='relu')(l_pool1)
    l_pool2 = MaxPooling1D(kernal_size)(l_cov2)
    l_pool2 = Dropout(dropout_CNN)(l_pool2)

    l_cov3 = Conv1D(filter_num, kernal_size, activation='relu')(l_pool2)
    l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
    l_pool3 = Dropout(dropout_CNN)(l_pool3)

    l_flat = Flatten()(l_pool3)
    l_dense = Dense(filter_num, activation='relu')(l_flat)
    l_dense = Dropout(dropout_full_con)(l_dense)
    preds = Dense(label_dim, activation='softmax')(l_dense)

    model = Model(sequence_input, preds)

    return model


def CNN_complex(MAX_SEQUENCE_LENGTH, embedding_layer, filter_num, kernal_size, label_dim, dropout_CNN, dropout_full_con):
    """
    Complex CNN network for text classification

    Arguments:
    MAX_SEQUENCE_LENGTH: Positive integer, max sequence #
    embedding_layer -- word embedding layer created by keras.layers.embedding
    filter_num: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)
    kernal_size: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window
    label_dim: # of labels in the classification

    Return:
    model -- a model instance in Keras
    """
    convs = []

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    for ks in kernal_size:
        l_conv = Conv1D(int(filter_num), ks, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(int(MAX_SEQUENCE_LENGTH-ks+1))(l_conv)
        l_pool = Dropout(dropout_CNN)(l_pool)
        convs.append(l_pool)

    # l_merge = Merge(mode='concat', concat_axis=1)(convs)
    l_merge = keras.layers.Concatenate(axis=1)(convs)

    l_flat = Flatten()(l_merge)
    l_dense = Dense(1024, activation='relu')(l_flat)
    l_dense = Dropout(dropout_full_con)(l_dense)
    preds = Dense(label_dim, activation='softmax')(l_dense)

    model = Model(sequence_input, preds)

    return model

