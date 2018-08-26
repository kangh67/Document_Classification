import numpy as np
import sys
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input, LSTM, GRU, Bidirectional, TimeDistributed, Activation, Dot, Reshape
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.layers.embeddings import Embedding


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    # if K.backend() == 'tensorflow':
    return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    # else:
    # return K.dot(x, kernel)


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.

    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.

    Note: The layer has been tested with Keras 2.0.6

    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def RNN_simple(MAX_SEQUENCE_LENGTH, embedding_layer, RNN_gate_type, RNN_units, dropout, label_dim):
    """
    Simple RNN network for text classification

    Arguments:
    MAX_SEQUENCE_LENGTH: Positive integer, max sequence #
    embedding_layer -- word embedding layer created by keras.layers.embedding
    RNN_gate_type: String, LSTM or GRU
    RNN_units: Positive integer, dimensionality of the output space
    dropout: value between [0, 1], no dropout when dropout = 0
    label_dim: # of labels in the classification

    Return:
    model -- a model instance in Keras
    """
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    if RNN_gate_type == 'LSTM':
        l_RNN = Bidirectional(LSTM(RNN_units, dropout=dropout))(embedded_sequences)
    elif RNN_gate_type == 'GRU':
        l_RNN = Bidirectional(GRU(RNN_units, dropout=dropout))(embedded_sequences)
    preds = Dense(label_dim, activation='softmax')(l_RNN)

    model = Model(sequence_input, preds)

    return model


def RNN_attention(MAX_SEQUENCE_LENGTH, embedding_layer, RNN_gate_type, RNN_units, dropout, label_dim):
    """
    RNN network with attention

    Arguments:
    MAX_SEQUENCE_LENGTH: Positive integer, max sequence #
    embedding_layer -- word embedding layer created by keras.layers.embedding
    RNN_gate_type: String, LSTM or GRU
    RNN_units: Positive integer, dimensionality of the output space
    dropout: value between [0, 1], no dropout when dropout = 0
    label_dim: # of labels in the classification

    Return:
    model -- a model instance in Keras
    """
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    if RNN_gate_type == 'LSTM':
        l_RNN = Bidirectional(LSTM(RNN_units, dropout=dropout, return_sequences=True))(embedded_sequences)
    elif RNN_gate_type == 'GRU':
        l_RNN = Bidirectional(GRU(RNN_units, dropout=dropout, return_sequences=True))(embedded_sequences)
    l_att = AttentionWithContext()(l_RNN)
    preds = Dense(label_dim, activation='softmax')(l_att)

    model = Model(sequence_input, preds)

    return model


def hierachical_network(MAX_SENTS, MAX_SENT_LENGTH, embedding_layer, RNN_gate_type, RNN_units, dropout, label_dim):
    """
    Hierachical network for document classification, no attention applied

    Arguments:
    MAX_SENTS: Positive integer, max # of sentences in a document
    MAX_SENT_LENGTH: ositive integer, max # of words in a sentence
    embedding_layer -- word embedding layer created by keras.layers.embedding
    emb_trainable: Boolean, Fix pre-trained word embedding (True) or make it trainable (False)
    RNN_gate_type: String, LSTM or GRU
    RNN_units: Positive integer, dimensionality of the output space
    dropout: value between [0, 1], no dropout when dropout = 0
    label_dim: # of labels in the classification

    Return:
    model -- a model instance in Keras
    """
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    # embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index, emb_trainable)
    embedded_sequences = embedding_layer(sentence_input)
    if RNN_gate_type == 'LSTM':
        l_lstm = Bidirectional(LSTM(RNN_units, dropout=dropout))(embedded_sequences)
    elif RNN_gate_type == 'GRU':
        l_lstm = Bidirectional(GRU(RNN_units, dropout=dropout))(embedded_sequences)
    else:
        sys.exit("ERROR: RNN_gate_type is incorrectly assigned.")
    sentEncoder = Model(sentence_input, l_lstm)

    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    if RNN_gate_type == 'LSTM':
        l_lstm_sent = Bidirectional(LSTM(RNN_units, dropout=dropout))(review_encoder)
    elif RNN_gate_type == 'GRU':
        l_lstm_sent = Bidirectional(GRU(RNN_units, dropout=dropout))(review_encoder)
    else:
        sys.exit("ERROR: RNN_gate_type is incorrectly assigned.")
    preds = Dense(label_dim, activation='softmax')(l_lstm_sent)
    model = Model(review_input, preds)

    return model


def hierachical_network_attention(MAX_SENTS, MAX_SENT_LENGTH, embedding_layer, RNN_gate_type, RNN_units, dropout,
                                  label_dim):
    """
    Hierachical network for document classification, attention applied

    Arguments:
    MAX_SENTS: Positive integer, max # of sentences in a document
    MAX_SENT_LENGTH: ositive integer, max # of words in a sentence
    embedding_layer -- word embedding layer created by keras.layers.embedding
    emb_trainable: Boolean, Fix pre-trained word embedding (True) or make it trainable (False)
    RNN_gate_type: String, LSTM or GRU
    RNN_units: Positive integer, dimensionality of the output space
    dropout: value between [0, 1], no dropout when dropout = 0
    label_dim: # of labels in the classification

    Return:
    model -- a model instance in Keras
    """
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    # embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index, emb_trainable)
    embedded_sequences = embedding_layer(sentence_input)
    if RNN_gate_type == 'LSTM':
        l_lstm = Bidirectional(LSTM(RNN_units, return_sequences=True, dropout=dropout))(embedded_sequences)
    elif RNN_gate_type == 'GRU':
        l_lstm = Bidirectional(GRU(RNN_units, return_sequences=True, dropout=dropout))(embedded_sequences)
    else:
        sys.exit("ERROR: RNN_gate_type is incorrectly assigned.")
    l_lstm = AttentionWithContext()(l_lstm)  # attention layer
    sentEncoder = Model(sentence_input, l_lstm)
    # print(sentEncoder.summary())

    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    if RNN_gate_type == 'LSTM':
        l_lstm_sent = Bidirectional(LSTM(RNN_units, return_sequences=True, dropout=dropout))(review_encoder)
    elif RNN_gate_type == 'GRU':
        l_lstm_sent = Bidirectional(GRU(RNN_units, return_sequences=True, dropout=dropout))(review_encoder)
    else:
        sys.exit("ERROR: RNN_gate_type is incorrectly assigned.")
    l_lstm_sent = AttentionWithContext()(l_lstm_sent)  # attention layer
    preds = Dense(label_dim, activation='softmax')(l_lstm_sent)
    model = Model(review_input, preds)

    return model


def one_step_attention(a, densor1, densor2, activator, dotor):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.

    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)

    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e.
    e = densor1(a)
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies.
    energies = densor2(e)
    # Use "activator" on "energies" to compute the attention weights "alphas"
    alphas = activator(energies)
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell
    context = dotor([alphas, a])

    return context


def hierachical_attention_network(MAX_SENTS, MAX_SENT_LENGTH, embedding_layer, RNN_gate_type, RNN_units, label_dim):
    """
    Hierachical_attention_network, following the paper "Hierarchical Attention Networks for Document Classification"

    Arguments:
    MAX_SENTS: Positive integer, max # of sentences in a document
    MAX_SENT_LENGTH: ositive integer, max # of words in a sentence
    embedding_layer -- word embedding layer created by keras.layers.embedding
    RNN_gate_type: String, LSTM or GRU
    RNN_units: Positive integer, dimensionality of the output space
    label_dim: # of labels in the classification

    Returns:
    model -- a model instance in Keras
    """
    # Attention layer for word encodings
    densor1_1 = Dense(10, activation="tanh")
    densor1_2 = Dense(1, activation="relu")
    activator1 = Activation(activation="softmax",
                            name='attention_weights')  # We are using a custom softmax(axis = 1) loaded in this notebook
    dotor1 = Dot(axes=1)

    # Attention layer for sentence encodings
    densor2_1 = Dense(10, activation="tanh")
    densor2_2 = Dense(1, activation="relu")
    activator2 = Activation(activation="softmax",
                            name='attention_weights')  # We are using a custom softmax(axis = 1) loaded in this notebook
    dotor2 = Dot(axes=1)

    # Build word encoding and word attention
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    # embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index, emb_trainable)
    embedded_sequences = embedding_layer(sentence_input)
    if RNN_gate_type == 'LSTM':
        l_lstm = Bidirectional(LSTM(RNN_units, return_sequences=True))(embedded_sequences)
    elif RNN_gate_type == 'GRU':
        l_lstm = Bidirectional(GRU(RNN_units, return_sequences=True))(embedded_sequences)
    else:
        sys.exit("ERROR: RNN_gate_type is incorrectly assigned.")
    attention_word = one_step_attention(l_lstm, densor1_1, densor1_2, activator1, dotor1)
    attention_word = Reshape((RNN_units * 2,))(attention_word)
    sentEncoder = Model(sentence_input, attention_word)
    sentEncoder.summary()

    # Build document encoding and document attention
    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    if RNN_gate_type == 'LSTM':
        l_lstm_sent = Bidirectional(LSTM(RNN_units, return_sequences=True))(review_encoder)
    elif RNN_gate_type == 'GRU':
        l_lstm_sent = Bidirectional(GRU(RNN_units, return_sequences=True))(review_encoder)
    else:
        sys.exit("ERROR: RNN_gate_type is incorrectly assigned.")
    attention_sent = one_step_attention(l_lstm_sent, densor2_1, densor2_2, activator2, dotor2)
    preds = Dense(label_dim, activation='softmax')(attention_sent)
    preds = Reshape((label_dim,))(preds)

    model = Model(review_input, preds)

    return model


def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


# Test hierachical_network

"""
embedding_matrix = np.random.random((10000 + 1, 100))
embedding_layer = Embedding(10000 + 1,
                            100,
                            weights=[embedding_matrix],
                            input_length=100,
                            trainable=False)
model = hierachical_network(MAX_SENTS=15, MAX_SENT_LENGTH=100, embedding_layer=embedding_layer,
                            RNN_gate_type='LSTM', RNN_units=100, label_dim=2)
model.summary()
"""
"""
embedding_matrix = np.random.random((10000 + 1, 100))
embedding_layer = Embedding(10000 + 1,
                            100,
                            weights=[embedding_matrix],
                            input_length=100,
                            trainable=False)
model = hierachical_network_attention(MAX_SENTS=15, MAX_SENT_LENGTH=100, embedding_layer=embedding_layer,
                                      RNN_gate_type='LSTM', RNN_units=100, dropout=0.0, label_dim=2)
model.summary()
"""

"""
embedding_matrix = np.random.random((10000 + 1, 100))
embedding_layer = Embedding(10000 + 1,
                            100,
                            weights=[embedding_matrix],
                            input_length=100,
                            trainable=False)
model = hierachical_attention_network(MAX_SENTS=15, MAX_SENT_LENGTH=100, embedding_layer=embedding_layer,
                                      RNN_gate_type='LSTM', RNN_units=100, label_dim=2)
model.summary()
"""
