from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, GRU, Activation
from word_embedding import *


def two_layers_RNN(input_shape, word_to_vec_map, word_to_index, emb_trainable, RNN_gate_type, RNN_units, dropout, softmax_shape):
    """
    2-layer LSTM sequence classifier
    Function creating the Emojify-v2 model's graph, in deeplearning.ai course 5 week 2.

    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)
    trainable -- Fix pre-trained word embedding (True) or make it trainable (False)
    RNN_gate_type -- String, LSTM or GRU
    RNN_units -- Positive integer, dimensionality of the output space
    dropout -- Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs
    softmax_shape -- Positive integer, output shape after softmax activation

    Returns:
    model -- a model instance in Keras
    """

    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(shape=input_shape, dtype=np.int32)

    # Create the embedding layer pretrained with GloVe Vectors
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index, emb_trainable)

    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)

    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    if RNN_gate_type == 'LSTM':
        X = LSTM(RNN_units, return_sequences=True)(embeddings)
    elif RNN_gate_type == 'GRU':
        X = GRU(RNN_units, return_sequences=True)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(dropout)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    if RNN_gate_type == 'LSTM':
        X = LSTM(RNN_units, return_sequences=False)(X)
    elif RNN_gate_type == 'GRU':
        X = GRU(RNN_units, return_sequences=False)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(dropout)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(softmax_shape, activation="softmax")(X)
    # Add a softmax activation
    X = Activation("softmax")(X)

    # Create Model instance which converts sentence_indices into X.
    model = Model(sentence_indices, X)

    return model


def test_two_layers_RNN(maxLen, word_to_vec_map, word_to_index, emb_trainable, RNN_gate_type, RNN_units, dropout, softmax_shape):
    """
    Test above function LSTM_two_layers()
    Expected key outputs:

        Total params: 20,223,927
        Trainable params: 20,223,927
        Non-trainable params: 0

    Arguments:
    maxLen -- max word # of a sentence
    """
    model = two_layers_RNN((maxLen,), word_to_vec_map, word_to_index, emb_trainable, RNN_gate_type, RNN_units, dropout, softmax_shape)
    model.summary()


# Test LSTM_two_layers model
"""
from utils_emo import *

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
test_two_layers_RNN(5, word_to_vec_map, word_to_index, emb_trainable=False, RNN_gate_type='GRU',
                    RNN_units=128, dropout=0.5, softmax_shape=5)
"""
#