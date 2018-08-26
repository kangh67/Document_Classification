from keras.models import Model
from keras.layers import Dense, Input, LSTM, GRU, Activation
from keras.layers import RepeatVector, Concatenate, Dot, Bidirectional
from utils_nmt import *


def one_step_attention(a, s_prev, repeator, concatenator, densor1, densor2, activator, dotor):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.

    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """

    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
    s_prev = repeator(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
    concat = concatenator([a, s_prev])
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
    e = densor1(concat)
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
    energies = densor2(e)
    # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
    alphas = activator(energies)
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
    context = dotor([alphas, a])

    return context


def BiRNN_attention_RNN(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size, RNN_gate_type):
    """
        Arguments:
        Tx -- length of the input sequence
        Ty -- length of the output sequence
        n_a -- hidden state size of the Bi-LSTM
        n_s -- hidden state size of the post-attention LSTM
        human_vocab_size -- size of the python dictionary "human_vocab"
        machine_vocab_size -- size of the python dictionary "machine_vocab"
        RNN_gate_type -- String, LSTM or GRU

        Returns:
        model -- Keras model instance
    """

    # Defined shared layers as global variables

    repeator = RepeatVector(Tx)
    concatenator = Concatenate(axis=-1)
    densor1 = Dense(10, activation="tanh")
    densor2 = Dense(1, activation="relu")
    activator = Activation(softmax,
                           name='attention_weights')  # We are using a custom softmax(axis = 1) loaded in this notebook
    dotor = Dot(axes=1)

    if RNN_gate_type == 'LSTM':
        post_activation_LSTM_cell = LSTM(n_s, return_state=True)
    elif RNN_gate_type == 'GRU':
        post_activation_LSTM_cell = GRU(n_s, return_state=True)
    output_layer = Dense(machine_vocab_size, activation=softmax)

    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0

    # Initialize empty list of outputs
    outputs = []

    # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True.
    if RNN_gate_type == 'LSTM':
        a = Bidirectional(LSTM(n_a, return_sequences=True))(X)
    elif RNN_gate_type == 'GRU':
        a = Bidirectional(GRU(n_a, return_sequences=True))(X)

    # Step 2: Iterate for Ty steps
    for t in range(Ty):
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t
        context = one_step_attention(a, s, repeator, concatenator, densor1, densor2, activator, dotor)

        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state]
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])

        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM
        out = output_layer(s)

        # Step 2.D: Append "out" to the "outputs" list
        outputs.append(out)

    # Step 3: Create model instance taking three inputs and returning the list of outputs.
    model = Model(inputs=[X, s0, c0], outputs=outputs)

    return model


# Test BiRNN_attention_RNN

# Total params: 52,960
# Trainable params: 52,960
# Non-trainable params: 0

"""
model = BiRNN_attention_RNN(Tx=30, Ty=10, n_a=32, n_s=64, human_vocab_size=37, machine_vocab_size=11, RNN_gate_type='LSTM')
model.summary()
"""
#