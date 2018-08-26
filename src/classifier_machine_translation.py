# Neural machine translation with attention
# Deeplearning.ai Course 5, Week 3
# Bi-RNN ==> attention ==> RNN

import time
from keras.optimizers import Adam
from model_machine_translation import *


# === Hyperparameters ===
# String, LSTM or GRU
RNN_gate_type = 'LSTM'
# Hidden state size of the Bi-LSTM/GRU
n_a = 32
# Hidden state size of the post-attention LSTM/GRU
n_s = 64

learning_rate = 0.005
loss = 'categorical_crossentropy'
epochs = 1
batch_size = 100
# ======


# Dataset

# datasetdataset: a list of tuples of (human readable date, machine readable date)
# human_vocab: a python dictionary mapping all characters used in the human readable dates to an integer-valued index
# machine_vocab: a python dictionary mapping all characters used in machine readable dates to an integer-valued index. These indices are not necessarily consistent with human_vocab.
# inv_machine_vocab: the inverse dictionary of machine_vocab, mapping from indices back to characters.

m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
print("=== First 10 tuples of " + str(m) + " pairs ===")
print(dataset[:10])
print("===")

# In case download is not completed
time.sleep(5)

Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Xoh.shape:", Xoh.shape)
print("Yoh.shape:", Yoh.shape)
print("===")

index = 0
print("Source date:", dataset[index][0])
print("Target date:", dataset[index][1])
print()
print("Source after preprocessing (indices):", X[index])
print("Target after preprocessing (indices):", Y[index])
print()
print("Source after preprocessing (one-hot):", Xoh[index])
print("Target after preprocessing (one-hot):", Yoh[index])
print("===")

# Create model
model = BiRNN_attention_RNN(Tx=Tx, Ty=Ty, n_a=n_a, n_s=n_s, human_vocab_size=Xoh.shape[2],
                            machine_vocab_size=Yoh.shape[2], RNN_gate_type=RNN_gate_type)
model.summary()

# Define loss, optimizer and metrics
opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(optimizer=opt, metrics=['accuracy'], loss=loss)

# define all your inputs and outputs to fit the model
s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))

# fit the model and run it for one epoch
model.fit([Xoh, s0, c0], outputs, epochs=epochs, batch_size=batch_size)

# load pre-trained weights
model.load_weights('models/model.h5')

"""
# See the results on new examples
EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001',
            'March 3rd 2001', '1 March 2001']
for example in EXAMPLES:
    source = string_to_int(example, Tx, human_vocab)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0, 1)
    prediction = model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis=-1)
    output = [inv_machine_vocab[int(i)] for i in prediction]

    print("source:", example)
    print("output:", ''.join(output))
"""

# Visualizing Attention
attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "Tuesday 09 Oct 1993", num=7, n_s=64)
