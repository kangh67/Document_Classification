# Emoji classification
# Deeplearning.ai Course 5, Week 2
# Two layers, single direction LSTM/GRU network

from model_emoji import *
from utils_emo import *

# === Hyperparameters ===
# Word embedding dimensions 50/100/200/300
word_embedding_dim = 300
# Fix pre-trained word embedding (True) or make it trainable (False)
emb_trainable = False
# String, LSTM or GRU
RNN_gate_type = 'LSTM'
# Positive integer, dimensionality of the output space
RNN_units = 128
# Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs
dropout = 0.5
loss = 'categorical_crossentropy'
optimizer = 'adam'
epochs = 50
batch_size = 32
shuffle = True
# ======


# Load training and testing set

X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')


# Represent Y as on-hot vectors

Y_oh_train = convert_to_one_hot(Y_train, C = 5)
Y_oh_test = convert_to_one_hot(Y_test, C = 5)


# Determine max word# of one sentence

maxLen = len(max(X_train, key=len).split())


# Print data summary

print("======================")
print("Max word# in one sentence (maxLen) = " + str(maxLen))
print("======================")
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("Y_one_hot_train shape: " + str(Y_oh_train.shape))
print("======================")
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))
print("Y_one_hot_test shape: " + str(Y_oh_test.shape))
print("======================")
print("An example of training data: ")
print(X_train[1], label_to_emoji(Y_train[1]))


# GloVe pre-trained word embeddings, 400,001 words, 50/100/200/300 dimensions
# word_to_index: dictionary mapping from words to their indices in the vocabulary (400,001 words, with the valid indices ranging from 0 to 400,000)
# index_to_word: dictionary mapping from indices to their corresponding words in the vocabulary
# word_to_vec_map: dictionary mapping words to their GloVe vector representation.

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.' + str(word_embedding_dim) + 'd.txt')


# Create model

model = two_layers_RNN((maxLen,), word_to_vec_map, word_to_index, emb_trainable=emb_trainable, RNN_gate_type=RNN_gate_type,
                    RNN_units=RNN_units, dropout=dropout, softmax_shape=5)
model.summary()
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])


# Convert X_train (array of sentences as strings) to X_train_indices (array of sentences as list of word indices)
X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)

# Convert Y_train (labels as indices) to Y_train_oh (labels as one-hot vectors)
Y_train_oh = convert_to_one_hot(Y_train, C = 5)

# Fit the Keras model on X_train_indices and Y_train_oh. We will use epochs = 50 and batch_size = 32.
model.fit(X_train_indices, Y_train_oh, epochs=epochs, batch_size=batch_size, shuffle=shuffle)


# Evaluate the model on test set

X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C = 5)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print()
print("Test accuracy = ", acc)


# This code allows you to see the mislabelled examples

C = 5
y_test_oh = np.eye(C)[Y_test.reshape(-1)]
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
pred = model.predict(X_test_indices)
for i in range(len(X_test)):
    x = X_test_indices
    num = np.argmax(pred[i])
    if(num != Y_test[i]):
        print('Expected emoji:'+ label_to_emoji(Y_test[i]) + ' prediction: '+ X_test[i] + label_to_emoji(num).strip())