from nlp_IMDB import *
from keras import optimizers


# === Hyper parameters ===
attention_switch = True    # with or without attention

EMB_DIM = 100
trainable = True

RNN_gate_type = 'GRU'
RNN_units = 128
dropout = 0
label_dim = 2

loss = 'categorical_crossentropy'
optimizer = 'adam'
lr = 0.001
metrics = 'acc'

epochs = 10
batch_size = 50

MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000

VALIDATION_SPLIT = 0.2
# ======


# Read IMDB data downloaded from website
print("=== Reading IMDB data ===")
labels, texts = read_IMDB_by_document()

# Tokenization
print("=== Tokenizing ===")
data, labels, word_index = tokenazation_CNN(labels, texts, MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH)

# Preparing training and developing data
print("=== Preparing training (" + str(100 - 100 * VALIDATION_SPLIT) + "%) and developing data (" +
      str(100 * VALIDATION_SPLIT) + "%) ===")
x_train, y_train, x_val, y_val = prepare_train_dev(data, labels, VALIDATION_SPLIT)

# Generate word embedding matrix for embedding layer establishment
print("=== Generating word embedding matrix ===")
embedding_matrix = calculate_embedding_matrix(EMB_DIM, word_index)

# Build word embedding layer
print("=== Building word embedding layer (" + str(EMB_DIM) + " dimensions) ===")
embedding_layer = Embedding(len(word_index) + 1,
                            EMB_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=trainable)

# Construct model
if not attention_switch:
    print("=== Building a RNN model without Attention ===")
    model = RNN_simple(MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
                       embedding_layer=embedding_layer,
                       RNN_gate_type=RNN_gate_type,
                       RNN_units=RNN_units,
                       dropout=dropout,
                       label_dim=label_dim)
else:
    print("=== Building an Attention RNN model ===")
    model = RNN_attention(MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
                          embedding_layer=embedding_layer,
                          RNN_gate_type=RNN_gate_type,
                          RNN_units=RNN_units,
                          dropout=dropout,
                          label_dim=label_dim)

model.summary()


# Compile model
if optimizer == 'adam':
    opt = optimizers.Adam(lr=lr)
else:
    sys.exit("ERROR: The assigned optimizer is not supported.")
model.compile(loss=loss,
              optimizer=opt,
              metrics=[metrics])

# Training
print("=== Training ===")
model.fit(x_train,
          y_train,
          validation_data=(x_val, y_val),
          epochs=epochs,
          batch_size=batch_size)
