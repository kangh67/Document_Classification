# Use google universal-sentence-encoder-large
# https://www.tensorflow.org/hub/modules/google/universal-sentence-encoder-large/3


import tensorflow as tf
import tensorflow_hub as hub
from keras import optimizers
from keras import layers
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from keras.callbacks import Callback, ModelCheckpoint
from nlp_MAUDE import *
from model_hatt import *

# === Hyper parameters ===
doc_or_sent = 'doc'  # either by documents 'doc' or by sentences 'sent'
attention_switch = False  # eligible only when doc_or_sent = 'sent'
RNN_gate_type = 'LSTM'  # eligible only when doc_or_sent = 'sent'
RNN_units = 128  # eligible only when doc_or_sent = 'sent'

trainable = True

EMB_DIM = 512  # universal-sentence-encoder embedding dimension
dense_num = 2  # dense layer # after the embedding layer
dense_unit = 1024  # dense unit #
dropout = 0.5
label_dim = 2

loss = 'categorical_crossentropy'
optimizer = 'adam'
lr = 0.001
metrics = 'acc'

epochs = 30
batch_size = 50

MAX_SENTS = 20

DATA_SPLIT = [7, 1, 2]
# ======


# Loaded the Universal Sentence Encoder
embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3", trainable=True)


# Wrap the Universal Sentence Encoder in a Keras Lambda layer and explicitly cast its input as a string.
def UniversalEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]


# Read MAUDE data downloaded from .tsv file
print("=== Reading MAUDE data ===")
# by document
if doc_or_sent == 'doc':
    labels, documents = read_MAUDE_simple()  # simple tokenization
    # labels, documents = read_MAUDE_by_document()  # complex tokenization
    data = np.asarray(documents)
# by sentence
else:
    documents_sent, labels, documents = read_MAUDE_hierarchical_simple()  # simple tokenization
    # documents_sent, labels, documents = read_MAUDE()  # complex tokenization
    data = documents_sent
    for i in range(len(data)):
        if len(data[i]) > MAX_SENTS:
            data[i] = data[i][: MAX_SENTS]
        elif len(data[i]) < MAX_SENTS:
            while len(data[i]) < MAX_SENTS:
                data[i] = ['None.'] + data[i]
        data[i] = np.asarray(data[i])
    data = np.asarray(data)

labels = to_categorical(np.asarray(labels))
print('Input data shape:', data.shape)
print('Label shape:', labels.shape)


# Preparing training, developing, and testing data
if len(DATA_SPLIT) == 2:
    print("=== Preparing training (" + str(10 * DATA_SPLIT[0]) + "%) and developing data (" +
          str(10 * DATA_SPLIT[1]) + "%) ===")
elif len(DATA_SPLIT) == 3:
    print("=== Preparing training (" + str(10 * DATA_SPLIT[0]) + "%), developing data (" +
          str(10 * DATA_SPLIT[1]) + "%), and testing data (" + str(10 * DATA_SPLIT[2]) + "%) ===")
else:
    sys.exit("ERROR: Length of DATA_SPLIT should be either 2 or 3.")
x_train, y_train, x_val, y_val, x_test, y_test = prepare_train_dev(data, labels, DATA_SPLIT)


# Build the Keras model
if doc_or_sent == 'doc':
    input_text = Input(shape=(1,), dtype="string")
    embedding = layers.Lambda(UniversalEmbedding, output_shape=(EMB_DIM,))(input_text)
    output = Dense(dense_unit, activation='relu')(embedding)
    output = Dropout(dropout)(output)
    current_dense = 1
    while current_dense < dense_num:
        output = Dense(dense_unit, activation='relu')(output)
        output = Dropout(dropout)(output)
        current_dense += 1
    pred = Dense(2, activation='softmax')(output)
    model = Model(inputs=[input_text], outputs=pred)
else:
    embedding = embed(["The quick brown fox jumps over the lazy dog.", "I am a sentence for which I would like to get its embedding"])
    with tf.Session() as session:
        session.run(embedding)
    """
    # Sentence encoder
    input_sent = Input(shape=(1,), dtype="string")
    embedding = layers.Lambda(UniversalEmbedding, output_shape=(EMB_DIM,))(input_sent)
    sentEncoder = Model(input_sent, embedding)
    print('=== Sentence level encoder summary ===')
    print(sentEncoder.summary())
    # Document encoder
    input_doc = Input(shape=(MAX_SENTS,), dtype="string")
    for i in range(MAX_SENTS):
        sent_dim = sentEncoder(input_doc[i])
        input_doc[i] = sent_dim
    
    input_text = Input(shape=(MAX_SENTS,), dtype="string")
    embedding = layers.Lambda(UniversalEmbedding, output_shape=(MAX_SENTS, EMB_DIM))(input_text)
    # embedding = Flatten()(embedding)
    
    if RNN_gate_type == 'LSTM':
        output = Bidirectional(LSTM(RNN_units, dropout=dropout, return_sequences=attention_switch))(embedding)
    elif RNN_gate_type == 'GRU':
        output = Bidirectional(GRU(RNN_units, dropout=dropout, return_sequences=attention_switch))(embedding)
    else:
        sys.exit("ERROR: RNN_gate_type is incorrectly assigned.")
    if attention_switch:
        output = AttentionWithContext()(output)  # attention layer
    
    pred = Dense(2, activation='softmax')(embedding)
    model = Model(inputs=[input_text], outputs=pred)
    """

print(model.summary())


# Compile model
if optimizer == 'adam':
    opt = optimizers.Adam(lr=lr)
else:
    sys.exit("ERROR: The assigned optimizer is not supported.")
model.compile(loss=loss,
              optimizer=opt,
              metrics=['acc'])


# Apply Callback to calculate f1, precision, and recall on dev set after each epoch
class f1_precision_recall(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.test_f1s = []
        self.test_recalls = []
        self.test_precisions = []
        self.test_accs = []

    def on_epoch_end(self, epoch, logs={}):
        # val
        val_predict = (np.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ.T[1], val_predict.T[1])
        _val_recall = recall_score(val_targ.T[1], val_predict.T[1])
        _val_precision = precision_score(val_targ.T[1], val_predict.T[1])
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print('- val_f1: %0.4f - val_precision: %0.4f - val_recall: %0.4f' % (_val_f1, _val_precision, _val_recall))
        # test
        test_predict = (np.asarray(self.model.predict(
            x_test))).round()
        test_targ = y_test
        _test_f1 = f1_score(test_targ.T[1], test_predict.T[1])
        _test_recall = recall_score(test_targ.T[1], test_predict.T[1])
        _test_precision = precision_score(test_targ.T[1], test_predict.T[1])
        _test_acc = accuracy_score(test_targ.T[1], test_predict.T[1])
        self.test_f1s.append(_test_f1)
        self.test_recalls.append(_test_recall)
        self.test_precisions.append(_test_precision)
        self.test_accs.append(_test_acc)
        print('- test_f1: %0.4f - test_precision: %0.4f - test_recall: %0.4f - test_acc: %0.4f'
              % (_test_f1, _test_precision, _test_recall, _test_acc))
        return


recall_metrics = f1_precision_recall()

# saves the model and weights after each epoch if the validation loss decreased
checkpointer = ModelCheckpoint(filepath='./models/model.{epoch:02d}-{val_loss:.2f}-{val_acc:.3f}.hdf5',
                               monitor='val_acc',
                               mode='max',
                               verbose=1,
                               save_best_only=True,
                               save_weights_only=False)

# Training
with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    history = model.fit(x_train,
                        y_train,
                        validation_data=(x_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        callbacks=[recall_metrics, checkpointer])
    # model.save_weights('./models/model_use.h5')

# Write log
customized_metrics = {'val_f1': recall_metrics.val_f1s, 'val_pre:': recall_metrics.val_precisions,
                      'val_rec': recall_metrics.val_recalls, 'test_f1': recall_metrics.test_f1s,
                      'test_pre:': recall_metrics.test_precisions, 'test_rec': recall_metrics.test_recalls,
                      'test_acc:': recall_metrics.test_accs}
history.history.update(customized_metrics)
DataFrame(history.history).to_csv("logs/universal_by_" + doc_or_sent + "_dim_" + str(EMB_DIM) +
                                  "_trainable_" + str(trainable) + ".csv")
