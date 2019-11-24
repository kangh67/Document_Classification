# Text classifiers with RNN

from keras import optimizers
from nlp_MAUDE import *
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from keras.callbacks import Callback, ModelCheckpoint


# === Hyper parameters ===
attention_switch = True  # with or without attention

EMB_DIM = 300
trainable = True

RNN_gate_type = 'LSTM'
RNN_units = 128
dropout = 0.5
label_dim = 2

loss = 'categorical_crossentropy'
optimizer = 'adam'
lr = 0.0005
metrics = 'acc'

epochs = 20
batch_size = 50

MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 2000

DATA_SPLIT = [7, 1, 2]
# ======


# Create .tsv file from .xlsx file
# Not necessary if .tsv has been already created
xlsx_to_tsv()


# Read MAUDE data downloaded from .tsv file
print("=== Reading MAUDE data ===")
labels, documents = read_MAUDE_simple()


# Tokenization
print("=== Tokenizing ===")
data, labels, word_index = tokenazation_CNN(labels, documents, MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH)


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
checkpointer = ModelCheckpoint(filepath='./models/weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.3f}.hdf5',
                               monitor='val_acc',
                               mode='max',
                               verbose=1,
                               save_best_only=True,
                               save_weights_only=True)


# Training
print("=== Training ===")
history = model.fit(x_train,
                    y_train,
                    validation_data=(x_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    callbacks=[recall_metrics, checkpointer])


# Write log
customized_metrics = {'val_f1': recall_metrics.val_f1s, 'val_pre:': recall_metrics.val_precisions,
                      'val_rec': recall_metrics.val_recalls, 'test_f1': recall_metrics.test_f1s,
                      'test_pre:': recall_metrics.test_precisions, 'test_rec': recall_metrics.test_recalls,
                      'test_acc:': recall_metrics.test_accs}
history.history.update(customized_metrics)
DataFrame(history.history).to_csv("logs/RNN_" + str(attention_switch) + "_dim_" + str(EMB_DIM) +
                                  "_trainable_" + str(trainable) + ".csv")
