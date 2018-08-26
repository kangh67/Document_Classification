# Hierarchical text classifiers with/without attention

from keras import optimizers
from nlp_MAUDE import *
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from keras.callbacks import Callback, ModelCheckpoint
from additional_metrics import *


# === Hyper parameters ===
attention_switch = True  # with or without attention

EMB_DIM = 300
trainable = True

RNN_gate_type = 'LSTM'
RNN_units = 128
dropout = 0.5   # default = 0, no dropout
label_dim = 2

loss = 'categorical_crossentropy'
optimizer = 'adam'
lr = 0.0005

epochs = 20
batch_size = 50

MAX_NUM_WORDS = 20000
MAX_SENTS = 30
MAX_SENT_LENGTH = 100

DATA_SPLIT = [7, 1, 2]
# ======


# Create .tsv file from .xlsx file
# Not necessary if .tsv has been already created
xlsx_to_tsv()


# Read MAUDE data downloaded from .tsv file
print("=== Reading MAUDE data ===")
documents_sent, labels, documents = read_MAUDE_hierarchical_simple()


# Tokenization
print("=== Tokenizing ===")
data, labels, word_index = tokenazation(documents_sent, labels, documents, MAX_NUM_WORDS, MAX_SENTS, MAX_SENT_LENGTH)


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
                            input_length=MAX_SENT_LENGTH,
                            trainable=trainable)


# Construct model
if attention_switch:
    print("=== Building hierarchical RNN model WITH attention ===")
    model = hierachical_network_attention(MAX_SENTS=MAX_SENTS,
                                          MAX_SENT_LENGTH=MAX_SENT_LENGTH,
                                          embedding_layer=embedding_layer,
                                          RNN_gate_type=RNN_gate_type,
                                          RNN_units=RNN_units,
                                          dropout=dropout,
                                          label_dim=label_dim)
else:
    print("=== Building hierarchical RNN model WITHOUT attention ===")
    model = hierachical_network(MAX_SENTS=MAX_SENTS,
                                MAX_SENT_LENGTH=MAX_SENT_LENGTH,
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
checkpointer = ModelCheckpoint(filepath='./models/model.{epoch:02d}-{val_loss:.2f}-{val_acc:.3f}.hdf5',
                               monitor='val_acc',
                               mode='max',
                               verbose=1,
                               save_best_only=True,
                               save_weights_only=False)


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
DataFrame(history.history).to_csv("logs/att_" + str(attention_switch) + "_dim_" + str(EMB_DIM) +
                                  "_trainable_" + str(trainable) + ".csv")

"""
# Prepare train, dev, and test set for baseline models
x_train_mean_emb = mean_emb_for_document(x_train, embedding_matrix)
x_val_mean_emb = mean_emb_for_document(x_val, embedding_matrix)
x_test_mean_emb = mean_emb_for_document(x_test, embedding_matrix)
y_train_mean_emb = y_train.T[1]
y_val_mean_emb = y_val.T[1]
y_test_mean_emb = y_test.T[1]


# train and test logistic regression
clf_lr = LogisticRegression()

print('====== Logistic Regression model ======')
print('Training on', x_train_mean_emb.shape[0], 'samples (train set) ...')
clf_lr.fit(x_train_mean_emb, y_train_mean_emb)

print('Predicting on', x_val_mean_emb.shape[0], 'samples (val set) ...')
predicted_val = clf_lr.predict(x_val_mean_emb)
print('Accuracy on val: {:4.2f}'.format(np.mean(predicted_val == y_val_mean_emb)))
print(metrics.classification_report(y_val_mean_emb, predicted_val))

print('Predicting on', x_test_mean_emb.shape[0], 'samples (test set) ...')
predicted_test = clf_lr.predict(x_test_mean_emb)
print('Accuracy on test: {:4.2f}'.format(np.mean(predicted_test == y_test_mean_emb)))
print(metrics.classification_report(y_test_mean_emb, predicted_test))


# train and test Random Forest model
clf_rf = RandomForestClassifier()

print("====== Random Forest model ======")
print('Training on', x_train_mean_emb.shape[0], 'samples (train set) ...')
clf_rf.fit(x_train_mean_emb, y_train_mean_emb)

print('Predicting on', x_val_mean_emb.shape[0], 'samples (val set) ...')
predicted_val = clf_rf.predict(x_val_mean_emb)
print('Accuracy on val: {:4.2f}'.format(np.mean(predicted_val == y_val_mean_emb)))
print(metrics.classification_report(y_val_mean_emb, predicted_val))

print('Predicting on', x_test_mean_emb.shape[0], 'samples (test set) ...')
predicted_test = clf_rf.predict(x_test_mean_emb)
print('Accuracy on test: {:4.2f}'.format(np.mean(predicted_test == y_test_mean_emb)))
print(metrics.classification_report(y_test_mean_emb, predicted_test))


# train and test SVM model
clf_svm = svm.SVC(kernel='linear', C=1)

print("====== SVM model ======")
print('Training on', x_train_mean_emb.shape[0], 'samples (train set) ...')
clf_svm.fit(x_train_mean_emb, y_train_mean_emb)

print('Predicting on', x_val_mean_emb.shape[0], 'samples (val set) ...')
predicted_val = clf_svm.predict(x_val_mean_emb)
print('Accuracy on val: {:4.2f}'.format(np.mean(predicted_val == y_val_mean_emb)))
print(metrics.classification_report(y_val_mean_emb, predicted_val))

print('Predicting on', x_test_mean_emb.shape[0], 'samples (test set) ...')
predicted_test = clf_svm.predict(x_test_mean_emb)
print('Accuracy on test: {:4.2f}'.format(np.mean(predicted_test == y_test_mean_emb)))
print(metrics.classification_report(y_test_mean_emb, predicted_test))
"""
