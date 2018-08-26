# Try different model combinations

from keras import optimizers
from nlp_MAUDE import *
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import load_model
from additional_metrics import *

mode = 'compare'  # 'CNN', 'RNN', 'RNN_att', 'H_RNN', 'H_RNN_att', 'compare'

EMB_DIM = 300
trainable = True

loss = 'categorical_crossentropy'
optimizer = 'adam'
lr = 0.0005

epochs = 20
batch_size = 50

MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000
MAX_SENTS = 20
MAX_SENT_LENGTH = 100

tsv_file = './data/MAUDE_2008_2016_review.tsv'
train_file = './data/MAUDE_train.tsv'
dev_file = './data/MAUDE_dev.tsv'
test_file = './data/MAUDE_test.tsv'

"""
# === Split the original data to train, dev, and test, run for the first time only ===
# read file
print("Reading raw data ...")
data = pd.read_csv(tsv_file, sep='\t')
data = np.asarray(data)
print("Raw data shape: " + str(data.shape))

# shuffle
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
data_train = data[: int(0.7 * data.shape[0])]
data_dev = data[int(0.7 * data.shape[0]): (int(0.7 * data.shape[0]) + int(0.1 * data.shape[0]))]
data_test = data[(int(0.7 * data.shape[0]) + int(0.1 * data.shape[0])):]

# generate train, dev, and test files
with open(train_file, "w") as f:
    f.write("ID\tHIT\tREPORT\n")
    for one_line in data_train:
        f.write(str(one_line[0]) + "\t" + str(one_line[1]) + "\t" + str(one_line[2]) + "\n")
print('Training file written to:', train_file)

with open(dev_file, "w") as f:
    f.write("ID\tHIT\tREPORT\n")
    for one_line in data_dev:
        f.write(str(one_line[0]) + "\t" + str(one_line[1]) + "\t" + str(one_line[2]) + "\n")
print('Dev file written to:', dev_file)

with open(test_file, "w") as f:
    f.write("ID\tHIT\tREPORT\n")
    for one_line in data_test:
        f.write(str(one_line[0]) + "\t" + str(one_line[1]) + "\t" + str(one_line[2]) + "\n")
print('Test file written to:', test_file)
"""

# === Prepare train, dev, and test data ===
data_train = pd.read_csv(train_file, sep='\t')
data_dev = pd.read_csv(dev_file, sep='\t')
data_test = pd.read_csv(test_file, sep='\t')
print('=== Reading MAUDE data ===')
doc_hie_train, y_train, doc_train = read_MAUDE_train_dev_test(train_file)
doc_hie_dev, y_dev, doc_dev = read_MAUDE_train_dev_test(dev_file)
doc_hie_test, y_test, doc_test = read_MAUDE_train_dev_test(test_file)
doc_hie_all, y_all, doc_all = read_MAUDE_hierarchical_simple()
y_train_tfidf = y_train.T[1]
y_test_tfidf = y_test.T[1]
print('Number of positive and negative reviews in training, dev, and test set: ')
print('Train:', y_train.sum(axis=0))
print('Dev:', y_dev.sum(axis=0))
print('Test:', y_test.sum(axis=0))

# === Tokenization ===
print('=== Tokenization ===')
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(doc_all)
word_index = tokenizer.word_index
print("Total unique tokens: " + str(len(word_index)))

# TFIDF input, for SVM and LR models
x_train_tfidf, x_test_tfidf = tokenazation_tfidf(doc_train, doc_test)

# sequence input, for CNN, RNN, and RNN_att models
sequences_train = tokenizer.texts_to_sequences(doc_train)
sequences_dev = tokenizer.texts_to_sequences(doc_dev)
sequences_test = tokenizer.texts_to_sequences(doc_test)
x_train_seq = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
x_dev_seq = pad_sequences(sequences_dev, maxlen=MAX_SEQUENCE_LENGTH)
x_test_seq = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

# hierarchical input, for H_RNN and H_RNN_att models
x_train_hie = tokenazation_combined_models(word_index, doc_hie_train, MAX_NUM_WORDS, MAX_SENTS, MAX_SENT_LENGTH)
x_dev_hie = tokenazation_combined_models(word_index, doc_hie_dev, MAX_NUM_WORDS, MAX_SENTS, MAX_SENT_LENGTH)
x_test_hie = tokenazation_combined_models(word_index, doc_hie_test, MAX_NUM_WORDS, MAX_SENTS, MAX_SENT_LENGTH)
print('train shape: seq -', x_train_seq.shape, ', hie -', x_train_hie.shape)
print('dev shape: seq -', x_dev_seq.shape, ', hie -', x_dev_hie.shape)
print('test shape: seq -', x_test_seq.shape, ', hie -', x_test_hie.shape)

# === Enbedding matrix calculation ===
print("=== Generating word embedding matrix ===")
embedding_matrix = calculate_embedding_matrix(EMB_DIM, word_index)

# Build word embedding layer
print("Building word embedding layer (" + str(EMB_DIM) + " dimensions)")
embedding_layer = Embedding(len(word_index) + 1,
                            EMB_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=trainable)

print("Building word embedding layer for hierarchical models (" + str(EMB_DIM) + " dimensions)")
embedding_layer_H = Embedding(len(word_index) + 1,
                              EMB_DIM,
                              weights=[embedding_matrix],
                              input_length=MAX_SENT_LENGTH,
                              trainable=trainable)

# === CNN model ===
if mode == 'CNN':
    print('=== CNN model selected ===')
    model = CNN_complex(MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
                        embedding_layer=embedding_layer,
                        filter_num=256,
                        kernal_size=[2, 3, 4],
                        label_dim=2,
                        dropout_CNN=0.3,
                        dropout_full_con=0.5)

# === RNN model ===
elif mode == 'RNN':
    print('=== RNN model selected ===')
    model = RNN_simple(MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
                       embedding_layer=embedding_layer,
                       RNN_gate_type='LSTM',
                       RNN_units=128,
                       dropout=0.5,
                       label_dim=2)

# === RNN_att model ===
elif mode == 'RNN_att':
    print("=== RNN_att model selected ===")
    model = RNN_attention(MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
                          embedding_layer=embedding_layer,
                          RNN_gate_type='LSTM',
                          RNN_units=128,
                          dropout=0.5,
                          label_dim=2)

# === H_RNN model ===
elif mode == 'H_RNN':
    print("=== H_RNN model selected ===")
    model = hierachical_network(MAX_SENTS=MAX_SENTS,
                                MAX_SENT_LENGTH=MAX_SENT_LENGTH,
                                embedding_layer=embedding_layer_H,
                                RNN_gate_type='LSTM',
                                RNN_units=128,
                                dropout=0.5,
                                label_dim=2)

# === H_RNN_att model ===
elif mode == 'H_RNN_att':
    print("=== H_RNN_att model selected ===")
    model = hierachical_network_attention(MAX_SENTS=MAX_SENTS,
                                          MAX_SENT_LENGTH=MAX_SENT_LENGTH,
                                          embedding_layer=embedding_layer_H,
                                          RNN_gate_type='LSTM',
                                          RNN_units=128,
                                          dropout=0.5,
                                          label_dim=2)

if mode != 'compare':
    model.summary()


# Apply Callback to calculate f1, precision, and recall on dev set after each epoch
class f1_precision_recall(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

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
        return


# === Common codes for model training ===
if mode != 'compare':
    # Compile model
    if optimizer == 'adam':
        opt = optimizers.Adam(lr=lr)
    else:
        sys.exit("ERROR: The assigned optimizer is not supported.")
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['acc'])

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
    if mode == 'CNN' or mode == 'RNN' or mode == 'RNN_att':
        x_train = x_train_seq
        x_dev = x_dev_seq
    else:
        x_train = x_train_hie
        x_dev = x_dev_hie

    print('Training set shape:', x_train.shape)
    history = model.fit(x_train,
                        y_train,
                        validation_data=(x_dev, y_dev),
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        callbacks=[recall_metrics, checkpointer])
    del model
    # Write log
    customized_metrics = {'val_f1': recall_metrics.val_f1s, 'val_pre:': recall_metrics.val_precisions,
                          'val_rec': recall_metrics.val_recalls}
    history.history.update(customized_metrics)
    DataFrame(history.history).to_csv("logs/" + str(mode) + ".csv")

# compare mode
else:
    # get model list
    print('=== Available models ===')
    path = './models/best_weights'
    files = os.listdir(path)
    models = []
    for file in files:
        if '.hdf5' in file:
            models.append(path + '/' + file)
    models = np.asarray(models)
    print(models)

    model_names = []
    model_probas = []

    # make prediction by using each model
    print('=== Individual model performance on test set ===')
    for m in models:
        if 'CNN' in m:
            model = CNN_complex(MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
                                embedding_layer=embedding_layer,
                                filter_num=256,
                                kernal_size=[2, 3, 4],
                                label_dim=2,
                                dropout_CNN=0.3,
                                dropout_full_con=0.5)
            model.load_weights(m)
            pred_cnn = np.asarray(model.predict(x_test_seq))
            model_names.append('CNN')
            model_probas.append(pred_cnn)
            y_cnn = pred_cnn.round()
            f1_cnn = f1_score(y_test.T[1], y_cnn.T[1])
            recall_cnn = recall_score(y_test.T[1], y_cnn.T[1])
            precision_cnn = precision_score(y_test.T[1], y_cnn.T[1])
            acc_cnn = accuracy_score(y_test.T[1], y_cnn.T[1])
            print('CNN model performance: acc = %0.4f, f1 = %0.4f, precision = %0.4f, recall = %0.4f'
                  % (acc_cnn, f1_cnn, precision_cnn, recall_cnn))
        elif 'H_RNN_att' in m:
            model = hierachical_network_attention(MAX_SENTS=MAX_SENTS,
                                                  MAX_SENT_LENGTH=MAX_SENT_LENGTH,
                                                  embedding_layer=embedding_layer_H,
                                                  RNN_gate_type='LSTM',
                                                  RNN_units=128,
                                                  dropout=0.5,
                                                  label_dim=2)
            model.load_weights(m)
            pred_h_rnn_att = np.asarray(model.predict(x_test_hie))
            model_names.append('H_RNN_att')
            model_probas.append(pred_h_rnn_att)
            y_h_rnn_att = pred_h_rnn_att.round()
            f1_h_rnn_att = f1_score(y_test.T[1], y_h_rnn_att.T[1])
            recall_h_rnn_att = recall_score(y_test.T[1], y_h_rnn_att.T[1])
            precision_h_rnn_att = precision_score(y_test.T[1], y_h_rnn_att.T[1])
            acc_h_rnn_att = accuracy_score(y_test.T[1], y_h_rnn_att.T[1])
            print('H_RNN_att model performance: acc = %0.4f, f1 = %0.4f, precision = %0.4f, recall = %0.4f'
                  % (acc_h_rnn_att, f1_h_rnn_att, precision_h_rnn_att, recall_h_rnn_att))
        elif 'H_RNN' in m:
            model = hierachical_network(MAX_SENTS=MAX_SENTS,
                                        MAX_SENT_LENGTH=MAX_SENT_LENGTH,
                                        embedding_layer=embedding_layer_H,
                                        RNN_gate_type='LSTM',
                                        RNN_units=128,
                                        dropout=0.5,
                                        label_dim=2)
            model.load_weights(m)
            pred_h_rnn = np.asarray(model.predict(x_test_hie))
            model_names.append('H_RNN')
            model_probas.append(pred_h_rnn)
            y_h_rnn = pred_h_rnn.round()
            f1_h_rnn = f1_score(y_test.T[1], y_h_rnn.T[1])
            recall_h_rnn = recall_score(y_test.T[1], y_h_rnn.T[1])
            precision_h_rnn = precision_score(y_test.T[1], y_h_rnn.T[1])
            acc_h_rnn = accuracy_score(y_test.T[1], y_h_rnn.T[1])
            print('H_RNN model performance: acc = %0.4f, f1 = %0.4f, precision = %0.4f, recall = %0.4f'
                  % (acc_h_rnn, f1_h_rnn, precision_h_rnn, recall_h_rnn))
        elif 'RNN_att' in m:
            model = RNN_attention(MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
                                  embedding_layer=embedding_layer,
                                  RNN_gate_type='LSTM',
                                  RNN_units=128,
                                  dropout=0.5,
                                  label_dim=2)
            model.load_weights(m)
            pred_rnn_att = np.asarray(model.predict(x_test_seq))
            model_names.append('RNN_att')
            model_probas.append(pred_rnn_att)
            y_rnn_att = pred_rnn_att.round()
            f1_rnn_att = f1_score(y_test.T[1], y_rnn_att.T[1])
            recall_rnn_att = recall_score(y_test.T[1], y_rnn_att.T[1])
            precision_rnn_att = precision_score(y_test.T[1], y_rnn_att.T[1])
            acc_rnn_att = accuracy_score(y_test.T[1], y_rnn_att.T[1])
            print('RNN_att model performance: acc = %0.4f, f1 = %0.4f, precision = %0.4f, recall = %0.4f'
                  % (acc_rnn_att, f1_rnn_att, precision_rnn_att, recall_rnn_att))
        elif 'RNN' in m:
            model = RNN_simple(MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
                               embedding_layer=embedding_layer,
                               RNN_gate_type='LSTM',
                               RNN_units=128,
                               dropout=0.5,
                               label_dim=2)
            model.load_weights(m)
            pred_rnn = np.asarray(model.predict(x_test_seq))
            model_names.append('RNN')
            model_probas.append(pred_rnn)
            y_rnn = pred_rnn.round()
            f1_rnn = f1_score(y_test.T[1], y_rnn.T[1])
            recall_rnn = recall_score(y_test.T[1], y_rnn.T[1])
            precision_rnn = precision_score(y_test.T[1], y_rnn.T[1])
            acc_rnn = accuracy_score(y_test.T[1], y_rnn.T[1])
            print('RNN model performance: acc = %0.4f, f1 = %0.4f, precision = %0.4f, recall = %0.4f'
                  % (acc_rnn, f1_rnn, precision_rnn, recall_rnn))

    def combined_model_metrics(y_test, y_pred, model_num):
        pred = (y_pred / model_num).round().T[1]
        acc = np.mean(pred == y_test)
        f1 = f1_score(y_test, pred)
        recall = recall_score(y_test, pred)
        precision = precision_score(y_test, pred)
        return acc, f1, precision, recall


    # SVM model
    clf_svm = svm.SVC(kernel='linear', C=1, probability=True)
    clf_svm.fit(x_train_tfidf, y_train_tfidf)
    pred_svm = clf_svm.predict_proba(x_test_tfidf)
    model_names.append('SVM')
    model_probas.append(pred_svm)
    acc_svm, f1_svm, precision_svm, recall_svm = combined_model_metrics(y_test_tfidf, pred_svm, 1)
    print('SVM model performance: acc = %0.4f, f1 = %0.4f, precision = %0.4f, recall = %0.4f'
          % (acc_svm, f1_svm, precision_svm, recall_svm))

    # LR model
    clf_lr = LogisticRegression()
    clf_lr.fit(x_train_tfidf, y_train_tfidf)
    pred_lr = clf_lr.predict_proba(x_test_tfidf)
    model_names.append('LR')
    model_probas.append(pred_lr)
    acc_lr, f1_lr, precision_lr, recall_lr = combined_model_metrics(y_test_tfidf, pred_lr, 1)
    print('LR model performance: acc = %0.4f, f1 = %0.4f, precision = %0.4f, recall = %0.4f'
          % (acc_lr, f1_lr, precision_lr, recall_lr))


    # Combined models
    print('=== Combined models ===')
    model_names_combined = []
    model_metrics_combined = []

    # 7 models
    print('- 7 models ... -')
    name_current = '7-' + model_names[0] + '-' + model_names[1] + '-' + model_names[2] + '-' + model_names[3]\
                   + '-' + model_names[4] + '-' + model_names[5] + '-' + model_names[6] + '-'
    proba_current = model_probas[0] + model_probas[1] + model_probas[2] + model_probas[3] + model_probas[4]\
                    + model_probas[5] + model_probas[6]
    model_names_combined.append(name_current)
    model_metrics_combined.append(combined_model_metrics(y_test_tfidf, proba_current, 7))
    print(len(model_names_combined), ':', name_current, '- acc -', model_metrics_combined[0][0])

    # 6 models
    print('- 6 models ... -')
    for i in range(len(model_names)):
        name_current = '6-NO-' + model_names[i] + '-'
        proba_current = np.zeros((len(model_probas[0]), len(model_probas[0][0])))
        for j in range(len(model_probas)):
            if i != j:
                proba_current += model_probas[j]
        model_names_combined.append(name_current)
        model_metrics_combined.append(combined_model_metrics(y_test_tfidf, proba_current, 6))
        print(len(model_names_combined), ':', name_current, '- acc -',
              model_metrics_combined[len(model_metrics_combined) - 1][0])

    # 5 and 2 models
    print('- 5 and 2 models ... -')
    for i in range(len(model_names)):
        for j in range((i + 1), len(model_names)):
            name_current_2 = '2-' + model_names[i] + '-' + model_names[j] + '-'
            name_current_5 = '5-NO-' + model_names[i] + '-' + model_names[j] + '-'
            proba_current_2 = np.zeros((len(model_probas[0]), len(model_probas[0][0])))
            proba_current_5 = np.zeros((len(model_probas[0]), len(model_probas[0][0])))
            for k in range(len(model_probas)):
                if k == i or k == j:
                    proba_current_2 += model_probas[k]
                else:
                    proba_current_5 += model_probas[k]
            model_names_combined.append(name_current_2)
            model_metrics_combined.append(combined_model_metrics(y_test_tfidf, proba_current_2, 2))
            print(len(model_names_combined), ':', name_current_2, '- acc -',
                  model_metrics_combined[len(model_metrics_combined) - 1][0])
            model_names_combined.append(name_current_5)
            model_metrics_combined.append(combined_model_metrics(y_test_tfidf, proba_current_5, 5))
            print(len(model_names_combined), ':', name_current_5, '- acc -',
                  model_metrics_combined[len(model_metrics_combined) - 1][0])

    # 4 and 3 models
    print('- 4 and 3 models ... -')
    for i in range(len(model_names)):
        for j in range((i + 1), len(model_names)):
            for k in range((j + 1), len(model_names)):
                name_current_3 = '3-' + model_names[i] + '-' + model_names[j] + '-' + model_names[k] + '-'
                name_current_4 = '4-NO-' + model_names[i] + '-' + model_names[j] + '-' + model_names[k] + '-'
                proba_current_3 = np.zeros((len(model_probas[0]), len(model_probas[0][0])))
                proba_current_4 = np.zeros((len(model_probas[0]), len(model_probas[0][0])))
                for m in range(len(model_probas)):
                    if m == i or m == j or m == k:
                        proba_current_3 += model_probas[m]
                    else:
                        proba_current_4 += model_probas[m]
                model_names_combined.append(name_current_3)
                model_metrics_combined.append(combined_model_metrics(y_test_tfidf, proba_current_3, 3))
                print(len(model_names_combined), ':', name_current_3, '- acc -',
                      model_metrics_combined[len(model_metrics_combined) - 1][0])
                model_names_combined.append(name_current_4)
                model_metrics_combined.append(combined_model_metrics(y_test_tfidf, proba_current_4, 4))
                print(len(model_names_combined), ':', name_current_4, '- acc -',
                      model_metrics_combined[len(model_metrics_combined) - 1][0])

    def get_full_name(model_names_combined):
        name_list = ['H_RNN_att', 'H_RNN', 'RNN_att', 'RNN', 'CNN', 'SVM', 'LR']
        full_names = []
        for i in range(len(model_names_combined)):
            full_name = '-'
            for j in range(len(name_list)):
                if '-NO-' in model_names_combined[i]:
                    if '-'+name_list[j]+'-' not in model_names_combined[i]:
                        full_name += name_list[j] + '-'
                else:
                    if '-'+name_list[j]+'-' in model_names_combined[i]:
                        full_name += name_list[j] + '-'
            full_names.append(full_name)
        return full_names

    # Output 1
    print('=== Output ===')
    output = DataFrame(model_metrics_combined)
    output.columns = ['Accuracy', 'F1 score', 'Precision', 'Recall']
    output['Model'] = model_names_combined
    output['Full name'] = get_full_name(model_names_combined)
    cols = list(output)
    cols.insert(0, cols.pop(cols.index('Full name')))
    cols.insert(0, cols.pop(cols.index('Model')))
    output = output.ix[:, cols]
    output = output.sort_values(by='Accuracy', ascending=False)
    output.index = range(1, len(model_names_combined) + 1)
    output.index.name = 'Rank'
    output.to_csv('logs/Metrics_of_combined_models.csv')
    print('Written to logs/Metrics_of_combined_models.csv')