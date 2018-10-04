# Apply classifiers on the .tsv file of MAUDE 2017 filtered data
from nlp_MAUDE import *
from sklearn.linear_model import LogisticRegression


EMB_DIM = 300
trainable = True

MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000
MAX_SENTS = 20
MAX_SENT_LENGTH = 100


# MAUDE 2017 tsv file for classifier
maude_2017 = 'data/MAUDE_2017_noLabel.tsv'

# Training data, 70% of 2008-2016 MAUDE data, used by baseline models
train_file = './data/MAUDE_train.tsv'


# === read file ===
print("Reading prediction data from", maude_2017)
data = pd.read_csv(maude_2017, sep='\t')
print("prediction data shape: " + str(data.shape))

documents_sent = []
documents = []
for idx in range(data.TEXT.shape[0]):
    documents.append(data.TEXT[idx])
    sentences = tokenize.sent_tokenize(data.TEXT[idx])
    documents_sent.append(sentences)
documents = np.asarray(documents)


# === Tokenization ===
print('=== Tokenization ===')
doc_hie_all, y_all, doc_all = read_MAUDE_hierarchical_simple()
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(doc_all)
word_index = tokenizer.word_index
print("Total unique tokens: " + str(len(word_index)))

# sequence input, for CNN, RNN, and RNN_att models
print('=== Format data ===')
sequences = tokenizer.texts_to_sequences(documents)
x_seq = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('data shape of seq:', x_seq.shape)

# hierarchical input, for H_RNN and H_RNN_att models
x_hie = tokenazation_combined_models(word_index, documents_sent, MAX_NUM_WORDS, MAX_SENTS, MAX_SENT_LENGTH)
print('data shape of hie:', x_hie.shape)

# TFIDF input, for baseline models
data_train = pd.read_csv(train_file, sep='\t')
_, y_train, doc_train = read_MAUDE_train_dev_test(train_file)
x_train_tfidf, x_tfidf = tokenazation_tfidf(doc_train, documents)
y_train_tfidf = y_train.T[1]


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


# === Build CNN, H_RNN models ===
# CNN model
model_CNN = CNN_complex(MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
                        embedding_layer=embedding_layer,
                        filter_num=256,
                        kernal_size=[2, 3, 4],
                        label_dim=2,
                        dropout_CNN=0.3,
                        dropout_full_con=0.5)
# H_RNN model
model_H_RNN = hierachical_network(MAX_SENTS=MAX_SENTS,
                                  MAX_SENT_LENGTH=MAX_SENT_LENGTH,
                                  embedding_layer=embedding_layer_H,
                                  RNN_gate_type='LSTM',
                                  RNN_units=128,
                                  dropout=0.5,
                                  label_dim=2)


print('=== Applying models ===')
print('Making prediction using CNN model ...')
model_CNN.load_weights('./models/best_weights/CNN-0.8919-0.8639.hdf5')
pred_CNN = np.asarray(model_CNN.predict(x_seq))

print('Making prediction using H_RNN model ...')
model_H_RNN.load_weights('./models/best_weights/H_RNN-0.8919-0.8648.hdf5')
pred_H_RNN = np.asarray(model_H_RNN.predict(x_hie))

print('Training LR model ...')
clf_lr = LogisticRegression()
clf_lr.fit(x_train_tfidf, y_train_tfidf)
print('Making prediction using LR model ...')
pred_lr = clf_lr.predict_proba(x_tfidf)


print('=== Results ===')
print('- Model - 0 - 1 -')
print('CNN:', pred_CNN.round().sum(axis=0))
print('H_RNN:', pred_H_RNN.round().sum(axis=0))
print('LR:', pred_lr.round().sum(axis=0))
print('Hybrid:', ((pred_CNN + pred_H_RNN + pred_lr) / 3).round().sum(axis=0))

overlap_CH = 0
overlap_CL = 0
overlap_HL = 0
overlap_all = 0

for i in range(len(pred_CNN)):
    if pred_CNN.round().T[1][i] == pred_H_RNN.round().T[1][i]:
        overlap_CH += 1
    if pred_CNN.round().T[1][i] == pred_lr.round().T[1][i]:
        overlap_CL += 1
    if pred_H_RNN.round().T[1][i] == pred_lr.round().T[1][i]:
        overlap_HL += 1
    if pred_CNN.round().T[1][i] == pred_lr.round().T[1][i] and pred_CNN.round().T[1][i] == pred_H_RNN.round().T[1][i]:
        overlap_all += 1

print('Overlap CNN & H_RNN:', overlap_CH, '/', len(pred_CNN))
print('Overlap CNN & LR:', overlap_CL, '/', len(pred_CNN))
print('Overlap LR & H_RNN:', overlap_HL, '/', len(pred_CNN))
print('Overlap all:', overlap_all, '/', len(pred_CNN))