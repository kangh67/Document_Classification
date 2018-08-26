# classifiers of baseline models, logistic regression, random forest, SVM

from nlp_MAUDE import *
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics


# === Hyper parameters ===
EMB_DIM = 300

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


# Tokenization word embedding
print("=== Tokenizing for word embedding ===")
data_we, labels, word_index = tokenazation_CNN(labels, documents, MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH)


# Generate word embedding matrix for embedding layer establishment
print("=== Generating word embedding matrix ===")
embedding_matrix = calculate_embedding_matrix(EMB_DIM, word_index)


# Use the mean value of word embeddings as the features of document
print("=== Calculating the mean of word embeddings ===")
data_we = mean_emb(data_we, embedding_matrix)


# Prepare training, developing, and testing data
if len(DATA_SPLIT) == 2:
    print("=== Preparing training (" + str(10 * DATA_SPLIT[0]) + "%) and developing data (" +
          str(10 * DATA_SPLIT[1]) + "%) ===")
elif len(DATA_SPLIT) == 3:
    print("=== Preparing training (" + str(10 * DATA_SPLIT[0]) + "%), developing data (" +
          str(10 * DATA_SPLIT[1]) + "%), and testing data (" + str(10 * DATA_SPLIT[2]) + "%) ===")
else:
    sys.exit("ERROR: Length of DATA_SPLIT should be either 2 or 3.")
# word embedding features
x_train_we, x_train_tfidf, y_train, x_test_we, x_test_tfidf, y_test = \
    prepare_train_dev_baseline(data_we, np.asarray(documents), labels, DATA_SPLIT)


# Calculate tfidf
print("=== Calculating tfidf ===")
x_train_tfidf, x_test_tfidf = tokenazation_tfidf(x_train_tfidf, x_test_tfidf)


y_train = y_train.T[1]
y_test = y_test.T[1]


# train and test Naive Bayes
clf_lr = BernoulliNB()

print('====== Naive Bayes model ======')

print('-- NB mean word embeddings ---')
print('Training on', x_train_we.shape[0], 'samples (train set) ...')
clf_lr.fit(x_train_we, y_train)

print('Predicting on', x_test_we.shape[0], 'samples (test set) ...')
predicted_test = clf_lr.predict(x_test_we)
print('Accuracy on test: {:4.4}'.format(np.mean(predicted_test == y_test)))
print(metrics.classification_report(y_test, predicted_test, digits=4))

print('-- NB tfidf ---')
print('Training on', x_train_tfidf.shape[0], 'samples (train set) ...')
clf_lr.fit(x_train_tfidf, y_train)

print('Predicting on', x_test_tfidf.shape[0], 'samples (test set) ...')
predicted_test = clf_lr.predict(x_test_tfidf)
print('Accuracy on test: {:4.4}'.format(np.mean(predicted_test == y_test)))
print(metrics.classification_report(y_test, predicted_test, digits=4))


# train and test logistic regression
clf_lr = LogisticRegression()

print('====== Logistic Regression model ======')

print('-- LR mean word embeddings ---')
print('Training on', x_train_we.shape[0], 'samples (train set) ...')
clf_lr.fit(x_train_we, y_train)

print('Predicting on', x_test_we.shape[0], 'samples (test set) ...')
predicted_test = clf_lr.predict(x_test_we)
print('Accuracy on test: {:4.4}'.format(np.mean(predicted_test == y_test)))
print(metrics.classification_report(y_test, predicted_test, digits=4))

print('-- LR tfidf ---')
print('Training on', x_train_tfidf.shape[0], 'samples (train set) ...')
clf_lr.fit(x_train_tfidf, y_train)

print('Predicting on', x_test_tfidf.shape[0], 'samples (test set) ...')
predicted_test = clf_lr.predict(x_test_tfidf)
print('Accuracy on test: {:4.4}'.format(np.mean(predicted_test == y_test)))
print(metrics.classification_report(y_test, predicted_test, digits=4))


# train and test Random Forest model
clf_rf = RandomForestClassifier()

print("====== Random Forest model ======")

print('-- RF mean word embeddings ---')
print('Training on', x_train_we.shape[0], 'samples (train set) ...')
clf_rf.fit(x_train_we, y_train)

print('Predicting on', x_test_we.shape[0], 'samples (test set) ...')
predicted_test = clf_rf.predict(x_test_we)
print('Accuracy on test: {:4.4f}'.format(np.mean(predicted_test == y_test)))
print(metrics.classification_report(y_test, predicted_test, digits=4))

print('-- RF tfidf ---')
print('Training on', x_train_tfidf.shape[0], 'samples (train set) ...')
clf_rf.fit(x_train_tfidf, y_train)

print('Predicting on', x_test_tfidf.shape[0], 'samples (test set) ...')
predicted_test = clf_rf.predict(x_test_tfidf)
print('Accuracy on test: {:4.4f}'.format(np.mean(predicted_test == y_test)))
print(metrics.classification_report(y_test, predicted_test, digits=4))


# train and test SVM model
clf_svm = svm.SVC(kernel='linear', C=1)

print("====== SVM model ======")

print('-- SVM mean word embeddings ---')
print('Training on', x_train_we.shape[0], 'samples (train set) ...')
clf_svm.fit(x_train_we, y_train)

print('Predicting on', x_test_we.shape[0], 'samples (test set) ...')
predicted_test = clf_svm.predict(x_test_we)
print('Accuracy on test: {:4.4f}'.format(np.mean(predicted_test == y_test)))
print(metrics.classification_report(y_test, predicted_test, digits=4))

print('-- SVM tfidf ---')
print('Training on', x_train_tfidf.shape[0], 'samples (train set) ...')
clf_svm.fit(x_train_tfidf, y_train)

print('Predicting on', x_test_tfidf.shape[0], 'samples (test set) ...')
predicted_test = clf_svm.predict(x_test_tfidf)
print('Accuracy on test: {:4.4f}'.format(np.mean(predicted_test == y_test)))
print(metrics.classification_report(y_test, predicted_test, digits=4))
