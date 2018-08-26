import pandas as pd
import re
import os
from bs4 import BeautifulSoup
from nltk import tokenize
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from model_hatt import *
from model_textCNN import *


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", str(string))
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


def read_IMDB():
    """
    Read IMDB data downloaded from website

    Return:
    reviews -- List[document #][sentence #], tokenized sentences
    labels -- List[document #], sentiment labels
    texts -- List[document #], list of individual reviews
    """
    print("Reading raw data ...")
    data_train = pd.read_csv('./data/labeledTrainData.tsv', sep='\t')
    print("Raw data shape: " + str(data_train.shape))

    reviews = []
    labels = []
    texts = []

    print("Processing sentence tokenization ...")
    for idx in range(data_train.review.shape[0]):
        text = BeautifulSoup(data_train.review[idx], "html5lib")
        text = clean_str(text.get_text().encode('ascii', 'ignore'))
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        reviews.append(sentences)
        labels.append(data_train.sentiment[idx])

    return reviews, labels, texts


def read_IMDB_by_document():
    """
    Read IMDB data downloaded from website

    Return:
    labels -- List[document #], sentiment labels
    texts -- List[document #], list of individual reviews
    """
    print("Reading raw data ...")
    data_train = pd.read_csv('./data/labeledTrainData.tsv', sep='\t')
    print("Raw data shape: " + str(data_train.shape))

    labels = []
    texts = []

    print("Processing sentence tokenization ...")
    for idx in range(data_train.review.shape[0]):
        text = BeautifulSoup(data_train.review[idx], "html5lib")
        text = clean_str(text.get_text().encode('ascii', 'ignore'))
        texts.append(text)
        labels.append(data_train.sentiment[idx])

    return labels, texts


def tokenazation(reviews, labels, texts, MAX_NUM_WORDS, MAX_SENTS, MAX_SENT_LENGTH):
    """
    Tokenization For hierarchical RNN

    Arguments:
    reviews -- List[document #][sentence #], tokenized sentences
    labels -- List[document #], sentiment labels
    texts -- List[document #], list of individual reviews
    MAX_NUM_WORDS -- Positive integer, the maximum number of words to keep in a document, based on word frequency
    MAX_SENTS -- Positive integer, maximum sentence # in a document
    MAX_SENT_LENGTH -- Positive integer, maximum word # in a sentence

    Return:
    data -- shape:(document #, sentence #, word #), word_index after tokenization
    labels -- shape:(document #, label #)
    word_index -- dict, key: tokens ranked by frequence, value: ranking index
    """
    print("Fitting text by MAX_NUM_WORDS = " + str(MAX_NUM_WORDS) + " ...")
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)

    data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

    print("Putting word token indexes into input matrix ...")
    for i, sentences in enumerate(reviews):
        for j, sent in enumerate(sentences):
            if j < MAX_SENTS:
                wordTokens = text_to_word_sequence(sent)
                k = 0
                for _, word in enumerate(wordTokens):
                    if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NUM_WORDS:
                        data[i, j, k] = tokenizer.word_index[word]
                        k = k + 1

    word_index = tokenizer.word_index
    print("Total unique tokens: " + str(len(word_index)))

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    return data, labels, word_index


def tokenazation_CNN(labels, texts, MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH):
    """
    Tokenization for CNN

    Arguments:
    labels -- List[document #], sentiment labels
    texts -- List[document #], list of individual reviews
    MAX_NUM_WORDS -- Positive integer, the maximum number of words to keep in a document, based on word frequency
    MAX_SEQUENCE_LENGTH -- Positive integer, maximum sequence #

    Return:
    data -- shape:(document #, MAX_SEQUENCE_LENGTH #), word_index after tokenization
    labels -- shape:(document #, label #)
    word_index -- dict, key: tokens ranked by frequence, value: ranking index
    """
    print("Fitting text by MAX_NUM_WORDS = " + str(MAX_NUM_WORDS) + " ...")
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print("Total unique tokens: " + str(len(word_index)))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    return data, labels, word_index


def prepare_train_dev(data, labels, VALIDATION_SPLIT):
    """
    Preparing training and developing data

    Arguments:
    data -- shape:(document #, sentence #, word #), word_index after tokenization
    labels -- shape:(document #, label #)
    VALIDATION_SPLIT -- digit between [0, 1], dev set ratio

    Return:
    x_train -- shape:(document #, sentence #, word #), x train
    y_train -- shape:(document #, label #), y train
    x_val -- shape:(document #, sentence #, word #), x dev
    y_val -- shape:(document #, label #), y dev
    """
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    print("Dataset Shuffled.")

    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]
    print(str(len(x_train)) + " (" + str(1 - VALIDATION_SPLIT) + ") for training, " +
        str(len(x_val)) + " (" + str(VALIDATION_SPLIT) + ") for validation.")

    print('Number of positive and negative reviews in training and validation set: ')
    print(y_train.sum(axis=0))
    print(y_val.sum(axis=0))

    return x_train, y_train, x_val, y_val


def calculate_embedding_matrix(EMB_DIM, word_index):
    """
    Generate word embedding matrix for embedding layer establishment

    Arguments:
    GLOVE_DIR -- String, GloVe file direction
    EMB_DIM -- Positive integer, word embedding dimension, i.e., 50/100/200/300
    word_index -- dict, key: tokens ranked by frequence, value: ranking index

    Return:
    embedding_matrix -- shape:((len(word_index) + 1, EMB_DIM), embedding matrix for building embedding layer
    """
    embeddings_index = {}
    f = open(os.path.join('./data/glove.6B.' + str(EMB_DIM) + 'd.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print("Total tokens in word embedding resource: " + str(len(embeddings_index)))
    print("Dimensions of word embedding: " + str(EMB_DIM))

    embedding_matrix = np.random.random((len(word_index) + 1, EMB_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix
