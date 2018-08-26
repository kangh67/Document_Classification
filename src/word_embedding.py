import numpy as np
from keras.layers.embeddings import Embedding


def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similarity between u and v

    Arguments:
        u -- a word vector of shape (n,)
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """

    distance = 0.0

    # Compute the dot product between u and v (≈1 line)
    dot = np.dot(u, v)
    # Compute the L2 norm of u (≈1 line)
    norm_u = np.sqrt(np.sum(u * u))

    # Compute the L2 norm of v (≈1 line)
    norm_v = np.sqrt(np.sum(v * v))
    # Compute the cosine similarity defined by formula (1) (≈1 line)
    cosine_similarity = dot / (norm_u * norm_v)

    return cosine_similarity


def test_cosine_similarity(word_to_vec_map):
    """
    Test above function - cosine_similarity()
    Expect output (50 dimensions):

        cosine_similarity(father, mother) =  0.890903844289
        cosine_similarity(ball, crocodile) =  0.274392462614
        cosine_similarity(france - paris, rome - italy) =  -0.675147930817

    Arguments:
    word_to_vec_map: dictionary mapping words to their GloVe vector representation.
    """

    father = word_to_vec_map["father"]
    mother = word_to_vec_map["mother"]
    ball = word_to_vec_map["ball"]
    crocodile = word_to_vec_map["crocodile"]
    france = word_to_vec_map["france"]
    italy = word_to_vec_map["italy"]
    paris = word_to_vec_map["paris"]
    rome = word_to_vec_map["rome"]

    print("cosine_similarity(father, mother) = ", cosine_similarity(father, mother))
    print("cosine_similarity(ball, crocodile) = ", cosine_similarity(ball, crocodile))
    print("cosine_similarity(france - paris, rome - italy) = ", cosine_similarity(france - paris, rome - italy))


def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task as explained above: a is to b as c is to ____.

    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_to_vec_map -- dictionary that maps words to their corresponding vectors.

    Returns:
    best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
    """

    # convert words to lower case
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()

    # Get the word embeddings v_a, v_b and v_c (≈1-3 lines)
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]

    words = word_to_vec_map.keys()
    max_cosine_sim = -100  # Initialize max_cosine_sim to a large negative number
    best_word = None  # Initialize best_word with None, it will help keep track of the word to output

    # loop over the whole word vector set
    for w in words:
        # to avoid best_word being one of the input words, pass on them.
        if w in [word_a, word_b, word_c]:
            continue

        # Compute cosine similarity between the vector (e_b - e_a) and the vector ((w's vector representation) - e_c)  (≈1 line)
        cosine_sim = cosine_similarity((e_b - e_a), (word_to_vec_map[w] - e_c))

        # If the cosine_sim is more than the max_cosine_sim seen so far,
        # then: set the new max_cosine_sim to the current cosine_sim and the best_word to the current word (≈3 lines)
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w

    return best_word


def test_complete_analogy(word_to_vec_map):
    """
    Test above function - complete_analogy()
    Expect output (50 dimensions):

        italy -> italian :: spain -> spanish
        india -> delhi :: japan -> tokyo
        man -> woman :: boy -> girl
        small -> smaller :: large -> larger
    """

    triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'),
                     ('small', 'smaller', 'large')]
    for triad in triads_to_try:
        print('{} -> {} :: {} -> {}'.format(*triad, complete_analogy(*triad, word_to_vec_map)))


def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4 of Emojify-V2).

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """

    m = X.shape[0]  # number of training examples

    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len))

    for i in range(m):  # loop over training examples

        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = X[i].lower().split()

        # Initialize j to 0
        j = 0

        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            X_indices[i, j] = word_to_index[w]
            # Increment j to j + 1
            j = j + 1

    return X_indices


def test_sentences_to_indices(word_to_index):
    """
    Test above function - sentences_to_indices()
    Expected output:

        X1 = ['funny lol' 'lets play baseball' 'food is ready for you']
        X1_indices = [[ 155345.  225122.       0.       0.       0.]
        [ 220930.  286375.   69714.       0.       0.]
        [ 151204.  192973.  302254.  151349.  394475.]]

    Arguments:
    word_to_index -- a dictionary containing the each word mapped to its index
    """

    X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
    X1_indices = sentences_to_indices(X1, word_to_index, max_len=5)
    print("X1 =", X1)
    print("X1_indices =", X1_indices)


def pretrained_embedding_layer(word_to_vec_map, word_to_index, trainable):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)
    trainable -- Fix pre-trained word embedding (True) or make it trainable (False)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    vocab_len = len(word_to_index) + 1  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]  # define dimensionality of your GloVe word vectors (= 50)

    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))

    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False.
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=trainable)

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))

    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pre-trained.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


def test_pretrained_embedding_layer(word_to_vec_map, word_to_index, trainable):
    """
    Test above function - pretrained_embedding_layer()
    Expected output (50 dimension):

        weights[0][1][3] = -0.3403
    """
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index, trainable)
    print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])


### Test fuctions

# GloVe pre-trained word embeddings, 400,001 words, 50/100/200/300 dimensions
# word_to_index: dictionary mapping from words to their indices in the vocabulary (400,001 words, with the valid indices ranging from 0 to 400,000)
# index_to_word: dictionary mapping from indices to their corresponding words in the vocabulary
# word_to_vec_map: dictionary mapping words to their GloVe vector representation.

"""
from utils_emo import *

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

test_cosine_similarity(word_to_vec_map)
test_complete_analogy()
test_sentences_to_indices(word_to_index)
test_pretrained_embedding_layer(word_to_vec_map, word_to_index, trainable=False)
"""

###
