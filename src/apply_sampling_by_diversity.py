# Sampling from filtered data by using diversity strategy
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from scipy.spatial import distance


# MAUDE 2017 filtered data
dev17_filtered = 'data/dev_2017_filtered.txt'

# MAUDE 2017 tsv file for classifier
maude_2017 = 'data/MAUDE_2017_noLabel.tsv'

# (OUTPUT) MAUDE 2017 text vector file
maude_2017_vector = 'data/MAUDE_2017_vectors.txt'


# 1 = calculate vectors; 2 = sampling
mode = 1


# === Calculate vectors for filtered texts by using Universal Sentence Encoder
if mode == 1:
    print('=== Calculate vectors ===')
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
    embed = hub.Module(module_url)
    tf.logging.set_verbosity(tf.logging.ERROR)
    session = tf.Session()
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])

    status = 1  # track status
    with open(maude_2017, 'r') as r:
        line = r.readline()
        w = open(maude_2017_vector, 'w')
        w.write('MDR\tVECTOR\n')
        while True:
            line = r.readline()
            if not line:
                break
            else:
                v = session.run(embed([line.split('\t')[1]]))
                print(status, '...', line.split('\t')[0], ':', v[0][:3], '...')
                w.write(line.split('\t')[0] + '\t' + str(v[0]) + '\n')
                status += 1
        w.close()

    session.close()
    print('Vectors were saved to:', maude_2017_vector)



'''
vectors = []    # 512 dimension vectors
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
embed = hub.Module(module_url)
tf.logging.set_verbosity(tf.logging.ERROR)
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    vectors = session.run(embed(np.asarray(mdr_text.TEXT)))

print(len(vectors))
print(type(vectors))
print(vectors[0])
print(vectors[len(vectors)-1])
'''



'''

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

# Compute a representation for each message, showing various lengths supported.
p1 = "Universal Sentence Encoder embeddings also support short paragraphs. " \
     "There is no hard limit on how long the paragraph is. Roughly, " \
     "the longer the more 'diluted' the embedding will be."
p2 = "Universal Sentence Encoder embeddings also support short paragraphs. " \
     "There is no hard limit on how long the paragraph is."

p = [p1, p2]

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    v = session.run(embed(p))
    print(distance.euclidean(v[0], v[1]))
'''