# Sampling from filtered data by using diversity strategy
import tensorflow as tf
import tensorflow_hub as hub
import sys
import re
import numpy as np
import pandas as pd
from scipy.spatial import distance

# MAUDE 2017 filtered data
dev17_filtered = 'data/dev_2017_filtered.txt'

# MAUDE 2017 tsv file for classifier
maude_2017 = 'data/MAUDE_2017_noLabel.tsv'

# (OUTPUT) MAUDE 2017 text vector file
maude_2017_vector = 'data/MAUDE_2017_vectors.txt'

# (OUTPUT) MAUDE 2017 with Euclidean distance
maude_2017_ed = 'data/MAUDE_2017_text_ed.txt'

# 1 = calculate vectors; 2 = sampling
mode = 2

# Reset cutoff of the USE session, only works when mode = 1
reset_cutoff = 30


# === Calculate vectors for filtered texts by using Universal Sentence Encoder
if mode == 1:
    print('=== Calculate vectors ===')
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
    embed = hub.Module(module_url)
    tf.logging.set_verbosity(tf.logging.ERROR)
    session = tf.Session()
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])

    file_exist = True  # if the vector file has been already created

    try:
        f = open(maude_2017_vector, 'r')
        f.close()
    except FileNotFoundError:
        file_exist = False
        print('First vectorization.')

    status = 1  # track status
    with open(maude_2017, 'r') as r:
        line_o = r.readline()
        # Not the first time initialization
        if file_exist:
            w = open(maude_2017_vector, 'r')
            while True:
                line_v = w.readline()
                if not line_o:
                    print('Vectorization completed!')
                    w.close()
                    sys.exit()
                if not line_v or line_v == '\n':
                    print(str(status - 2), 'reports have been vectorized before.')
                    w.close()
                    break
                else:
                    if line_o.split('\t')[0] == line_v.split('\t')[0]:
                        print(str(status - 1), '...', line_o.split('\t')[0], ': has been vectorized before')
                        status += 1
                        line_o = r.readline()
                    else:
                        print('Error: line', status, 'does not match:', line_o.split('\t')[0], 'vs',
                              line_v.split('\t')[0])
                        w.close()
                        sys.exit()
            w.close()

        if status == 1:
            with open(maude_2017_vector, 'a+') as w:
                w.write('MDR\tVECTOR\n')
            line_o = r.readline()
            status += 1
        reset = 0
        while True:
            if not line_o:
                print('Vectorization completed!')
                break
            else:
                # reset the session to speed up
                if reset == reset_cutoff:
                    session.close()
                    embed = hub.Module(module_url)
                    tf.logging.set_verbosity(tf.logging.ERROR)
                    session = tf.Session()
                    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
                    print('Session reset every', reset_cutoff, 'reports')
                    reset = 0
                v = session.run(embed([line_o.split('\t')[1]]))
                print(str(status - 1), '...', line_o.split('\t')[0], ':', v[0][:3], '...')
                vec = str(v[0]).replace('[ ', '[').replace(' ]', ']').replace('\n', '')
                vec = re.sub(' +', ',', vec)
                with open(maude_2017_vector, 'a+') as w:
                    w.write(line_o.split('\t')[0] + '\t' + vec + '\n')
                line_o = r.readline()
                status += 1
                reset += 1

    session.close()
    print('Vectors were saved to:', maude_2017_vector)


# === Sampling according to Euclidean distance
elif mode == 2:
    # Read mdr_vector file
    def interpret_one_line(line):
        line = line.replace('\n', '').split('\t')
        mdr = line[0]
        vector = line[1].replace('[', '').replace(',]', '').replace(']', '').split(',')
        if len(vector) != 512:
            print('The vector dimension of this report is', len(vector), 'less than 512:', mdr)
        vector = [float(x) for x in vector]
        MDR_vector[mdr] = np.asarray(vector)

    print('=== Read report vectors ===')
    MDR_vector = dict()
    with open(maude_2017_vector, 'r') as r:
        line = r.readline()
        while True:
            line = r.readline()
            if not line:
                break
            else:
                interpret_one_line(line)
    print('Found', len(MDR_vector), 'vectors from', maude_2017_vector)

    print('=== Group reports by generic names ===')
    mdrs = set()
    generic_mdrSet = dict()
    with open(dev17_filtered, 'r') as r:
        line = r.readline()
        while True:
            line = r.readline()
            if not line:
                break
            else:
                line = line.split('|')
                # Do not consider the mdrs who do not have text vectors
                if line[0] not in MDR_vector.keys() or line[0] in mdrs:
                    continue
                mdrs.add(line[0])
                if line[7] in generic_mdrSet.keys():
                    generic_mdrSet[line[7]].add(line[0])
                else:
                    newSet = set()
                    newSet.add(line[0])
                    generic_mdrSet[line[7]] = newSet
    sum_report = 0
    for gen in generic_mdrSet.keys():
        sum_report += len(generic_mdrSet[gen])
    print('Unique generic names:', len(generic_mdrSet))
    print('Unique MDRs with text vectors:', sum_report)

    # Read MDR_TEXT file
    mdr_text = dict()
    with open(maude_2017, 'r') as r:
        line = r.readline()
        while True:
            line = r.readline()
            if not line:
                break
            else:
                line = line.split('\t')
                mdr_text[line[0]] = line[1].replace('\n', '')
    print(len(mdr_text), 'MDR have text')

    print('=== Calculating Euclidean distances ===')

    # Sample by generic names
    ed = open(maude_2017_ed, 'w')
    ed.write('MDR\tGENERIC_NAME\tTEXT\tEUCLIDEAN_DISTANCE\n')
    count_ge = 0
    for ge in generic_mdrSet.keys():
        count_ge += 1
        print(count_ge, '-', ge, ':', len(generic_mdrSet[ge]))
        # Sampled MDR
        inclu_mdr = set()
        # Randomly pick the first mdr
        mdr_root = generic_mdrSet[ge].pop()
        current_vec = MDR_vector[mdr_root]
        inclu_mdr.add(mdr_root)
        ed.write(mdr_root + '\t' + ge + '\t' + mdr_text[mdr_root] + '\t1.0\n')
        while len(generic_mdrSet[ge]) > 0:
            mdr_distance = dict()
            # The furthest distance by now
            furthest_dis = 0
            # The furthest MDR by now
            furthest_mdr = mdr_root
            for m in generic_mdrSet[ge]:
                dis = distance.euclidean(current_vec, MDR_vector[m])
                mdr_distance[m] = dis
                # update if further one is found
                if dis >= furthest_dis:
                    furthest_dis = dis
                    furthest_mdr = m

            current_vec = (current_vec * len(inclu_mdr) + MDR_vector[furthest_mdr]) / (len(inclu_mdr) + 1)
            inclu_mdr.add(furthest_mdr)
            ed.write(furthest_mdr + '\t' + ge + '\t' + mdr_text[furthest_mdr] + '\t' + str(furthest_dis) + '\n')
            generic_mdrSet[ge].remove(furthest_mdr)

    ed.close()