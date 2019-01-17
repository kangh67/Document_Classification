from nlp_MAUDE import *
from sklearn.linear_model import LogisticRegression

# OUTPUT
identified_HIT = 'MAUDE_data/Identified_HIT.txt'
summary_all = 'MAUDE_data/summary.txt'

EMB_DIM = 300
trainable = True

MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000
MAX_SENTS = 20
MAX_SENT_LENGTH = 100

# === Tokenization ===
print('=== Tokenization on training set===')
doc_hie_all, y_all, doc_all = read_MAUDE_hierarchical_simple()
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(doc_all)
word_index = tokenizer.word_index
print("Total unique tokens: " + str(len(word_index)))

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

# ========== Prepare inclusion and exclusion keywords, filter function
# Inclusion and exclusion keywords
in_gen = 'data/inclusion_generic.txt'
ex_gen = 'data/exclusion_generic.txt'
in_manu = 'data/inclusion_manu.txt'

# dict of inclusion and exclusion keywords
inclusion_generic = dict()
exclusion_generic = dict()
inclusion_manu = dict()

# inclusion_generic
with open(in_gen, 'r') as d:
    while True:
        line = d.readline()
        if not line:
            break
        else:
            inclusion_generic[line.replace('\n', '').lower()] = 0
# exclusion_generic
with open(ex_gen, 'r') as d:
    while True:
        line = d.readline()
        if not line:
            break
        else:
            exclusion_generic[line.replace('\n', '').lower()] = 0
# inclusion_manu
with open(in_manu, 'r') as d:
    while True:
        line = d.readline()
        if not line:
            break
        else:
            inclusion_manu[line.replace('\n', '').lower()] = 0

# Summary of inclusion and exclusion keywords
print('=== Summary of inclusion and exclusion keywords ===')
print('Inclusion keywords - generic names:', len(inclusion_generic))
print('Exclusion keywords - generic names:', len(exclusion_generic))
print('Inclusion keywords - manufacturer names:', len(inclusion_manu))

# Count generic and manu hit of filtered data
generic_inclu_hit = set()
manu_inclu_hit = set()


# Function of filter application
def apply_filter(line):
    brand = line.split('|')[6].lower()
    generic = line.split('|')[7].lower()
    manu = line.split('|')[8].lower()
    flag_in_ge = False
    flag_in_manu = False
    flag_ex = False

    # exclusion
    for keyword in exclusion_generic.keys():
        if re.search(r'\b' + keyword + r'\b', brand) or re.search(r'\b' + keyword + r'\b', generic):
            exclusion_generic[keyword] += 1
            flag_ex = True
    if flag_ex:
        return False

    # inclusion
    for keyword in inclusion_generic.keys():
        if re.search(r'\b' + keyword + r'\b', brand) or re.search(r'\b' + keyword + r'\b', generic):
            inclusion_generic[keyword] += 1
            flag_in_ge = True
            # Count generic inclusion keyword hit
            generic_inclu_hit.add(keyword)
    for keyword in inclusion_manu.keys():
        if re.search(r'\b' + keyword + r'\b', manu):
            inclusion_manu[keyword] += 1
            flag_in_manu = True
            # Count manu inclusion keyword hit
            manu_inclu_hit.add(keyword)

    return flag_in_ge or flag_in_manu


# ========== Interpret MAUDE data year by year
# MAUDE original data
dev_root = 'MAUDE_data/foidev'
text_root = 'MAUDE_data/foitext'

# Training data, 70% of 2008-2016 MAUDE data, used by baseline models
train_file = './data/MAUDE_train.tsv'

mdr_all = dict()
generic_all = dict()
brand_all = dict()
manu_all = dict()


def interpret_dev(year):
    mdr_current = dict()
    generic_current = dict()
    generic_new_current = dict()
    brand_current = dict()
    brand_new_current = dict()
    manu_current = dict()
    manu_new_current = dict()

    generic_inclu_hit.clear()
    manu_inclu_hit.clear()

    filtered_data = []
    filtered_mdr = set()
    filtered_gen = set()
    filtered_gen_new = set()
    filtered_brand = set()
    filtered_brand_new = set()
    filtered_manu = set()
    filtered_manu_new = set()

    with open(dev_root + year + '.txt', 'rb') as d:
        d.readline()
        while True:
            line = d.readline().decode('utf8', 'ignore')
            if not line:
                break
            else:
                l = line.split('|')
                if len(l) < 9:
                    continue
                mdr_all.setdefault(l[0], 0)  # mdr
                mdr_current.setdefault(l[0], 0)
                generic_all.setdefault(l[7], 0)  # generic name
                generic_current.setdefault(l[7], 0)
                brand_all.setdefault(l[6], 0)  # brand name
                brand_current.setdefault(l[6], 0)
                manu_all.setdefault(l[8], 0)  # manufacturer name
                manu_current.setdefault(l[8], 0)
                mdr_all[l[0]] += 1
                mdr_current[l[0]] += 1
                generic_all[l[7]] += 1
                generic_current[l[7]] += 1
                brand_all[l[6]] += 1
                brand_current[l[6]] += 1
                manu_all[l[8]] += 1
                manu_current[l[8]] += 1

                if generic_all[l[7]] == 1:
                    generic_new_current[l[7]] = 1
                if brand_all[l[6]] == 1:
                    brand_new_current[l[6]] = 1
                if manu_all[l[8]] == 1:
                    manu_new_current[l[8]] = 1

                if apply_filter(line) and l[0] not in filtered_mdr:
                    filtered_mdr.add(l[0])
                    filtered_gen.add(l[7])
                    if l[7] in generic_new_current.keys():
                        filtered_gen_new.add(l[7])
                    filtered_brand.add(l[6])
                    if l[6] in brand_new_current.keys():
                        filtered_brand_new.add(l[6])
                    filtered_manu.add(l[8])
                    if l[8] in manu_new_current.keys():
                        filtered_manu_new.add(l[8])
                    filtered_data_one = [year, l[0], l[7], l[6], l[8]]
                    filtered_data.append(filtered_data_one)
        summary = [year, len(mdr_current), len(filtered_data), len(generic_current), len(filtered_gen),
                   len(generic_new_current), len(filtered_gen_new), len(brand_current), len(filtered_brand),
                   len(brand_new_current), len(filtered_brand_new), len(manu_current), len(filtered_manu),
                   len(manu_new_current), len(filtered_manu_new), len(generic_inclu_hit), len(manu_inclu_hit)]

    # filtered_data[mdr# in a year][0:year, 1:mdr, 2:generic, 3:brand, 4:manu]
    # summary[0:year, 1:mdr#, 2:filtered_mdr#, 3:gen#, 4:filtered_gen#, 5:new_gen#, 6:filtered_new_gen#, 7:brand#,
    #           8:filtered_brand#, 9:new_brand#, 10:filtered_new_brand#, 11:manu#, 12:filtered_manu#,
    #           13:new_manu#, 14:filtered_new_manu#, 15:activated_gene_key#, 16:activated_manu_key#]
    return filtered_data, summary


def interpret_text(filtered_data, year):
    text_data = dict()
    # get all mdr
    for entry in filtered_data:
        text_data[entry[1]] = 'N/A'
    # match text for each mdr
    with open(text_root + year + '.txt', 'rb') as d:
        d.readline()
        while True:
            line = d.readline().decode('utf8', 'ignore')
            if not line:
                break
            else:
                line = line.split('|')
                if line[0] in text_data.keys():
                    if text_data[line[0]] == 'N/A':
                        text_data[line[0]] = line[5].replace('\n', '').replace('\r', '')
                    else:
                        text_data[line[0]] += ' >< ' + line[5].replace('\n', '').replace('\r', '')
    # before 1997
    if year == '1997':
        with open(text_root + '1996.txt', 'rb') as d:
            d.readline()
            while True:
                line = d.readline().decode('utf8', 'ignore')
                if not line:
                    break
                else:
                    line = line.split('|')
                    if line[0] in text_data.keys():
                        if text_data[line[0]] == 'N/A':
                            text_data[line[0]] = line[5].replace('\n', '').replace('\r', '')
                        else:
                            text_data[line[0]] += ' >< ' + line[5].replace('\n', '').replace('\r', '')
        with open(text_root + 'thru1995.txt', 'rb') as d:
            d.readline()
            while True:
                line = d.readline().decode('utf8', 'ignore')
                if not line:
                    break
                else:
                    line = line.split('|')
                    if line[0] in text_data.keys():
                        if text_data[line[0]] == 'N/A':
                            text_data[line[0]] = line[5].replace('\n', '').replace('\r', '')
                        else:
                            text_data[line[0]] += ' >< ' + line[5].replace('\n', '').replace('\r', '')

    # append text to filtered_data
    for entry in filtered_data:
        entry.append(text_data[entry[1]])

    return filtered_data


def apply_classifier(filtered_data, summary):
    documents_sent = []
    documents = []
    for idx in range(len(filtered_data)):
        documents.append(filtered_data[idx][len(filtered_data[idx]) - 1])
        sentences = tokenize.sent_tokenize(filtered_data[idx][len(filtered_data[idx]) - 1])
        documents_sent.append(sentences)
    documents = np.asarray(documents)

    # sequence input, for CNN, RNN, and RNN_att models
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

    # HIT # identified by the hybrid classifier
    summary.append(int(((pred_CNN + pred_H_RNN + pred_lr) / 3).round().sum(axis=0)[1]))

    # prediction result
    res_CNN = pred_CNN.round().T[1]
    res_H_RNN = pred_H_RNN.round().T[1]
    res_LR = pred_lr.round().T[1]
    res_hybrid = ((pred_CNN + pred_H_RNN + pred_lr) / 3).round().T[1]

    # add prediction result to filtered_data
    for idx in range(len(filtered_data)):
        filtered_data[idx].append(int(res_CNN[idx]))
        filtered_data[idx].append(int(res_H_RNN[idx]))
        filtered_data[idx].append(int(res_LR[idx]))
        filtered_data[idx].append(int(res_hybrid[idx]))

    return filtered_data, summary


def write_identified_HIT(filtered_data):
    for entry in filtered_data:
        if entry[len(entry) - 1] == 1:
            with open(identified_HIT, 'a+') as w:
                w.write(entry[0] + '\t' + entry[1] + '\t' + entry[2] + '\t' + entry[3] + '\t' + entry[4]
                        + '\t' + entry[5] + '\n')


def write_summary(summary_list):
    with open(summary_all, 'w') as w:
        w.write('YEAR\tMDR\tMDR_filtered\tGEN\tGEN_filtered\tNEW_GEN\tNEW_GEN_filtered\tBRAND\tBRAND_filtered\t'
                'NEW_BRAND\tNEW_BRAND_filtered\tMANU\tMANU_filtered\tNEW_MANU\tNEW_MANU_filtered\tACTIVATED_GEN\t'
                'ACTIVATED_MANU\tIDENTIFIED_HIT\n')
        overall = dict()
        for s in summary_list:
            for idx in range(len(s)):
                if idx == 0:
                    w.write(s[idx])
                else:
                    w.write('\t' + str(s[idx]))
                    if idx in overall.keys():
                        overall[idx] += s[idx]
                    else:
                        overall[idx] = s[idx]
            w.write('\n')
        w.write('Overall')
        for i in range(len(summary_list[0]) - 1):
            w.write('\t' + str(overall[i + 1]))
        w.write('\n')


summary_list = []
with open(identified_HIT, 'a+') as w:
    w.write('YEAR\tMDR\tGENERIC_NAME\tBRAND_NAME\tMANUFACTURER_NAME\tREPORT\n')

# Before 1997
print('=== thru1997 ===')
filtered_data, summary = interpret_dev('thru1997')  # apply filter
filtered_data = interpret_text(filtered_data, '1997')  # add text to filtered_data
filtered_data, summary = apply_classifier(filtered_data, summary)  # generate text for applying the classifier
write_identified_HIT(filtered_data)  # write identified reports
summary_list.append(summary)
print('YEAR\tMDR\tMDR_filtered\tGEN\tGEN_filtered\tNEW_GEN\tNEW_GEN_filtered\tBRAND\tBRAND_filtered\tNEW_BRAND\t'
      'NEW_BRAND_filtered\tMANU\tMANU_filtered\tNEW_MANU\tNEW_MANU_filtered\tACTIVATED_GEN\tACTIVATED_MANU\tHIT')
print(summary)


# After 1997
for i in range(1998, 2018):
    print('===', i, '===')
    filtered_data, summary = interpret_dev(str(i))  # apply filter
    filtered_data = interpret_text(filtered_data, str(i))  # add text to filtered_data
    filtered_data, summary = apply_classifier(filtered_data, summary)  # generate text for applying the classifier
    write_identified_HIT(filtered_data)  # write identified reports
    summary_list.append(summary)
    print('YEAR\tMDR\tMDR_filtered\tGEN\tGEN_filtered\tNEW_GEN\tNEW_GEN_filtered\tBRAND\tBRAND_filtered\tNEW_BRAND\t'
          'NEW_BRAND_filtered\tMANU\tMANU_filtered\tNEW_MANU\tNEW_MANU_filtered\tACTIVATED_GEN\tACTIVATED_MANU\tHIT')
    print(summary)


write_summary(summary_list)


