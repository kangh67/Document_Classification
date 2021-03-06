import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc


# Choose the figure need to draw
# 1 = ROC of individual models
# 2 = ROC of hybrid models
# 3 = acc vs. f1 of all hybrid and individual models
mode = 2


if mode == 1 or mode == 2:
    df_2 = pd.read_csv('models/roc/roc_2-CNN-H_RNN-.csv')
    auc_2 = auc(df_2['fpr'], df_2['tpr'])

    df_3 = pd.read_csv('models/roc/roc_3-CNN-H_RNN-LR-.csv')
    auc_3 = auc(df_3['fpr'], df_3['tpr'])

    df_4 = pd.read_csv('models/roc/roc_4-NO-RNN-CNN-H_RNN-.csv')
    auc_4 = auc(df_4['fpr'], df_4['tpr'])

    df_5 = pd.read_csv('models/roc/roc_5-NO-H_RNN_att-SVM-.csv')
    auc_5 = auc(df_5['fpr'], df_5['tpr'])

    df_6 = pd.read_csv('models/roc/roc_6-NO-RNN-.csv')
    auc_6 = auc(df_6['fpr'], df_6['tpr'])

    df_7 = pd.read_csv('models/roc/roc_7_All.csv')
    auc_7 = auc(df_7['fpr'], df_7['tpr'])

    df_h_rnn_att = pd.read_csv('models/roc/roc_H_RNN_att.csv')
    auc_h_rnn_att = auc(df_h_rnn_att['fpr'], df_h_rnn_att['tpr'])

    df_h_rnn = pd.read_csv('models/roc/roc_H_RNN.csv')
    auc_h_rnn = auc(df_h_rnn['fpr'], df_h_rnn['tpr'])

    df_rnn_att = pd.read_csv('models/roc/roc_RNN_att.csv')
    auc_rnn_att = auc(df_rnn_att['fpr'], df_rnn_att['tpr'])

    df_rnn = pd.read_csv('models/roc/roc_RNN.csv')
    auc_rnn = auc(df_rnn['fpr'], df_rnn['tpr'])

    df_cnn = pd.read_csv('models/roc/roc_CNN.csv')
    auc_cnn = auc(df_cnn['fpr'], df_cnn['tpr'])

    df_lr = pd.read_csv('models/roc/roc_LR.csv')
    auc_lr = auc(df_lr['fpr'], df_lr['tpr'])

    df_svm = pd.read_csv('models/roc/roc_SVM.csv')
    auc_svm = auc(df_svm['fpr'], df_svm['tpr'])

    df_nb = pd.read_csv('models/roc/roc_NB.csv')
    auc_nb = auc(df_nb['fpr'], df_nb['tpr'])

    df_rf = pd.read_csv('models/roc/roc_RF.csv')
    auc_rf = auc(df_rf['fpr'], df_rf['tpr'])

ls = dict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

alpha = 0.8

if mode == 1:
    plt.figure(figsize=(10, 10))
    plt.plot(df_cnn['fpr'], df_cnn['tpr'], alpha=alpha, color='orange', lw=2, linestyle=ls['densely dashdotted'],
             label='CNN = %0.4f' % auc_cnn)
    plt.plot(df_lr['fpr'], df_lr['tpr'], alpha=alpha, color='darkblue', lw=2, linestyle=ls['densely dashdotdotted'],
             label='LR = %0.4f' % auc_lr)
    plt.plot(df_svm['fpr'], df_svm['tpr'], alpha=alpha, color='darkgreen', lw=2, linestyle=ls['densely dashed'],
             label='SVM = %0.4f' % auc_svm)
    plt.plot(df_h_rnn_att['fpr'], df_h_rnn_att['tpr'], alpha=alpha, color='blue', lw=2, linestyle=ls['solid'],
             label='H_RNN_att = %0.4f' % auc_h_rnn_att)
    plt.plot(df_rnn['fpr'], df_rnn['tpr'], alpha=alpha, color='green', lw=2, linestyle=ls['dashed'],
             label='RNN = %0.4f' % auc_rnn)
    plt.plot(df_h_rnn['fpr'], df_h_rnn['tpr'], alpha=alpha, color='red', lw=2, linestyle=ls['densely dotted'],
             label='H_RNN = %0.4f' % auc_h_rnn)
    plt.plot(df_rnn_att['fpr'], df_rnn_att['tpr'], alpha=alpha, color='purple', lw=2, linestyle=ls['dashdotted'],
             label='RNN_att = %0.4f' % auc_rnn_att)
    plt.plot(df_nb['fpr'], df_nb['tpr'], alpha=alpha, color='cyan', lw=2, linestyle=ls['dotted'],
             label='NB = %0.4f' % auc_nb)
    plt.plot(df_rf['fpr'], df_rf['tpr'], alpha=alpha, color='magenta', lw=2, linestyle=ls['loosely dashed'],
             label='RF = %0.4f' % auc_rf)

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=ls['dotted'])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('FP Rate', fontsize=20)
    plt.ylabel('TP Rate', fontsize=20)
    legend = plt.legend(loc='lower right', handlelength=10, borderpad=1, labelspacing=1.2, fontsize=12)
    legend.set_title('AUC Ranking (individual models)', prop={'size': 14})
    plt.show()

elif mode == 2:
    plt.figure(figsize=(10, 10))
    plt.plot(df_2['fpr'], df_2['tpr'], alpha=alpha, color='orange', lw=2, linestyle=ls['solid'],
             label='2-CNN+H_RNN = %0.4f' % auc_2)
    plt.plot(df_3['fpr'], df_3['tpr'], alpha=alpha, color='darkblue', lw=2, linestyle=ls['densely dashdotdotted'],
             label='3-CNN+H_RNN+LR = %0.4f' % auc_3)
    plt.plot(df_7['fpr'], df_7['tpr'], alpha=alpha, color='purple', lw=2, linestyle=ls['loosely dashed'],
             label='7-All = %0.4f' % auc_7)
    plt.plot(df_5['fpr'], df_5['tpr'], alpha=alpha, color='red', lw=2, linestyle=ls['densely dotted'],
             label='5-H_RNN+RNN_att+RNN+CNN+LR = %0.4f' % auc_5)
    plt.plot(df_6['fpr'], df_6['tpr'], alpha=alpha, color='cyan', lw=2, linestyle=ls['dashed'],
             label='6-(without)RNN = %0.4f' % auc_6)
    plt.plot(df_4['fpr'], df_4['tpr'], alpha=alpha, color='darkgreen', lw=2, linestyle=ls['densely dashed'],
             label='4-H_RNN_att+RNN_att+SVM+LR = %0.4f' % auc_4)

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=ls['dotted'])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('FP Rate', fontsize=20)
    plt.ylabel('TP Rate', fontsize=20)
    legend = plt.legend(loc='lower right', handlelength=10, borderpad=1, labelspacing=1.4, fontsize=12)
    legend.set_title('AUC Ranking (hybrid models)', prop={'size': 14})
    plt.show()

elif mode == 3:
    df_classic = pd.read_csv('models/metrics/classic_models.csv')
    df_deep = pd.read_csv('models/metrics/deep_learning_models.csv')
    df_hybrid = pd.read_csv('models/metrics/combined_models.csv')

    plt.figure(figsize=(10, 10))
    plt.scatter(df_classic['acc'][:2], df_classic['f1'][:2], alpha=0.8,
                color='red', marker='x', label='Classic Models')  # only SVM and LR
    plt.scatter(df_deep['Accuracy'], df_deep['F1 score'], alpha=0.8,
                color='blue', marker='v', label='Deep Learning Models')
    plt.scatter(df_hybrid['Accuracy'], df_hybrid['F1 score'], alpha=0.5,
                color='purple', marker='o', label='Hybrid Models')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Accuracy', fontsize=20)
    plt.ylabel('F1 Measure', fontsize=20)
    plt.legend(loc='lower right', handlelength=2, borderpad=1, labelspacing=1.2, fontsize=16)
    plt.show()

else:
    print('Wrong mode assigned!')
