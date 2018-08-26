from keras import optimizers
from nlp_MAUDE import *
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import load_model
from additional_metrics import *

m = []
m.append('a')
m.append('b')
m.append('c')

n = []
n.append((1,2,3))
n.append((4,5,6))
n.append((7,8,9))


x = DataFrame(n)
x.index = m
x.index.name = 'model'
x.columns = ['ACC', 'REC', 'PRE']
x.columns.name = 'metrics'
x['new'] = m

print(x)

x = x.sort_values(by='ACC', ascending=False)
print(x)

x.to_csv('logs/test.csv')

a = [range(1, 10)]
print(a)
