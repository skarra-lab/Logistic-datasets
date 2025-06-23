import sys
import pandas as pd
import numpy as np
import scipy.optimize as opt
import pylab as pl
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools

#load our data
df = pd.read_csv("cell_samples.csv")
print(df.head())

#preprocessing and selection
for col in ["UnifSize","UnifShape","MargAdh","SingEpiSize","BareNuc","BlandChrom","NormNucl","Mit"]: #converts columns to numeric
    df[col] = pd.to_numeric(df[col], errors='coerce') 

df.dropna(inplace=True) #drops rows with missing values
X = np.asarray(df[["UnifSize","UnifShape","MargAdh","SingEpiSize","BareNuc","BlandChrom","NormNucl","Mit"]])
y = np.asarray(df["Class"])
#normalize the data
X = preprocessing.StandardScaler().fit(X).transform(X)

#train / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#modelling
lr = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
print(lr)

#predicting
pred_lr = lr.predict(X_test)
print(pred_lr)

proba_lr = lr.predict_proba(X_test)
print(proba_lr)

jaccard_score(y_test, pred_lr, average='macro')

print(classification_report(y_test, pred_lr))
print(log_loss(y_test, proba_lr))

#compute a confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap = plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized Confusion matrix')
    else:
        print('Confusion matrix without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment = 'center', color = 'white' if cm[i, j] > thresh else 'black')
            plt.tight_layout()
            plt.ylabel('True label')
            plt.ylabel('Predicted label')

cnf_matrix = confusion_matrix(y_test, pred_lr, labels=[2,4])
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['benign=2', 'malignant=4'], normalize=False, title='Confusion matrix')
plt.show()