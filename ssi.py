
'''
steps:
1. data normalizer steps
2. 
'''


import sklearn
import sklearn.feature_extraction.text
import sklearn.svm
import random
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

def bagOfWords(files_data):
    """
    Converts a list of strings (which are loaded from files) to a BOW representation of it
    parameter 'files_data' is a list of strings
    returns a `scipy.sparse.coo_matrix`
    """

    count_vector = sklearn.feature_extraction.text.CountVectorizer()
    ## get vocabilary: count_vector.get
    return count_vector.fit_transform(files_data)



def tfidf(files_data):
    '''
    converts a list of strings to 1) bow sparse matrix, 2) tfidf matrix
    :param files_data: a list of strings
    :return: a matrix
    '''
    bow = bagOfWords(files_data)
    return sklearn.feature_extraction.text.TfidfTransformer().fit_transform(bow)

# data = ['hello i am here', 'why is it so dark', 'the ship is sunk']
#
# bagOfWords(data)
# tfidf(data)


# def svm(data):
#     clf = svm.SVC(kernel='linear', probability=True)
#     clf.fit(X, y)

def trn_tst_split(data, train_ratio = 0.7):
    # data is a list of dictionary
    random.shuffle(data)
    trn_size = int(round(len(data)*train_ratio))
    return (data[:trn_size], data[trn_size:])

def svm_trn_tst(data):

    X = [x['text'] for x in data]
    X_tfidf = tfidf(X)
    y = [x['label'] for x in data]
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y, stratify=y, test_size=0.3)


    clf = sklearn.svm.SVC(kernel='linear', probability=True)
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)
    report = classification_report(y_test, y_pred)
    print report


def svm_cv(data):
    clf = sklearn.svm.SVC(kernel='linear', probability=True)
    random.shuffle(data)
    X = [x['text'] for x in data]
    X_tfidf = tfidf(X)
    y = [x['label'] for x in data]
    print cross_val_score(clf, X_tfidf, y, cv = 3)


def svm_cv_detail(data):
    clf = sklearn.svm.SVC(kernel='linear', probability=True)
    random.shuffle(data)
    X = [x['text'] for x in data]
    X_tfidf = tfidf(X)
    y = np.array([x['label'] for x in data])

    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.5)
    for train_idx, test_idx in sss.split(X_tfidf,y):
®        X_train_tfidf, X_test_tfidf = X_tfidf[train_idx], X_tfidf[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf = sklearn.svm.SVC(kernel='linear', probability=True)
        clf.fit(X_train_tfidf, y_train)

        y_pred = clf.predict(X_test_tfidf)
        report = classification_report(y_test, y_pred)
        print report



if __name__ == '__main__':
    import os
    import pandas as pd
    os.chdir('/Users/apple/Documents/practice')
    data = pd.read_csv('data.csv').to_dict('records')
    mode = 'train_cv'  # 'train_cv

    if mode == 'train_test':
        svm_trn_tst(data)
    if mode == 'train_cv':
        svm_cv_detail(data)
