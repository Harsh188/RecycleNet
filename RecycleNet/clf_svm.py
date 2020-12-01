import os
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from feature_extraction import extract_features
import config

import pickle

date_string = time.strftime("%Y-%m-%d-%H:%M")

FILE_NAME = 'resnet_svm_' + date_string + '.sav'

timestr = time.strftime("%Y-%m-%d_%H:%M:%S")

def plot_cm(clf_svm, X_test, y_test):
    np.set_printoptions(precision=2)

    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", "true")]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf_svm, X_test, y_test,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.savefig(config.GRAPHDIR_CM+timestr)
    plt.show()

def plot_cf(y_pred, y_test):
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

    print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

    print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

    print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))

def train_svm():
    print(FILE_NAME)
    svm_features, svm_labels = extract_features(config.TRAIN_DIR,config.sample_count)
    X = svm_features.reshape(config.sample_count, config.OUTPUT_SIZE)
    y = svm_labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf_svm = SVC(C=config.C, kernel='rbf', gamma=config.GAMMA)

    print("Fitting the model...")
    clf_svm.fit(X_train, y_train)

    os.makedirs(os.path.dirname(config.TRAINED_MODEL_DIR_SVM+FILE_NAME), exist_ok=True)
    pickle.dump(clf_svm, open(config.TRAINED_MODEL_DIR_SVM+FILE_NAME, 'wb'))

    y_pred = clf_svm.predict(X_test)

    print("Model is trained. Plotting metrics...")
    plot_cf(y_pred, y_test)

    plot_cm(clf_svm, X_test, y_test)

def grid_search():
    print(FILE_NAME)
    svm_features, svm_labels = extract_features(config.TRAIN_DIR,config.sample_count)
    X = svm_features.reshape(config.sample_count, config.OUTPUT_SIZE)
    y = svm_labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # defining parameter range 
    param_grid = {'C': [0.1, 1, 10, 100, 1000],  
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                  'kernel': ['rbf']}  
      
    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
      
    # fitting the model for grid search 
    grid.fit(X_train, y_train)

    # print best parameter after tuning 
    print(grid.best_params_) 
      
    # print how our model looks after hyper-parameter tuning 
    print(grid.best_estimator_) 

    grid_predictions = grid.predict(X_test) 
  
    # print classification report 
    print(classification_report(y_test, grid_predictions)) 

if __name__ == '__main__':
    print('Now training the extracted features on SVM...')
    train_svm()
    # grid_search()