# -*- coding: utf-8 -*-
# @Time    : 2022/8/25 16:29
# @Author  : Z.Wang
# @Email   : w8614@hotmail.com
# @File    : training.py
import time
import numpy as np
import pandas as pd
pd.set_option('expand_frame_repr', False)
import matplotlib.pyplot as plt
plt.style.use('seaborn-ticks')
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = 11.0
plt.rcParams["figure.figsize"] = (10, 6)
import joblib
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn import model_selection, tree, svm
from keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

from utils import util

# Classes of nrrd Dataset
cases = ['liver', 'kidney', 'spleen', 'muscle', 'bone']
# Create model directly form sklearn, such as: Logistic Regression, Gaussian NB, Decision Tree , SVM, AdaBoost and MLP
model_df = {'LR': LogisticRegression(penalty='l2', C=1,
                                            multi_class='multinomial', solver='lbfgs', random_state=1),
            'NB': GaussianNB(),  'DT': tree.DecisionTreeClassifier(),
            'SVM': svm.SVC(decision_function_shape='ovo'), 'ADA': AdaBoostClassifier(n_estimators=100),
            'MLP': MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)}

# This stage takes training of the selected machine learning models and save the training hyperparameters.
def trainingIKIM(imagePath, maskPath, modelName):
    # Must call before using the dataset
    X_std, y = util.dataPreprocessing(imagePath, maskPath)

    # Random permutation cross-validator and  yields indices to split data into training and test sets.
    cv = model_selection.ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
    clf = model_df[modelName]
    bestScore = 0
    for train, test in cv.split(X_std, y):
        tmpModel = clf.fit(X_std[train], y[train])
        score = tmpModel.score(X_std[test], y[test])
        if score > bestScore:
            bestModel = tmpModel

    # Local path to trained weights file
    joblib.dump(bestModel, f'../weights/{modelName}.pkl')

# This stage evaluates the trained model with auc on test subsets.
def evaluateIKIM(imagePath, maskPath, modelName):
    # Must call before using the dataset
    X_std, y = util.dataPreprocessing(imagePath, maskPath)
    # Random permutation cross-validator and  yields indices to split data into training and test sets.
    cv = model_selection.ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
    clf = LogisticRegression(penalty='l2', C=1, multi_class='multinomial', solver='lbfgs', random_state=1)

    # Draw Area Under Curve
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    plt.xticks(np.arange(0, 1.1, step=0.1))
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.rcParams['figure.figsize'] = (8.0, 8.0)
    # Load weights
    model = joblib.load(f'../weights/{modelName}.pkl')
    for train, test in cv.split(X_std, y):
        # Run object detection
        prediction = clf.fit(X_std[train], y[train]).predict(X_std[test])
        # Compute ROC curve and area the curve
        y_true = to_categorical(y[test])
        y_pred = to_categorical(prediction)

        fpr, tpr, thresholds = roc_curve(np.array(y_true).ravel(), np.array(y_pred).ravel())
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        optimal_th, optimal_point = util.findOptimalCutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.8, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray', alpha=.6)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0, 1])
    plt.ylim([-0, 1])
    plt.xlabel('1-Specificity', fontsize='x-large')
    plt.ylabel('Sensitivity', fontsize='x-large')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right", fontsize='medium')
    plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
    plt.text(optimal_point[0], optimal_point[1], f'Threshold:{optimal_th:.2f}')

    plt.show()

# This stage performs for classfying the test dataset by using prediction and confusion matrix.
def testIKIM(imagePath, maskPath, modelName):
    # Must call before using the dataset
    X_data, y_data = util.dataPreprocessing(imagePath, maskPath)
    # Load weights
    model = joblib.load(f'../weights/{modelName}.pkl')

    y_pred = model.predict(X_data)
    score = accuracy_score(y_data, y_pred)

    # Grid of true class and their predictions
    conf_mtx = confusion_matrix(y_data, y_pred)
    util.plot_confusion_matrix(conf_mtx, np.unique(y_data))

    # Display results
    predictions = [cases[p] for p in y_pred]

    print(f'[INFO] The classes of the test dataset are:{[cases[p] for p in y_data]}.')
    print(f'[INFO] The corresponding predictions of the test dataset are:{predictions}')
    return score

if __name__ == '__main__':
    tt = time.time()

    # Directory to images and masks
    imagePath = r'../data/images'
    maskPath = r'../data/masks'
    # Abbreviation of the typical machine learning model.
    modelNames = ['LR', 'NB', 'DT', 'ADA', 'SVM', 'MLP']
    # performance of the utilized model.
    modelScores = []
    for i, m in enumerate(modelNames):
        # trainingIKIM(imagePath, maskPath, m)

        s = testIKIM('../data/test/images', '../data/test/masks', m)
        modelScores.append(s)
    print(f'[INFO] The model is {modelNames[modelScores.index(max(modelScores))]} with accuracy {max(modelScores)}.')

    print('[INFO] Time used: {} sec'.format(time.time() - tt))