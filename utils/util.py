# -*- coding: utf-8 -*-
# @Time    : 2022/8/25 23:13
# @Author  : Z.Wang
# @Email   : w8614@hotmail.com
# @File    : util.py
import time, sys, os, logging, six, radiomics
import numpy as np
import pandas as pd
pd.set_option('expand_frame_repr', False)
import matplotlib.pyplot as plt
plt.style.use('seaborn-ticks')
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = 11.0
plt.rcParams["figure.figsize"] = (10, 6)
from radiomics import featureextractor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

cases = ['liver', 'kidney', 'spleen', 'muscle', 'bone']

def classBalance(feature_df):
    feature_column = list(set(list(feature_df.columns)))
    class_label = pd.unique(np.array(list(feature_df['case'])))
    class_label.sort()
    data_balance = []
    outcome = feature_df['case']
    for l in class_label:
        data_balance.append(np.sum(np.array(list(outcome)) == l) / len(outcome))

    print('Number of observations: {}\nClass labels: {}\nClasses balance: {}'.format(len(outcome),
                                                                                     class_label,
                                                                                     data_balance))

# Initialize feature extractor using the settings file. The imagePath and maskPath are the corresponding nrrd file in
# direction image and mask, respectively. We used only the features with 'original' start and add a 'case' value for
# class to which it belongs.
def loadRawData(imagePath, maskPath):
    feat_dictionary, key_number = {}, 0
    extractor = featureextractor.RadiomicsFeatureExtractor()

    files = os.listdir(imagePath)
    for i, file in enumerate(files):
        case_id = cases.index([c for c in cases if c in file][0])
        pat_features = extractor.execute(os.path.join(imagePath, file),
                                   os.path.join(maskPath, file))
        if pat_features['diagnostics_Image-original_Hash'] != '':
            pat_features.update({'case':case_id})
            feat_dictionary[key_number] = pat_features
            key_number += 1

    # output_features = pd.DataFrame.from_dict(feat_dictionary).T.iloc[:,22:]
    pd_features = pd.DataFrame.from_dict(feat_dictionary).T
    drop_columns = [c for c in pd_features if 'diagnostics' in c]
    output_features = pd_features.drop(labels=drop_columns, axis=1)

    return output_features

# pre-processing for the raw data and the cleaned data will divided into features(X) and target(y).
def dataPreprocessing(imagePath, maskPath):
    feature_df = loadRawData(imagePath, maskPath)

    class_mapping = {label: idx for idx, label in enumerate(np.unique(feature_df['case']))}
    # print(class_mapping)
    feature_df['case'] = feature_df['case'].map(class_mapping)
    y = LabelEncoder().fit_transform(feature_df['case'].values)
    X = feature_df.iloc[:, :-1].values
    # print(y.shape, X.shape)
    std = StandardScaler()
    X_std = std.fit_transform(X)
    # print(X_std.shape)

    return X_std, y

def findOptimalCutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names)        #, rotation=45
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()

    width, height = cm.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), color='red', fontsize='11',
                         horizontalalignment='center',
                         verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

