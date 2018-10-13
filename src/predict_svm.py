import multiprocessing, os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.utils import resample

from imblearn import under_sampling

from scipy.stats import ks_2samp

from FeatureExtractor.ScalerBOVW import ScalerBOVW

np.set_printoptions(suppress=True)

CUR_DIR = '/home/apm/git/image-aesthetics-assessment/'

if __name__ == '__main__':
    
    dataset = pd.read_csv('datasets/photonet/features_r.csv', dtype={'photo_id': str}).sample(frac=0.5, random_state=42)
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.fillna(0)

    dataset['quality'] = np.where(dataset['mean_ratings'] >= 5.5, 1, 0)

    df_majority = dataset[dataset.quality==0]
    df_minority = dataset[dataset.quality==1]
    df_majority_downsampled = resample(df_majority, replace=False, n_samples=df_minority.shape[0], random_state=42)
 
    # Combine minority class with downsampled majority class
    dataset = pd.concat([df_majority_downsampled, df_minority])

    y = dataset.quality
    X = dataset.drop(['quality', 'nb_aesthetics_ratings', 'index', 'mean_ratings'], axis=1)
    X = X.drop(['photo_id'], axis=1)

    # tl = under_sampling.RandomUnderSampler()
    # X, y = tl.fit_sample(X, y)

    print('X size:', X.shape)
    print('Y size:', y.shape)

    # pipe = Pipeline([
    #     ('bovw', ScalerBOVW()),
    #     ('scaler', StandardScaler()),
    #     ('model', SVC(probability=True, class_weight='balanced'))
    # ])

    l_acc = []

    kf = StratifiedKFold(n_splits=5)
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # BOVW
        # bovw_scaler = ScalerBOVW(CUR_DIR)
        # bovw_scaler.fit(X_train)
        # X_train = bovw_scaler.transform(X_train)
        # X_test  = bovw_scaler.transform(X_test)
        # del bovw_scaler

        # Standard Scaler
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)  

        # model = SVC(probability=True, class_weight='balanced', random_state=42)
        print('Fitting')
        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        l_acc.append(accuracy_score(y_test, y_pred))
        print('ACC', l_acc)
        

    # scoring = {
    #     'acc': 'accuracy',
    #     'bal_acc': 'balanced_accuracy',
    #     'f1': 'f1',
    #     'f1_weighted': 'f1_weighted'
    # }

    # rf = SVC()
    # scores = cross_validate(pipe, X, y, cv=10, scoring=scoring, 
    #     n_jobs=multiprocessing.cpu_count()-1)

    # print("Avg ACC: %0.3f (+/- %0.3f)" % (scores['test_acc'].mean(), scores['test_acc'].std() * 2))
    # print("Avg Balanced ACC: %0.3f (+/- %0.3f)" % (scores['test_bal_acc'].mean(), scores['test_bal_acc'].std() * 2))
    # print("Avg F1: %0.3f (+/- %0.3f)" % (scores['test_f1'].mean(), scores['test_f1'].std() * 2))
    # print("Avg F1 Weighted: %0.3f (+/- %0.3f)" % (scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std() * 2))

    
    # plt.style.use('seaborn-whitegrid')
    # plt.boxplot(scores['test_acc'], labels=['SVM'])
    # plt.ylabel('Accuracy')
    # plt.title('Model accuracy')
    # plt.show()
