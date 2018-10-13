import cv2
import numpy as np
import pandas as pd
import os, time

from sklearn.model_selection import train_test_split

class ScalerBOVW:
    def __init__(self, directory):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.dictionarySize = 32
        self.directory = directory

        self.BOW = cv2.BOWKMeansTrainer(self.dictionarySize)
        self.BOW_descriptor = cv2.BOWImgDescriptorExtractor(self.sift, cv2.BFMatcher(cv2.NORM_L2))
    
    def fit(self, X, y=None):
        print('Fit Started')
        start = time.time()
        for _, row in X.iterrows():
            p_img = 'datasets/photonet/imgs/{}.jpg'.format(row['photo_id'])
            p_img = os.path.join(self.directory, p_img)
            image = cv2.imread(p_img)
            # gray = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            kp, dsc= self.sift.detectAndCompute(image, None)
            try:
                self.BOW.add(dsc)
            except Exception as e:
                print(e)
                print(p_img)
        print('Took {} seconds to add descriptors'.format(time.time()-start))
        start = time.time()
        self.BOW_descriptor = cv2.BOWImgDescriptorExtractor(self.sift, cv2.BFMatcher(cv2.NORM_L2))
        self.BOW_descriptor.setVocabulary(self.BOW.cluster())
        print('Took {} seconds to set vocab'.format(time.time()-start))

        print('Fit Done')
        return self

    def transform(self, X):
        X = X.copy()

        for i in range (0, self.dictionarySize):
            column = 'bovw'+str(i)
            X[column] = 0

        for index, row in X.iterrows():
            p_img = 'datasets/photonet/imgs/{}.jpg'.format(row['photo_id'])
            p_img = os.path.join(self.directory, p_img)
            # print(p_img)
            image = cv2.imread(p_img)
            # gray = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            kp, dsc= self.sift.detectAndCompute(image, None)
            descriptor = self.BOW_descriptor.compute(image, kp)
            # print(descriptor)
            for i in range(0, self.dictionarySize):
                try:
                    X.loc[index, 'bovw'+str(i)] = descriptor[0][i]
                except Exception as e:
                    print(X)
                    print(e)
                    print(index)
                    print(descriptor)

        return X.drop(['photo_id'], axis=1)

# dataset = pd.read_csv('datasets/photonet/features_r.csv', dtype={'photo_id':str}).sample(n=100)

# dataset['quality'] = np.where(dataset['mean_ratings'] >= 5.5, 1, 0)
# y = dataset.quality
# X = dataset.drop(['quality', 'nb_aesthetics_ratings', 'index', 'mean_ratings'], axis=1)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# scaler = ScalerBOVW()
# scaler.fit(X_train)

# print(scaler.transform(X_train))