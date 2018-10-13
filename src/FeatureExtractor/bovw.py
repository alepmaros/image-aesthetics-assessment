import numpy as np
import pandas as pd
import os, cv2, time


image_csv = pd.read_csv('datasets/photonet/features_r.csv', dtype={'photo_id':str}).sample(n=100)
image_csv2 = pd.read_csv('datasets/photonet/features_r.csv', dtype={'photo_id':str})

# Create Sift Extractor
sift = cv2.xfeatures2d.SIFT_create()

dictionarySize = 16

BOW = cv2.BOWKMeansTrainer(dictionarySize)

for _, row in image_csv.iterrows():
    p_img = 'datasets/photonet/imgs/{}.jpg'.format(row['photo_id'])
    try:
        print(p_img)
        image = cv2.imread(p_img)
        # gray = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        kp, dsc= sift.detectAndCompute(image, None)
        BOW.add(dsc)
    except Exception as e:
        print('Failed to use image {}: {}'.format(p_img, e))

#dictionary created
sift2 = cv2.xfeatures2d.SIFT_create()
BOW_descriptor = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
BOW_descriptor.setVocabulary(BOW.cluster())


### GET THE DESCRIPTORS
for _, row in image_csv2.iterrows():
    p_img = 'datasets/photonet/imgs/{}.jpg'.format(row['photo_id'])
    
    try:
        print(p_img)
        image = cv2.imread(p_img)
        # gray = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        kp, dsc= sift.detectAndCompute(image, None)
        print(BOW_descriptor.compute(image, kp))
    except Exception as e:
        print('Failed to use image {}: {}'.format(p_img, e))


