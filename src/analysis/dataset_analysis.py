import pandas as pd
import numpy as np 
import cv2, os, shutil



imgs_csv = pd.read_csv('datasets/photonet/photonet_cleaned_tf.csv')

low_scores = imgs_csv.sort_values('mean_ratings').head(8)
high_scores = imgs_csv.sort_values('mean_ratings', ascending=False).head(8)

for index, row in low_scores.iterrows():
    print(row['photo_id'], row['mean_ratings'])
    path_in = 'datasets/photonet/imgs/{}.jpg'.format(row['photo_id'])
    path_out = 'datasets/selected_imgs/{}_{}.jpg'.format(str(row['mean_ratings']).replace('.', 'd'), row['photo_id'])
    shutil.copy(path_in, path_out)
    # img = cv2.imread('datasets/photonet/imgs/{}.jpg'.format(row['photo_id']))
    # cv2.imshow('dst_rt', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

for index, row in high_scores.iterrows():
    print(row['photo_id'], row['mean_ratings'])
    path_in = 'datasets/photonet/imgs/{}.jpg'.format(row['photo_id'])
    path_out = 'datasets/selected_imgs/{}_{}.jpg'.format(str(row['mean_ratings']).replace('.', 'd'), row['photo_id'])
    shutil.copy(path_in, path_out)
    # img = cv2.imread('datasets/photonet/imgs/{}.jpg'.format(row['photo_id']))
    # cv2.imshow('dst_rt', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()