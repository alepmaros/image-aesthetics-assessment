import pandas as pd
import shutil, os
import numpy as np

from sklearn.utils import resample

dataset = pd.read_csv('datasets/photonet/photonet_dataset_cleaned.csv')
dataset['quality'] = np.where(dataset['mean_ratings'] > 5.0, 1, 0)

dataset = dataset.sample(frac=1)

df_majority = dataset[dataset.quality==0]
df_minority = dataset[dataset.quality==1].sample(frac=0.1)

df_majority_downsampled = resample(df_majority, 
                            replace=False,
                            n_samples=df_minority.shape[0])

dataset = pd.concat([df_minority, df_majority_downsampled])

if not os.path.exists('datasets/photonet_flow/train/0/'):
    os.makedirs('datasets/photonet_flow/train/0/')

if not os.path.exists('datasets/photonet_flow/train/1/'):
    os.makedirs('datasets/photonet_flow/train/1/')

i = 0
for index, row in dataset.iterrows():
    print('{}/{} {}%'.format(i, dataset.shape[0], round(i*100/dataset.shape[0],4)))
    current_image = 'datasets/photonet/imgs/{}.jpg'.format(row['photo_id'])
    dst_image = 'datasets/photonet_flow/train/{}/{}.jpg'.format(row['quality'], row['photo_id'])
    shutil.copy(current_image, dst_image)
    i += 1