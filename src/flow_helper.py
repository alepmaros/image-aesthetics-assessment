import pandas as pd
import shutil, os
import numpy as np

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

CUR_DIR = '/home/apm/git/image-aesthetics-assessment/'

def downsample_df(df):
    df_minority = dataset[dataset.quality==0]
    df_majority = dataset[dataset.quality==1]

    # Swap
    if (df_minority.shape[0] > df_majority.shape[0]):
        df_minority, df_majority = df_majority, df_minority

    df_majority_downsampled = resample(df_majority, replace=False,
                                n_samples=df_minority.shape[0])

    return pd.concat([df_minority, df_majority_downsampled])

dataset = pd.read_csv('datasets/photonet/photonet_dataset_cleaned.csv', dtype={'photo_id': str})
dataset = dataset.drop_duplicates(subset=['photo_id'])

dataset['quality'] = np.where(dataset['mean_ratings'] > 5.5, 1, 0)



dataset = dataset.sample(frac=1)

# df = downsample_df(dataset)
df = dataset

train,test = train_test_split(df, test_size=0.2, stratify=df.quality)
validate, test = train_test_split(test, test_size=0.5, stratify=test.quality)

print('Train 0:', train[train.quality==0].shape)
print('Train 1:', train[train.quality==1].shape)

print ('Train Shape:', train.shape)
print(validate.shape)
print(test.shape)

dic_input = {
    'train': train,
    'validate': validate,
    'test': test
}

folders = ['train', 'validate', 'test']
labels = ['0', '1']

for folder in folders:
    for label in labels:
        directory = 'datasets/photonet_flow/{}/{}/'.format(folder, label)
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            shutil.rmtree(directory)
            os.makedirs(directory)

    directory = 'datasets/photonet_flow/{}/'.format(folder)
    for index, row in dic_input[folder].iterrows():
        current_image = 'datasets/photonet/imgs/{}.jpg'.format(row['photo_id'])
        current_image = os.path.join(CUR_DIR, current_image)

        dst_image = os.path.join(directory, '{}/{}.jpg'.format(row['quality'], row['photo_id']))
        dst_image = os.path.join(CUR_DIR, dst_image)

        os.symlink(current_image, dst_image)