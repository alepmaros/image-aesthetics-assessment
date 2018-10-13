import numpy as np
import pandas as pd
import multiprocessing
import cv2

import gist

def gist_from_df(df):
    # print('a')
    count = 0
    for index, row in df.iterrows():
        if (count % 50 == 0):
            print('Process {}: {}/{}'.format(multiprocessing.current_process(), count, df.shape[0]))

        p_img = 'datasets/photonet/imgs/{}.jpg'.format(row['photo_id'])
        img = cv2.imread(p_img)
        descriptor = gist.extract(img)

        for i in range (0,960):
            df.at[index, 'gist'+str(i)] = descriptor[i]

        count += 1

    print('{} done'.format(multiprocessing.current_process()))
    return df
    

image_csv = pd.read_csv('datasets/photonet/features_r.csv', dtype={'photo_id': str})

for i in range (0, 960):
    column = 'gist'+str(i)
    image_csv[column] = None

num_processes = multiprocessing.cpu_count()-1
chunk_size = int(image_csv.shape[0]/num_processes)

chunks = [image_csv.ix[image_csv.index[i:i + chunk_size]] for i in range(0, image_csv.shape[0], chunk_size)]

print(num_processes)
pool = multiprocessing.Pool(processes=num_processes)

result = pool.map(gist_from_df, chunks)

for i in range(len(result)):
   image_csv.ix[result[i].index] = result[i]

image_csv.to_csv('datasets/photonet/features_r_gist.csv', index=False)