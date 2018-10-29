
import numpy as np
import pandas as pd
import os

import tensorflow as tf

_base_path = '/data/alexandremaros/git/image-aesthetics-assessment'
imgs = pd.read_csv('datasets/photonet/photonet_dataset_cleaned.csv')

def parse_data(filename):
    image = tf.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    count = 0
    fn = tf.placeholder(dtype=tf.string)
    img = parse_data(fn)

    for index, row in imgs.iterrows():
        try:
            img_path = 'datasets/photonet/imgs/{}.jpg'.format(row['photo_id'])
            # print(img_path)
            # input()
            sess.run(img, feed_dict={fn: img_path})
        except Exception as e:
            print(row['photo_id'], "failed to load !")
            print()
            imgs.drop(index, inplace=True)
            count += 1

    print(count, "images failed to load !")

imgs.to_csv('datasets/photonet/photonet_cleaned_tf.csv', index=False)
print("All done !")