import sys
sys.path.append("..")

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from src.sys_config import _RANDOM_SEED

imgs_csv = pd.read_csv('datasets/photonet/photonet_dataset_cleaned.csv')

imgs_train, imgs_test = train_test_split(imgs_csv, test_size=0.2, random_state=_RANDOM_SEED)
imgs_test, imgs_cv = train_test_split(imgs_test, test_size=0.5, random_state=_RANDOM_SEED)

with plt.style.context(('seaborn-darkgrid')):
    f = plt.figure()
    
    plt.title('Distribuição das médias das notas - Dataset Completo')
    plt.ylabel('Quantidade')
    plt.xlabel('Nota')
    plt.xlim(2,7)
    plt.hist(imgs_csv['mean_ratings'].values, bins=25, histtype='stepfilled')
    plt.show()

    

with plt.style.context(('seaborn-darkgrid')):
    f = plt.figure()
    
    plt.title('Distribuição das médias das notas - Dataset Treinamento')
    plt.ylabel('Quantidade')
    plt.xlabel('Nota')
    plt.xlim(2,7)
    plt.hist(imgs_train['mean_ratings'].values, bins=25, histtype='stepfilled')
    plt.show()

with plt.style.context(('seaborn-darkgrid')):
    f = plt.figure()
    
    plt.title('Distribuição das médias das notas - Dataset Teste')
    plt.ylabel('Quantidade')
    plt.xlabel('Nota')
    plt.xlim(2,7)
    plt.hist(imgs_test['mean_ratings'].values, bins=25, histtype='stepfilled')
    plt.show()