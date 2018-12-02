import sys
sys.path.append("..")

import os, pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

with open('trained_models/loss_baseline.txt', 'rb') as fhandle:
    loss_baseline = pickle.load(fhandle)

with open('trained_models/val_loss_baseline.txt', 'rb') as fhandle:
    val_loss_baseline = pickle.load(fhandle)

with open('trained_models/loss_eml.txt', 'rb') as fhandle:
    loss_eml = pickle.load(fhandle)

with open('trained_models/val_loss_eml.txt', 'rb') as fhandle:
    val_loss_eml = pickle.load(fhandle)

with plt.style.context(('seaborn-darkgrid')):
    f = plt.figure(figsize=(4.5,3.5))

    plt.plot(np.arange(0, len(loss_baseline)), loss_baseline, label='Loss')
    plt.plot(np.arange(0, len(val_loss_baseline)), val_loss_baseline, label='Validation Loss'  )

    plt.title('Cross entropy')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, len(loss_baseline)+1, 2.0))

    plt.legend()
    plt.tight_layout()
    plt.savefig('experiments/loss_cross.pdf')

with plt.style.context(('seaborn-darkgrid')):
    f = plt.figure(figsize=(4.5,3.5))

    plt.plot(np.arange(0, len(loss_eml)), loss_eml, label='Loss')
    plt.plot(np.arange(0, len(val_loss_eml)), val_loss_eml, label='Validation Loss'  )

    plt.title('EMD')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, len(loss_eml)+1, 2.0))

    plt.legend()
    plt.tight_layout()
    plt.savefig('experiments/loss_eml.pdf')