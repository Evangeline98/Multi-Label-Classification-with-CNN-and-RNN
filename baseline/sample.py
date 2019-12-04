import argparse
from itertools import islice
import json
from pathlib import Path
import shutil
import warnings
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.exceptions import UndefinedMetricWarning
import torch
from torch import nn, cuda
from torch.optim import Adam
import tqdm
import torchvision

import models
from dataset import TrainDataset, TTADataset, get_ids, N_CLASSES, DATA_ROOT
from transforms import train_transform, test_transform
from utils import (
    write_event, load_model, mean_df, ThreadingDataLoader as DataLoader,
    ON_KAGGLE)


def sampler(size, batch):
    '''
    size: number of samples in train and validation set
    '''
    folds = pd.read_csv('folds.csv')
    train_root = DATA_ROOT / ('train')
    train_fold = folds[folds['fold'] != 0]
    valid_fold = folds[folds['fold'] == 0]
    if size:
        train_fold = train_fold[:size]
        valid_fold = valid_fold[:size]

    def make_loader(df: pd.DataFrame, image_transform) -> DataLoader:
        data =  DataLoader(
            TrainDataset(train_root, df, image_transform),
            batch_size = batch,
            shuffle=True,
            num_workers=2,
        )
        #print(data.batch_size)
        return data


    train_loader = make_loader(train_fold, train_transform)
    valid_loader = make_loader(valid_fold, test_transform)
    return train_loader,  valid_loader

#if __name__ == '__main__':
#    train, valid = sampler(100,10)
#    for i ,(input,targets) in enumerate(train):
#        if i == 0:
#            print(input.numpy().shape)
#            break
